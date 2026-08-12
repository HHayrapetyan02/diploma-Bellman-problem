import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from utils.geometry import from_pq


class HJBSubsolutionLP:
    def __init__(self, n_p=40, n_q=40, n_dirs=32, p_max=4.0, q_max=1.9,
                 p_min=1e-2, slack_weight=1e6, margin=0.0):
        self.n_p = n_p
        self.n_q = n_q
        self.n_dirs = n_dirs
        self.p_max = p_max
        self.q_max = q_max
        self.p_min = p_min
        self.slack_weight = slack_weight
        self.margin = margin

    def _grid(self):
        p = np.linspace(self.p_min, self.p_max, self.n_p)
        q = np.linspace(-self.q_max, self.q_max, self.n_q)
        P, Q = np.meshgrid(p, q, indexing="ij")
        mask = P >= Q**2
        return p, q, P, Q, mask

    @staticmethod
    def _box(P, Q, mask):
        from bounds.lower.square import LowerBoundBellmanFunction
        from bounds.upper.rectangle import UpperBoundBellmanFunction

        low = LowerBoundBellmanFunction()
        upp = UpperBoundBellmanFunction()

        lo = np.full(P.shape, np.nan)
        hi = np.full(P.shape, np.nan)
        for i, j in zip(*np.nonzero(mask)):
            x, y = from_pq(P[i, j], Q[i, j], ny=1.0)   
            hi[i, j] = low.lowerBoundBellman2D(x, y)
            lo[i, j] = upp.upperBoundBellman2DRectangle(x, y)
        return lo, hi

    def build_lp(self, box=None):
        p, q, P, Q, mask = self._grid()
        idx = -np.ones(P.shape, dtype=int)
        idx[mask] = np.arange(int(mask.sum()))
        n_var = int(mask.sum())

        dp = p[1] - p[0]
        dq = q[1] - q[0]

        ang = 2.0 * np.pi * np.arange(self.n_dirs) / self.n_dirs
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)

        rows, cols, vals, rhs = [], [], [], []
        row = 0
        cs = np.cos(np.pi / self.n_dirs)   

        for i in range(1, self.n_p - 1):
            for j in range(1, self.n_q - 1):
                if not mask[i, j]:
                    continue
                pi, qj = P[i, j], Q[i, j]
                here = idx[i, j]
                up_p, dn_p = idx[i + 1, j], idx[i - 1, j]
                up_q, dn_q = idx[i, j + 1], idx[i, j - 1]
                root = np.sqrt(max(pi - qj**2, 0.0))

                for e in dirs:
                    c_p = cs * 2.0 * qj - 4.0 * pi * e[0]
                    c_q = cs * 1.0 - 2.0 * qj * e[0] + root * e[1]
                    c_0 = 5.0 * e[0]

                    terms = {}

                    if c_p >= 0.0:
                        if up_p < 0:
                            continue
                        terms[up_p] = terms.get(up_p, 0.0) + c_p / dp
                        terms[here] = terms.get(here, 0.0) - c_p / dp
                    else:
                        if dn_p < 0:
                            continue
                        terms[here] = terms.get(here, 0.0) + c_p / dp
                        terms[dn_p] = terms.get(dn_p, 0.0) - c_p / dp

                    if c_q >= 0.0:
                        if up_q < 0:
                            continue
                        terms[up_q] = terms.get(up_q, 0.0) + c_q / dq
                        terms[here] = terms.get(here, 0.0) - c_q / dq
                    else:
                        if dn_q < 0:
                            continue
                        terms[here] = terms.get(here, 0.0) + c_q / dq
                        terms[dn_q] = terms.get(dn_q, 0.0) - c_q / dq

                    terms[here] = terms.get(here, 0.0) + c_0

                    for k, v in terms.items():
                        rows.append(row)
                        cols.append(int(k))
                        vals.append(float(v))
                    rows.append(row)          
                    cols.append(n_var)
                    vals.append(-1.0)
                    rhs.append(cs * (0.5 * pi - self.margin))
                    row += 1

        A_ub = sparse.coo_matrix((vals, (rows, cols)),
                                 shape=(row, n_var + 1)).tocsr()
        b_ub = np.array(rhs, dtype=float)

        c = np.zeros(n_var + 1)
        c[:n_var] = dp * dq                     
        c[n_var] = self.slack_weight

        lo, hi = self._box(P, Q, mask) if box is None else box
        bounds = [(float(lo[i, j]), float(hi[i, j]))
                  for i, j in zip(*np.nonzero(mask))] + [(0.0, None)]

        return A_ub, b_ub, c, bounds, idx, mask, p, q, (lo, hi)

    def solve(self, method="highs", box=None):
        A_ub, b_ub, c, bounds, idx, mask, p, q, (lo, hi) = self.build_lp(box)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)

        f = np.full(mask.shape, np.nan)
        slack = np.nan
        if res.success:
            f[mask] = res.x[:-1]
            slack = float(res.x[-1])
        return {"p": p, "q": q, "f": f, "mask": mask,
                "f_rect": lo, "f_square": hi, "slack": slack,
                "success": bool(res.success), "status": res.message}

    @staticmethod
    def verify(sol):
        from utils.reduced import hjb_residual

        p, q, f, mask = sol["p"], sol["q"], sol["f"], sol["mask"]
        dp, dq = p[1] - p[0], q[1] - q[0]
        worst = -np.inf
        for i in range(1, len(p) - 1):
            for j in range(1, len(q) - 1):
                if not (mask[i, j] and mask[i + 1, j] and mask[i - 1, j]
                        and mask[i, j + 1] and mask[i, j - 1]):
                    continue
                f_p = (f[i + 1, j] - f[i - 1, j]) / (2.0 * dp)
                f_q = (f[i, j + 1] - f[i, j - 1]) / (2.0 * dq)
                worst = max(worst, hjb_residual(p[i], q[j], f[i, j], f_p, f_q))
        sol["max_residual"] = float(worst)
        sol["certified"] = bool(worst <= 0.0 and sol.get("slack", 1.0) == 0.0)
        return sol["certified"], sol["max_residual"]

    @staticmethod
    def evaluate(sol, x, y):
        """omega-certificate at (x, y); -inf outside the computed grid."""
        from utils.geometry import to_pq

        pp, qq = to_pq(x, y)
        p, q, f = sol["p"], sol["q"], sol["f"]
        if not (p[0] <= pp <= p[-1] and q[0] <= qq <= q[-1]):
            return -np.inf

        i = int(np.clip(np.searchsorted(p, pp) - 1, 0, len(p) - 2))
        j = int(np.clip(np.searchsorted(q, qq) - 1, 0, len(q) - 2))
        tp = (pp - p[i]) / (p[i + 1] - p[i])
        tq = (qq - q[j]) / (q[j + 1] - q[j])

        block = f[i:i + 2, j:j + 2]
        if not np.all(np.isfinite(block)):
            finite = block[np.isfinite(block)]
            if finite.size == 0:
                return -np.inf
            val = float(np.max(finite))          
        else:
            val = float((1 - tp) * ((1 - tq) * block[0, 0] + tq * block[0, 1])
                        + tp * ((1 - tq) * block[1, 0] + tq * block[1, 1]))
        return float(np.linalg.norm(y) ** 5 * val)