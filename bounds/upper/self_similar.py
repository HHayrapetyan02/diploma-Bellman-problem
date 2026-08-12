import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bounds.common import degenerate_case
from utils.constants import Constants as Const
from utils.geometry import to_pq


class SelfSimilarControlBound:
    def __init__(self, rtol=1e-10, atol=1e-12):
        self.rtol = rtol
        self.atol = atol

    def _backward_trajectory(self, T, c, sigma, n_eval=400):
        def rhs(t, z):
            tau = max(T - t, 1e-14)
            phi = sigma * Const.SQRT5 * np.log(tau) + c
            u = np.array([np.cos(phi), np.sin(phi)])
            x, y = z[0:2], z[2:4]
            return np.concatenate([y, u, [0.5 * (x @ x)]])

        sol = solve_ivp(rhs, (T, 0.0), np.zeros(5),
                        t_eval=np.linspace(T, 0.0, n_eval),
                        rtol=self.rtol, atol=self.atol, method="DOP853")
        return sol

    def _invariants_at(self, T, c, sigma):
        sol = self._backward_trajectory(T, c, sigma)
        pq = []
        for k in range(sol.y.shape[1]):
            x, y = sol.y[0:2, k], sol.y[2:4, k]
            if np.linalg.norm(y) > 1e-12:
                pq.append(to_pq(x, y))
            else:
                pq.append((np.nan, np.nan))
        return sol, np.array(pq)

    def value(self, x, y, T, c, sigma):
        sol, pq = self._invariants_at(T, c, sigma)
        p_t, q_t = to_pq(np.asarray(x, float), np.asarray(y, float))

        d = np.hypot(pq[:, 0] - p_t, pq[:, 1] - q_t)
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        if not np.isfinite(d[k]) or d[k] > 1e-2:
            return -np.inf

        ny_traj = np.linalg.norm(sol.y[2:4, k])
        ny_targ = np.linalg.norm(y)
        if ny_traj < 1e-12:
            return -np.inf

        lam = ny_targ / ny_traj
        return -float(sol.y[4, k]) * lam**5

    def upper_bound_self_similar(self, x, y, T=1.0, n_c=48, return_arg=False):
        deg = degenerate_case(x, y)
        if deg is not None:
            return (0.0, deg) if return_arg else deg

        best_val, best_arg = -np.inf, None
        for sigma in (-1.0, 1.0):
            grid = np.linspace(0.0, 2.0 * np.pi, n_c, endpoint=False)
            vals = np.array([self.value(x, y, T, c, sigma) for c in grid])
            if not np.any(np.isfinite(vals)):
                continue
            k = int(np.nanargmax(np.where(np.isfinite(vals), vals, -np.inf)))
            res = minimize_scalar(
                lambda cc: -self.value(x, y, T, cc, sigma),
                bounds=(grid[k] - 2 * np.pi / n_c, grid[k] + 2 * np.pi / n_c),
                method="bounded", options={"xatol": 1e-9})
            if -res.fun > best_val:
                best_val, best_arg = float(-res.fun), (float(res.x), sigma)

        return (best_arg, best_val) if return_arg else best_val
    