import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from utils.constants import Constants as Const
from utils.geometry import Disc


class SelfSimilarHJBSolver:
    """Численное решение задачи Беллмана на факторе по симметриям.

    Задача min (1/2) int ||x||^2 dt при xdot = y, ydot = u in U
    обладает фуллеровской симметрией omega(lam^2 x, lam y) = lam^5 omega(x, y).
    Если U вращательно симметрично, добавляется и вращательная симметрия,
    и задача сводится к двум переменным. Для многоугольного U вращательная
    симметрия ослабляется до конечной группы C_n, и в фактор-пространстве
    остаётся дополнительная угловая координата.

    Решение ищется методом value iteration на полу-лагранжевой схеме:
    состояние сносится вдоль динамики на шаг dt, значение интерполируется,
    затем берётся минимум по управлениям из конечного набора образующих U.
    """

    def __init__(self, body=None, n_r=48, n_ang=96, n_controls=64,
                 r_max=1.0, dt=2.0e-2):
        """
        Args:
            body: множество управлений (объект ConvexBody).
            n_r: число узлов по радиальной координате фактор-пространства.
            n_ang: число узлов по угловой координате.
            n_controls: число дискретных управлений на границе U.
            r_max: максимальный радиус сетки.
            dt: шаг интегрирования полу-лагранжевой схемы.
        """
        self.body = body if body is not None else Disc()
        self.n_r = n_r
        self.n_ang = n_ang
        self.r_max = r_max
        self.dt = dt
        ang = 2.0 * np.pi * np.arange(n_controls) / n_controls
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        self.controls = self.body.argmax(dirs)

    def _state_grid(self):
        """Сетка в координатах (r, phi) четырёхмерного пространства состояний.

        Используется представление x = r^2 (cos a, sin a) * cos t,
        y = r (cos b, sin b) * sin t, редуцированное по вращению.
        """
        r = np.linspace(self.r_max / self.n_r, self.r_max, self.n_r)
        a = np.linspace(0.0, 2.0 * np.pi, self.n_ang, endpoint=False)
        return r, a

    def solve(self, n_iter=400, tol=1e-9):
        """Value iteration до сходимости.

        Returns:
            Словарь с сеткой и значениями функции Беллмана на ней.
        """
        r, a = self._state_grid()
        R, A = np.meshgrid(r, a, indexing="ij")

        x1 = R**2 * np.cos(A)
        x2 = R**2 * np.sin(A)
        y1 = R * np.cos(A)
        y2 = -R * np.sin(A)

        V = np.zeros_like(R)
        cost = 0.5 * (x1**2 + x2**2) * self.dt

        for _ in range(n_iter):
            V_new = np.full_like(V, np.inf)
            for u in self.controls:
                xn1 = x1 + y1 * self.dt
                xn2 = x2 + y2 * self.dt
                yn1 = y1 + u[0] * self.dt
                yn2 = y2 + u[1] * self.dt
                Vn = self._interp(V, r, a, xn1, xn2, yn1, yn2)
                V_new = np.minimum(V_new, cost + Vn)
            delta = np.max(np.abs(V_new - V))
            V = V_new
            if delta < tol:
                break

        return {"r": r, "a": a, "V": V}

    @staticmethod
    def _interp(V, r, a, x1, x2, y1, y2):
        """Билинейная интерполяция значения в снесённой точке."""
        rn = np.sqrt(np.hypot(y1, y2))
        an = np.arctan2(x2, x1)

        ir = np.clip(np.searchsorted(r, rn) - 1, 0, len(r) - 2)
        ia = np.mod(np.floor(an / (2 * np.pi) * len(a)).astype(int), len(a))
        ia2 = np.mod(ia + 1, len(a))

        wr = np.clip((rn - r[ir]) / (r[ir + 1] - r[ir]), 0.0, 1.0)
        wa = 0.5

        v = ((1 - wr) * ((1 - wa) * V[ir, ia] + wa * V[ir, ia2])
             + wr * ((1 - wa) * V[ir + 1, ia] + wa * V[ir + 1, ia2]))
        return v


class HJBSubsolutionLP:
    """Сертифицированная граница через субрешение уравнения Беллмана.

    Функция W(x, y) = ||y||^5 f(p, q) с p = ||x||^2/||y||^4,
    q = <x,y>/||y||^3 является субрешением, если

        <W_x, y> - ||W_y|| + (1/2)||x||^2 >= 0,   W(0, 0) = 0.

    Тогда W <= J* всюду, то есть -W даёт сертифицированную границу
    для omega = -J*. Норма ||W_y|| нелинейна по W, но представима как
    max по направлениям e единичной длины скалярного произведения
    <W_y, e>. Замена максимума на конечный набор направлений даёт
    линейную релаксацию в нужную сторону, и задача максимизации
    интеграла от f сводится к задаче линейного программирования.
    """

    def __init__(self, n_p=40, n_q=40, n_dirs=32, p_max=4.0, q_max=2.0):
        """
        Args:
            n_p, n_q: размеры сетки по инвариантным координатам.
            n_dirs: число направлений в линеаризации нормы.
            p_max, q_max: границы расчётной области.
        """
        self.n_p = n_p
        self.n_q = n_q
        self.n_dirs = n_dirs
        self.p_max = p_max
        self.q_max = q_max

    def _grid(self):
        """Допустимая часть сетки: по неравенству Коши-Буняковского p >= q^2."""
        p = np.linspace(1e-3, self.p_max, self.n_p)
        q = np.linspace(-self.q_max, self.q_max, self.n_q)
        P, Q = np.meshgrid(p, q, indexing="ij")
        mask = P >= Q**2
        return p, q, P, Q, mask

    def build_lp(self):
        """Сборка матрицы ограничений и вектора цели задачи ЛП.

        Возвращает разреженную матрицу A_ub, вектор b_ub и вектор c
        такие, что задача имеет вид min c^T f при A_ub f <= b_ub.
        """
        p, q, P, Q, mask = self._grid()
        idx = -np.ones(P.shape, dtype=int)
        idx[mask] = np.arange(int(mask.sum()))
        n_var = int(mask.sum())

        dp = p[1] - p[0]
        dq = q[1] - q[0]

        rows, cols, vals, rhs = [], [], [], []
        row = 0

        ang = 2.0 * np.pi * np.arange(self.n_dirs) / self.n_dirs
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)

        for i in range(1, self.n_p - 1):
            for j in range(1, self.n_q - 1):
                if not mask[i, j]:
                    continue
                pi, qj = P[i, j], Q[i, j]

                dfdp_c = np.array([-1.0, 1.0]) / (2.0 * dp)
                dfdp_i = np.array([idx[i - 1, j], idx[i + 1, j]])
                dfdq_c = np.array([-1.0, 1.0]) / (2.0 * dq)
                dfdq_i = np.array([idx[i, j - 1], idx[i, j + 1]])

                if np.any(dfdp_i < 0) or np.any(dfdq_i < 0):
                    continue

                drift_c = np.concatenate([2.0 * qj * dfdp_c,
                                          (pi - 4.0 * qj**2) * dfdq_c])
                drift_i = np.concatenate([dfdp_i, dfdq_i])

                for e in dirs:
                    grad_c = np.concatenate([
                        -e[0] * 4.0 * pi * dfdp_c - e[0] * 3.0 * qj * dfdq_c,
                        e[1] * dfdq_c / max(np.sqrt(max(pi - qj**2, 1e-12)), 1e-6),
                    ])
                    grad_i = np.concatenate([dfdp_i, dfdq_i])

                    all_c = np.concatenate([-drift_c, grad_c])
                    all_i = np.concatenate([drift_i, grad_i])

                    for c_val, c_idx in zip(all_c, all_i):
                        rows.append(row)
                        cols.append(int(c_idx))
                        vals.append(float(c_val))
                    rows.append(row)
                    cols.append(int(idx[i, j]))
                    vals.append(-5.0)
                    rhs.append(0.5 * pi)
                    row += 1

        A_ub = sparse.coo_matrix((vals, (rows, cols)),
                                 shape=(row, n_var)).tocsr()
        b_ub = np.array(rhs, dtype=float)
        c = -np.ones(n_var) * dp * dq
        return A_ub, b_ub, c, idx, mask, p, q

    def solve(self, method="highs"):
        """Решение задачи ЛП и возврат сертифицированного субрешения.

        Returns:
            Словарь с сеткой (p, q), значениями f и маской допустимости.
        """
        from scipy.optimize import linprog

        A_ub, b_ub, c, idx, mask, p, q = self.build_lp()
        bounds = [(None, 0.0)] * A_ub.shape[1]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)

        f = np.full(mask.shape, np.nan)
        if res.success:
            f[mask] = res.x
        return {"p": p, "q": q, "f": f, "mask": mask,
                "success": bool(res.success), "status": res.message}

    @staticmethod
    def evaluate(sol, x, y):
        """Значение сертифицированной границы в точке (x, y)."""
        from utils.geometry import to_pq

        pp, qq = to_pq(x, y)
        p, q, f = sol["p"], sol["q"], sol["f"]
        i = int(np.clip(np.searchsorted(p, pp) - 1, 0, len(p) - 1))
        j = int(np.clip(np.searchsorted(q, qq) - 1, 0, len(q) - 1))
        val = f[i, j]
        if not np.isfinite(val):
            return -np.inf
        return float(np.linalg.norm(y) ** 5 * val)
    