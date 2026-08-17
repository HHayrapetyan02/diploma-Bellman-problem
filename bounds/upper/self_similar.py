import numpy as np
from scipy.integrate import solve_ivp

from bounds.common import degenerate_case
from utils.geometry import to_pq

P_SELF_SIMILAR = 2.0 / 3.0
Q_SELF_SIMILAR = -2.0 * np.sqrt(6.0) / 9.0


class SelfSimilarControlBound:
    def __init__(self, rtol=1e-12, atol=1e-14, tol_pq=1e-6):
        self.rtol = rtol
        self.atol = atol
        self.tol_pq = tol_pq

    def _backward_trajectory(self, T=1.0, c=0.0, sigma=1.0, n_eval=400):
        def rhs(t, z):
            tau = max(T - t, 1e-14)
            phi = sigma * np.sqrt(5.0) * np.log(tau) + c
            u = np.array([np.cos(phi), np.sin(phi)])
            x = z[0:2]
            y = z[2:4]
            return np.concatenate([y, u, [0.5 * (x @ x)]])

        return solve_ivp(rhs, (T, 0.0), np.zeros(5),
                         t_eval=np.linspace(T, 0.5 * T, n_eval),
                         rtol=self.rtol, atol=self.atol, method="DOP853")

    def value(self, x, y, T=1.0, c=0.0, sigma=1.0):
        p_t, q_t = to_pq(np.asarray(x, float), np.asarray(y, float))

        sol = self._backward_trajectory(T, c, sigma)
        pq = np.full((sol.y.shape[1], 2), np.nan)
        for k in range(sol.y.shape[1]):
            if np.linalg.norm(sol.y[2:4, k]) > 1e-12:
                pq[k] = to_pq(sol.y[0:2, k], sol.y[2:4, k])

        d = np.hypot(pq[:, 0] - p_t, pq[:, 1] - q_t)
        d = np.where(np.isfinite(d), d, np.inf)
        k = int(np.argmin(d))
        if not np.isfinite(d[k]) or d[k] > self.tol_pq:
            return -np.inf

        ny_traj = float(np.linalg.norm(sol.y[2:4, k]))
        if ny_traj < 1e-12:
            return -np.inf

        lam = float(np.linalg.norm(y)) / ny_traj
        return float(sol.y[4, k]) * lam**5

    def upper_bound_self_similar(self, x, y, T=1.0, return_arg=False):
        deg = degenerate_case(x, y)
        if deg is not None:
            return (None, deg) if return_arg else deg

        best_val, best_arg = -np.inf, None
        for sigma in (-1.0, 1.0):
            v = self.value(x, y, T=T, c=0.0, sigma=sigma)
            if v > best_val:
                best_val, best_arg = v, sigma

        return (best_arg, best_val) if return_arg else best_val
    