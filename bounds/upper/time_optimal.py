import numpy as np
from scipy.optimize import fsolve

from bounds.common import degenerate_case
from utils.geometry import to_pq


class TimeOptimalBound:
    def __init__(self, tol=1e-12):
        self.tol = tol

    @staticmethod
    def _gram_terms(tau, tau_bar):
        beta = np.sqrt(max(1.0 - tau_bar**2, 0.0))
        s = np.sqrt(tau**2 + beta**2)
        at = np.arctanh(np.clip(tau_bar, -1 + 1e-15, 1 - 1e-15))
        ash = np.arcsinh(tau / beta) if beta > 1e-14 else 0.0

        nx = (0.25 * tau**4
              + (beta**2 * at**2 + 1.25 * beta**2 + 1.0) * tau**2
              - (beta**2 * at + tau_bar) * tau
              + 0.25 * (4 * beta**4 + 3 * beta**2 + 1
                        + beta**4 * at**2 - 2 * beta**2 * tau_bar * at)
              - tau**2 * s
              + 0.5 * (3 * beta**2 * at + tau_bar) * tau * s
              - 2 * beta**2 * s
              - 2 * beta**2 * tau**2 * at * ash
              + beta**2 * tau * ash
              - 0.5 * beta**2 * (beta**2 * at - tau_bar) * ash
              - 1.5 * beta**2 * tau * s * ash
              + beta**2 * tau**2 * ash**2
              + 0.25 * beta**4 * ash**2)

        xy = (0.5 * tau**3
              + (beta**2 * at**2 + 0.5 * beta**2 + 1.0) * tau
              - 0.5 * beta**2 * s * ash
              + 0.5 * (beta**2 * at + tau_bar) * s
              - 1.5 * tau * s
              + 0.5 * beta**2 * ash
              - 0.5 * (beta**2 * at + tau_bar)
              + beta**2 * tau * ash**2
              - 2 * beta**2 * tau * at * ash)

        ny = (tau**2 - 2 * s + beta**2 * ash**2
              - 2 * beta**2 * at * ash
              + (beta**2 * at**2 + beta**2 + 1))

        return nx, xy, ny

    @staticmethod
    def _decode(s):
        s1, s2 = float(s[0]), float(s[1])
        tau_bar = np.tanh(s1)
        tau0 = tau_bar - np.exp(np.clip(s2, -30.0, 30.0))
        return tau_bar, tau0

    def _residual(self, s, p_target, q_target):
        tau_bar, tau0 = self._decode(s)
        nx, xy, ny = self._gram_terms(tau0, tau_bar)
        if ny <= 1e-14 or not np.isfinite(nx) or not np.isfinite(xy):
            return [1e6, 1e6]
        return [nx / ny**2 - p_target, xy / ny**1.5 - q_target]

    def _fit_params(self, x, y):
        p_t, q_t = to_pq(np.asarray(x, float), np.asarray(y, float))

        best = None
        for tb0 in (0.99, 0.9, 0.5, 0.0, -0.5, -0.9):
            for gap0 in (0.1, 0.5, 1.0, 2.0, 3.0):
                s0 = [np.arctanh(tb0), np.log(gap0)]
                sol, info, ier, _ = fsolve(
                    self._residual, s0, args=(p_t, q_t),
                    full_output=True, xtol=self.tol)
                if ier != 1:
                    continue
                r = float(np.linalg.norm(self._residual(sol, p_t, q_t)))
                if r > 1e-8:
                    continue
                tau_bar, tau0 = self._decode(sol)
                if not (abs(tau_bar) < 1.0 and tau0 < tau_bar):
                    continue
                if best is None or r < best[1]:
                    best = ((tau_bar, tau0), r)
        if best is None:
            return None

        tau_bar, tau0 = best[0]
        _, _, ny = self._gram_terms(tau0, tau_bar)
        if ny <= 0.0:
            return None
        alpha = np.sqrt(ny) / np.linalg.norm(y)
        if not np.isfinite(alpha) or alpha <= 0.0:
            return None
        return float(alpha), float(tau_bar), float(tau0)

    @staticmethod
    def _objective(tau0, tau_bar):
        from scipy.integrate import quad

        def integrand(t):
            nx, _, _ = TimeOptimalBound._gram_terms(t, tau_bar)
            return nx

        val, _ = quad(integrand, tau0, tau_bar, limit=200)
        return 0.5 * val

    def upper_bound_time_optimal(self, x, y, return_arg=False):
        deg = degenerate_case(x, y)
        if deg is not None:
            return (None, deg) if return_arg else deg

        fit = self._fit_params(x, y)
        if fit is None:
            return (None, -np.inf) if return_arg else -np.inf

        alpha, tau_bar, tau0 = fit
        cost = self._objective(tau0, tau_bar) / alpha**5
        if not np.isfinite(cost) or cost < 0.0:
            return (None, -np.inf) if return_arg else -np.inf

        val = -float(cost)
        return ((alpha, tau_bar, tau0), val) if return_arg else val
    