import numpy as np

from bounds.common import degenerate_case
from bounds.upper.rectangle import UpperBoundBellmanFunction
from utils.geometry import Polygon
from utils.hjb import SelfSimilarHJBSolver
from utils.utils import OptimizationUtils as OU


class PolygonBound:
    def __init__(self, n=6, solver_kwargs=None):
        if n % 2 != 0:
            raise ValueError(f"n must be even for central symmetry, got {n}")
        self.n = n
        self.solver_kwargs = solver_kwargs or {}
        self._cache = {}

    def _solution(self, theta):
        key = round(theta % (2 * np.pi / self.n), 9)
        if key not in self._cache:
            body = Polygon.regular_inscribed(self.n, key)
            solver = SelfSimilarHJBSolver(body=body, **self.solver_kwargs)
            self._cache[key] = solver.solve()
        return self._cache[key]

    def value_at_angle(self, x, y, theta):
        sol = self._solution(theta)
        r = float(np.sqrt(np.linalg.norm(y)))
        a = float(np.arctan2(x[1], x[0]))
        i = int(np.clip(np.searchsorted(sol["r"], r) - 1, 0, len(sol["r"]) - 1))
        j = int(np.mod(a / (2 * np.pi) * len(sol["a"]), len(sol["a"])))
        return -float(sol["V"][i, j])

    def upper_bound_polygon(self, x, y, n=None, n_points=24, return_arg=False):
        if n is not None and n != self.n:
            self.n = n
            self._cache.clear()

        deg = degenerate_case(x, y)
        if deg is not None:
            return (0.0, deg) if return_arg else deg

        rect = UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y)

        def by_theta(theta):
            return self.value_at_angle(x, y, theta)

        theta_best, val = OU.two_stage_optimization(
            by_theta, (0.0, 2.0 * np.pi / self.n), n_points=n_points,
            tol=1e-8, maximize=True, return_arg=True)

        val = max(val, rect)
        return (theta_best, val) if return_arg else val
    