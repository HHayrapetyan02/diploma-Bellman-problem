import numpy as np

from bounds.common import degenerate_case
from bounds.lower.square import LowerBoundBellmanFunction
from utils.geometry import Polygon
from utils.hjb import SelfSimilarHJBSolver
from utils.utils import OptimizationUtils as OU


class OctagonBound:
    def __init__(self, n_sides=8, solver_kwargs=None):
        self.n_sides = n_sides
        self.solver_kwargs = solver_kwargs or {}
        self._cache = {}

    def _solution(self, theta):
        key = round(theta % (2 * np.pi / self.n_sides), 9)
        if key not in self._cache:
            body = Polygon.regular_circumscribed(self.n_sides, key)
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

    def lower_bound_octagon(self, x, y, n_points=24, return_arg=False):
        deg = degenerate_case(x, y)
        if deg is not None:
            return (0.0, deg) if return_arg else deg

        square = LowerBoundBellmanFunction().lowerBoundBellman2D(x, y)

        def by_theta(theta):
            return self.value_at_angle(x, y, theta)

        theta_best, val = OU.two_stage_optimization(
            by_theta, (0.0, 2.0 * np.pi / self.n_sides),
            n_points=n_points, tol=1e-8, maximize=False, return_arg=True)

        val = min(val, square)
        return (theta_best, val) if return_arg else val
    