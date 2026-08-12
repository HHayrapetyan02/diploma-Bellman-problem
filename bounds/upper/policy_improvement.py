import numpy as np

from bounds.common import degenerate_case
from bounds.upper.rectangle import UpperBoundBellmanFunction
from bounds.upper.time_optimal import TimeOptimalBound


def _running_cost(x, y, u, h):
    return 0.5 * (x @ x * h
                  + (x @ y) * h**2
                  + (y @ y + x @ u) * h**3 / 3.0
                  + (y @ u) * h**4 / 4.0
                  + (u @ u) * h**5 / 20.0)


class PolicyImprovementBound:
    def __init__(self, n_controls=24, n_steps=1, h_factors=(0.1, 0.2, 0.4),
                 use_time_optimal=False, rect_points=32):
        self.n_controls = n_controls
        self.n_steps = n_steps
        self.h_factors = tuple(h_factors)
        self.use_time_optimal = use_time_optimal
        self.rect_points = rect_points
        self._rect = UpperBoundBellmanFunction()
        self._to = TimeOptimalBound()

    def base(self, x, y):
        val = self._rect.upperBoundBellman2DRectangle(
            x, y, n_points=self.rect_points)
        if self.use_time_optimal:
            v = self._to.upper_bound_time_optimal(x, y)
            if np.isfinite(v):
                val = max(val, v)          
        return float(val)

    def _step(self, x, y, value_fn):
        ang = 2.0 * np.pi * np.arange(self.n_controls) / self.n_controls
        controls = np.stack([np.cos(ang), np.sin(ang)], axis=-1)

        scale = float(np.linalg.norm(y)) + float(np.linalg.norm(x)) ** 0.5
        best = -np.inf
        for hf in self.h_factors:
            h = hf * scale
            for u in controls:
                xn = x + y * h + u * h * h / 2.0
                yn = y + u * h
                val = value_fn(xn, yn) - _running_cost(x, y, u, h)
                if val > best:
                    best = val
        return float(best)

    def upper_bound_policy_improvement(self, x, y, n_steps=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        deg = degenerate_case(x, y)
        if deg is not None:
            return deg

        k = self.n_steps if n_steps is None else n_steps

        def value_at(level):
            if level == 0:
                return self.base
            lower = value_at(level - 1)
            return lambda xx, yy: max(self._step(xx, yy, lower),
                                      self.base(xx, yy))

        return float(max(value_at(k)(x, y), self.base(x, y)))