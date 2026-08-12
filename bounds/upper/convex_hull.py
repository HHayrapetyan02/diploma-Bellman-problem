import numpy as np

from bounds.common import degenerate_case
from bounds.convex.jensen import JensenUpperBound
from bounds.upper.rectangle import UpperBoundBellmanFunction


class ConvexHullBound:
    def __init__(self, combine_with_rectangle=True, **jensen_kwargs):
        self.engine = JensenUpperBound(**jensen_kwargs)
        self.combine = combine_with_rectangle

    def upper_bound_convex_hull(self, x, y, return_arg=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        deg = degenerate_case(x, y)
        if deg is not None:
            return (None, deg) if return_arg else deg

        val = self.engine.omega(x, y)
        if self.combine:
            rc = UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y)
            val = max(val, rc)                 
        return (None, val) if return_arg else val
    