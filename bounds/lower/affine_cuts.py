import numpy as np

from bounds.common import degenerate_case
from bounds.convex.affine import AffineCutLowerBound
from bounds.lower.square import LowerBoundBellmanFunction


class AffineCutsBound:
    def __init__(self, n_restarts=20, seed=0, extra_cuts=(),
                 combine_with_square=True):
        self.engine = AffineCutLowerBound(cuts=extra_cuts,
                                          n_restarts=n_restarts, seed=seed)
        self.combine = combine_with_square

    def lower_bound_affine(self, x, y, return_arg=False):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        deg = degenerate_case(x, y)
        if deg is not None:
            return (None, deg) if return_arg else deg

        val = self.engine.omega(x, y)          
        if self.combine:
            sq = LowerBoundBellmanFunction().lowerBoundBellman2D(x, y)
            val = min(val, sq)                 
        return (None, val) if return_arg else val
    