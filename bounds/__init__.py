import numpy as np

from bounds.common import pure_bellman_1d, scaled_bellman_1d, degenerate_case
from bounds.lower import (LowerBoundBellmanFunction, GeneralRectangleBound,
                          OctagonBound, HJBCertificateBound)
from bounds.upper import (UpperBoundBellmanFunction, PolygonBound,
                          SelfSimilarControlBound, TimeOptimalBound)


def best_bounds(x, y, use_slow=False):
    lower_vals = [UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y),
                  TimeOptimalBound().upper_bound_time_optimal(x, y),
                  SelfSimilarControlBound().upper_bound_self_similar(x, y)]
    upper_vals = [LowerBoundBellmanFunction().lowerBoundBellman2D(x, y),
                  GeneralRectangleBound().lower_bound_general_rectangle(x, y)]

    if use_slow:
        lower_vals.append(PolygonBound(n=8).upper_bound_polygon(x, y))
        upper_vals.append(OctagonBound().lower_bound_octagon(x, y))

    lo = max(v for v in lower_vals if np.isfinite(v))
    hi = min(v for v in upper_vals if np.isfinite(v))
    gap = (hi - lo) / abs(hi) if abs(hi) > 0 else np.nan
    return {"lower": lo, "upper": hi, "gap": gap}


__all__ = [
    "pure_bellman_1d", "scaled_bellman_1d", "degenerate_case",
    "LowerBoundBellmanFunction", "GeneralRectangleBound",
    "OctagonBound", "HJBCertificateBound",
    "UpperBoundBellmanFunction", "PolygonBound",
    "SelfSimilarControlBound", "TimeOptimalBound",
    "best_bounds",
]
