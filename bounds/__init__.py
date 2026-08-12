import numpy as np

from bounds.common import pure_bellman_1d, scaled_bellman_1d, degenerate_case
from bounds.lower import (LowerBoundBellmanFunction, HJBCertificateBound)
from bounds.upper import (UpperBoundBellmanFunction, PolicyImprovementBound,
                          SelfSimilarControlBound, TimeOptimalBound)


def best_bounds(x, y):
    lower_vals = [UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y),
                  TimeOptimalBound().upper_bound_time_optimal(x, y),
                  SelfSimilarControlBound().upper_bound_self_similar(x, y),
                  PolicyImprovementBound().upper_bound_policy_improvement(x, y)]
    upper_vals = [LowerBoundBellmanFunction().lowerBoundBellman2D(x, y)]

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
