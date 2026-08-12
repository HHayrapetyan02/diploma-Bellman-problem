import numpy as np

from bounds.common import degenerate_case
from bounds.lower.square import LowerBoundBellmanFunction
from utils.hjb import HJBSubsolutionLP


class HJBCertificateBound:
    
    def __init__(self, **lp_kwargs):
        self.lp = HJBSubsolutionLP(**lp_kwargs)
        self._sol = None

    def fit(self, verbose=False):
        self._sol = self.lp.solve()
        if self._sol["success"]:
            HJBSubsolutionLP.verify(self._sol)
        if verbose:
            print("LP:", self._sol["status"],
                  "slack =", self._sol.get("slack"),
                  "certified =", self._sol.get("certified"),
                  "max residual =", self._sol.get("max_residual"))
        return self._sol

    def lower_bound_hjb(self, x, y):
        deg = degenerate_case(x, y)
        if deg is not None:
            return deg

        square = LowerBoundBellmanFunction().lowerBoundBellman2D(x, y)
        if self._sol is None:
            self.fit()
        if not self._sol["success"] or not self._sol.get("certified", False):
            return square

        val = HJBSubsolutionLP.evaluate(self._sol, x, y)
        if not np.isfinite(val):
            return square
        return min(val, square)      
    