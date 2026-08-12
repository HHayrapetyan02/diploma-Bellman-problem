import numpy as np

from utils.hjb import HJBSubsolutionLP


class HJBCertificateBound:
    def __init__(self, **lp_kwargs):
        self.lp = HJBSubsolutionLP(**lp_kwargs)
        self._sol = None

    def fit(self):
        self._sol = self.lp.solve()
        return self._sol

    def lower_bound_hjb(self, x, y):
        if self._sol is None:
            self.fit()
        if not self._sol["success"]:
            return -np.inf
        return HJBSubsolutionLP.evaluate(self._sol, x, y)
    