import numpy as np

from bounds.common import scaled_bellman_1d
from utils.utils import OptimizationUtils as OU


class GeneralRectangleBound:
    def value(self, x, y, phi, a, b):
        if a < 1.0 or b < 1.0:
            return np.inf
        xr, yr = OU.rotate_vectors(x, y, phi)
        return float(scaled_bellman_1d(xr[0], yr[0], a)
                     + scaled_bellman_1d(xr[1], yr[1], b))

    def lower_bound_general_rectangle(self, x, y, n_points=48,
                                      a_max=3.0, return_arg=False):
        def by_phi(phi):
            xr, yr = OU.rotate_vectors(x, y, phi)

            def by_a(a):
                return float(scaled_bellman_1d(xr[0], yr[0], a))

            def by_b(b):
                return float(scaled_bellman_1d(xr[1], yr[1], b))

            va = OU.two_stage_optimization(
                by_a, (1.0, a_max), n_points=n_points,
                tol=1e-10, maximize=False)
            vb = OU.two_stage_optimization(
                by_b, (1.0, a_max), n_points=n_points,
                tol=1e-10, maximize=False)
            return va + vb

        return OU.two_stage_optimization(
            by_phi, (0.0, 0.5 * np.pi), n_points=n_points,
            tol=1e-10, maximize=False, return_arg=return_arg)
    