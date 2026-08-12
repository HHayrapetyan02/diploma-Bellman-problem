import numpy as np

from bounds.common import pure_bellman_1d
from utils.utils import OptimizationUtils as OU


class LowerBoundBellmanFunction:

    def pureBellman1D(self, x, y):
        return float(pure_bellman_1d(x, y))

    def lowerBoundBellman2DAngle(self, x, y, phi):
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        return (self.pureBellman1D(x_rotated[0], y_rotated[0])
                + self.pureBellman1D(x_rotated[1], y_rotated[1]))

    def lowerBoundBellman2D(self, x, y, n_points=50, return_arg=False):
        def func_to_minimize(phi):
            return self.lowerBoundBellman2DAngle(x, y, phi)

        return OU.two_stage_optimization(
            func=func_to_minimize,
            search_range=(0, 0.5 * np.pi),
            n_points=n_points,
            tol=1e-10,
            maximize=False,
            return_arg=return_arg
        )
    