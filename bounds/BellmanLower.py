import numpy as np
from utils.BellmanUtils import OptimizationUtils as OU


class LowwerBoundBellmanFunction:
    """
    Computes lower bounds for Bellman function in 2D optimal control problem.
    Provides pessimistic estimates (cannot do better than these values).
    Used for control in unit square [-1,1] rotated to best orientation.
    """
      
    def pureBellman1D(self, x, y):
        if x >= -OU.BETTA * y * abs(y):
            v = -x**2 * y / 2 - x * y**3 / 3 - y**5 / 15 - OU.GAMMA * (y**2 + 2*x)**(5/2)
        else:
            v = x**2 * y / 2 - x * y**3 / 3 + y**5 / 15 - OU.GAMMA * (y**2 - 2*x)**(5/2)

        return v
    

    def lowerBoundBellman2DAngle(self, x, y, phi):
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        v = self.pureBellman1D(x_rotated[0], y_rotated[0]) + self.pureBellman1D(x_rotated[1], y_rotated[1])
        return v
    

    def lowerBoundBellman2D(self, x, y):
        def func_to_minimize(phi):
            return self.lowerBoundBellman2DAngle(x, y, phi)
        
        return OU.two_stage_optimization(
            func=func_to_minimize,
            search_range=(0, 0.5 * np.pi),
            initial_step=0.01 * np.pi,
            tol=1e-10,
            maximize=False
        )