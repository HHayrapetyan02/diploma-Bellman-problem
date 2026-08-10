import numpy as np
from BellmanLower import LowwerBoundBellmanFunction as LB
from BellmanUtils import OptimizationUtils as OU


class UpperBoundBellmanFunction:
    """
    Computes upper bounds for Bellman function in 2D optimal control problem.
    Provides optimistic estimates (cannot do worse than these values).
    Uses rotated rectangles inscribed in unit disc for relaxation.
    """
    
    def scaledBellman1D(self, x, y, a):
        """
        Scaled 1D Bellman function for control in [-a,a].
        """
        return LB().pureBellman1D(a * x, y) / (a ** 3)

    def upperBoundBellman2DAngleRectangle(self, x, y, phi, xi):
        """
        Upper bound for specific rectangle orientation (phi) and aspect ratio (xi).
        """
        assert 0 <= xi <= np.pi/2
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        
        if xi == 0:
            if np.allclose([x_rotated[1], y_rotated[1]], [0, 0]):
                v = self.scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi))
            else:
                v = float('inf')
        
        elif np.isclose(xi, np.pi/2):
            if np.allclose([x_rotated[0], y_rotated[0]], [0, 0]):
                v = self.scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi))
            else:
                v = float('inf')
        
        else:
            v = (self.scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi)) + 
                self.scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi)))
        
        return v


    def _adaptive_grid_search_xi(self, x_rotated, y_rotated, initial_step=2**(-6)):
        h = initial_step
        tol = 1e-14
        
        if (abs(x_rotated[0]) < tol) and (abs(y_rotated[0]) < tol):
            return LB().pureBellman1D(x_rotated[1], y_rotated[1])
        
        if (abs(x_rotated[1]) < tol) and (abs(y_rotated[1]) < tol):
            return LB().pureBellman1D(x_rotated[0], y_rotated[0])
        
        while True:
            X = np.arange(0, 1 + h, h) * np.pi/2
            X = X[1:-1]
            
            F = np.zeros(len(X))
            
            for k in range(len(X)):
                xi = X[k]
                F[k] = self.upperBoundBellman2DAngleRectangle(
                    x_rotated, y_rotated, 0, xi
                )
            
            if (len(F) >= 2) and (F[0] < F[1]) and (F[-2] > F[-1]):
                break
            
            h = h / 2
        
        return X, F, h


    def upperBoundBellman2DRotatedRectangle(self, x, y, phi):
        """
        Best rectangle upper bound for fixed orientation phi.
        Finds optimal aspect ratio xi.
        """
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        X, F, h = self._adaptive_grid_search_xi(x_rotated, y_rotated)
        
        maxind = np.argmax(F)
        h_step = X[1] - X[0]
        
        def func_xi(xi_param):
            return self.upperBoundBellman2DAngleRectangle(
                x_rotated, y_rotated, 0, xi_param
            )
        
        ximin = X[maxind] - h_step
        ximax = X[maxind] + h_step
        
        return OU.golden_section_search(
            func=func_xi,
            a=ximin,
            b=ximax,
            tol=1e-10,
            maximize=True
        )


    def upperBoundBellman2DRectangle(self, x, y):
        """
        Optimal rectangle upper bound over all orientations.
        Finds best phi, then best xi for that phi.
        """
        tol = 1e-13
        nxy = np.linalg.norm([x, y])
        
        if nxy < tol:
            return 0.0
        
        nx = np.linalg.norm(x)
        if nx < tol:
            return LB().pureBellman1D(0, np.linalg.norm(y))
        
        det_val = np.linalg.det(np.column_stack((x, y)))
        if abs(det_val) / (nxy**2) < tol:
            return LB().pureBellman1D(nx, np.dot(x, y) / nx)
        
        def func_phi(phi_param):
            return self.upperBoundBellman2DRotatedRectangle(x, y, phi_param)
        
        return OU.two_stage_optimization(
            func=func_phi,
            search_range=(0, np.pi/2),
            initial_step=2**(-6),
            tol=1e-10,
            maximize=True
        )