import numpy as np


class OptimizationUtils:
    """
    Helper functions for optimizing lower and upper bounds
    """
    
    BETTA = 0.444623560185937                   # 36*β^4 + 3*β^2 - 2 = 0
    GAMMA = 6.753024861778741e-02               # γ = (-β^2 + 2β - 2/3) / (10*(1-2β)^(3/2))               
    GR = (np.sqrt(5) - 1) / 2
    GRP = 1 - GR
    
    @staticmethod
    def rotate_matrix(phi):
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def rotate_vectors(x, y, phi):
        O = OptimizationUtils.rotate_matrix(phi)
        return O @ x, O @ y
    

    @staticmethod
    def grid_search(func, a, b, step, maximize=False):
        points = np.arange(a, b + step, step)
        values = np.zeros(len(points))
        
        for i, point in enumerate(points):
            values[i] = func(point)
        
        if maximize:
            idx = np.argmax(values)
        else:
            idx = np.argmin(values)
        
        return points[idx], values[idx]
    

    @staticmethod
    def golden_section_search(func, a, b, tol=1e-10, maximize=False):
        gr = OptimizationUtils.GR
        grp = OptimizationUtils.GRP
        
        left, right = a, b
        x1 = left * gr + right * grp
        x2 = right * gr + left * grp
        
        f1, f2 = func(x1), func(x2)
        f_left, f_right = func(left), func(right)

        while right - left > tol:
            if maximize:
                if (f_left >= f1) or (f2 <= f_right):
                    break
            else:
                if (f_left <= f1) or (f2 >= f_right):
                    break
            
            if maximize:
                if f1 < f2:
                    # Shift left boundary
                    left, f_left = x1, f1
                    x1, f1 = x2, f2
                    x2 = right * gr + left * grp
                    f2 = func(x2)
                else:
                    # Shift right boundary
                    right, f_right = x2, f2
                    x2, f2 = x1, f1
                    x1 = left * gr + right * grp
                    f1 = func(x1)
            else:
                if f1 > f2:
                    # Shift left boundary
                    left, f_left = x1, f1
                    x1, f1 = x2, f2
                    x2 = right * gr + left * grp
                    f2 = func(x2)
                else:
                    # Shift right boundary
                    right, f_right = x2, f2
                    x2, f2 = x1, f1
                    x1 = left * gr + right * grp
                    f1 = func(x1)
        
        if maximize:
            return max(f_left, f1, f2, f_right)
        else:
            return min(f_left, f1, f2, f_right)
    

    @staticmethod
    def two_stage_optimization(func, search_range, initial_step=2**(-6), 
                               tol=1e-10, maximize=False):
        a, b = search_range
        x_grid, f_grid = OptimizationUtils.grid_search(
            func, a, b, initial_step, maximize
        )
        
        h = initial_step
        x_min = max(a, x_grid - h)
        x_max = min(b, x_grid + h)
        result = OptimizationUtils.golden_section_search(
            func, x_min, x_max, tol, maximize
        )
        
        return result
    

    