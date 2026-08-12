import numpy as np

from utils.constants import Constants as Const


class OptimizationUtils:

    @staticmethod
    def rotate_matrix(phi):
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array([[c, -s], [s, c]])


    @staticmethod
    def rotate_vectors(x, y, phi):
        rot = OptimizationUtils.rotate_matrix(phi)
        return rot @ x, rot @ y


    @staticmethod
    def grid_search(func, a, b, n_points, maximize=False):
        points = np.linspace(a, b, n_points)
        values = np.zeros(len(points))

        for i, point in enumerate(points):
            values[i] = func(point)

        bad = -np.inf if maximize else np.inf
        values = np.where(np.isfinite(values), values, bad)

        idx = int(np.argmax(values)) if maximize else int(np.argmin(values))
        return float(points[idx]), float(values[idx])


    @staticmethod
    def golden_section_search(func, a, b, tol=1e-10, maximize=False,
                              return_arg=False):
        left, right = float(a), float(b)
        if left > right:
            left, right = right, left

        x1 = left * Const.GR + right * Const.GRP
        x2 = right * Const.GR + left * Const.GRP

        f1, f2 = func(x1), func(x2)
        f_left, f_right = func(left), func(right)

        while right - left > tol:
            if maximize and ((f_left >= f1) or (f2 <= f_right)):
                break
            if (not maximize) and ((f_left <= f1) or (f2 >= f_right)):
                break

            if (f1 < f2) == maximize:
                left, f_left = x1, f1
                x1, f1 = x2, f2
                x2 = right * Const.GR + left * Const.GRP
                f2 = func(x2)
            else:
                right, f_right = x2, f2
                x2, f2 = x1, f1
                x1 = left * Const.GR + right * Const.GRP
                f1 = func(x1)

        pairs = [(left, f_left), (x1, f1), (x2, f2), (right, f_right)]
        best = max(pairs, key=lambda p: p[1]) if maximize \
            else min(pairs, key=lambda p: p[1])

        return (float(best[0]), float(best[1])) if return_arg else float(best[1])


    @staticmethod
    def two_stage_optimization(func, search_range, n_points=64,
                               tol=1e-10, maximize=False, return_arg=False):
        a, b = search_range
        x_grid, f_grid = OptimizationUtils.grid_search(
            func, a, b, n_points + 1, maximize
        )

        h = (b - a) / n_points
        x_min = max(a, x_grid - h)
        x_max = min(b, x_grid + h)

        x_fine, f_fine = OptimizationUtils.golden_section_search(
            func, x_min, x_max, tol, maximize, return_arg=True
        )

        better = (f_fine > f_grid) if maximize else (f_fine < f_grid)
        x_best, f_best = (x_fine, f_fine) if better else (x_grid, f_grid)

        return (x_best, f_best) if return_arg else f_best
    