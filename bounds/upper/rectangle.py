import numpy as np

from bounds.common import degenerate_case, pure_bellman_1d, scaled_bellman_1d
from utils.constants import Constants as Const
from utils.utils import OptimizationUtils as OU


class UpperBoundBellmanFunction:
    """Граница через вписанный прямоугольник.

    Прямоугольник R(phi, cos xi, sin xi) с полуосями, лежащими на единичной
    окружности, вписан в диск D. Меньше допустимых управлений означает
    большую стоимость, поэтому omega_rect <= omega_2D. Максимизация по
    ориентации phi и соотношению сторон xi даёт самую тугую границу
    этого семейства. Для прямоугольника задача расщепляется на две
    независимые одномерные задачи, что даёт замкнутую форму.
    """

    def scaledBellman1D(self, x, y, a):
        """Одномерная функция Беллмана для управления u in [-a, a]."""
        return float(scaled_bellman_1d(x, y, a))

    def upperBoundBellman2DAngleRectangle(self, x, y, phi, xi):
        """Значение для прямоугольника с ориентацией phi и параметром xi.

        Полуоси равны cos(xi) и sin(xi). При вырождении одной из полуосей
        движение вдоль соответствующей оси невозможно, и значение равно
        минус бесконечности, если начальные данные этой оси ненулевые.
        """
        if not (0 <= xi <= np.pi / 2):
            raise ValueError(f"xi must be in [0, pi/2], got {xi}")

        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        a, b = np.cos(xi), np.sin(xi)
        scale = float(np.hypot(np.linalg.norm(x_rotated),
                               np.linalg.norm(y_rotated)))

        if a <= Const.EPS:
            ok = (abs(x_rotated[0]) <= Const.EPS * scale
                  and abs(y_rotated[0]) <= Const.EPS * scale)
            return float(pure_bellman_1d(x_rotated[1], y_rotated[1])) if ok \
                else -np.inf

        if b <= Const.EPS:
            ok = (abs(x_rotated[1]) <= Const.EPS * scale
                  and abs(y_rotated[1]) <= Const.EPS * scale)
            return float(pure_bellman_1d(x_rotated[0], y_rotated[0])) if ok \
                else -np.inf

        return (self.scaledBellman1D(x_rotated[0], y_rotated[0], a)
                + self.scaledBellman1D(x_rotated[1], y_rotated[1], b))

    def _adaptive_grid_search_xi(self, x_rotated, y_rotated, n_points=64,
                                 max_refines=20):
        """Локализация внутреннего максимума по xi измельчением сетки.

        Концы отрезка исключаются, так как на них значение обращается
        в минус бесконечность. Сетка измельчается, пока максимум не
        окажется строго внутри, что гарантирует корректность скобки.
        """
        n = n_points
        for _ in range(max_refines):
            X = np.linspace(0, np.pi / 2, n + 1)[1:-1]
            F = np.zeros(len(X))
            for k in range(len(X)):
                F[k] = self.upperBoundBellman2DAngleRectangle(
                    x_rotated, y_rotated, 0, X[k])
            if (len(F) >= 3) and (F[0] < F[1]) and (F[-2] > F[-1]):
                return X, F
            n = n * 2
        raise RuntimeError(
            f"xi maximum not bracketed for x={x_rotated}, y={y_rotated}")

    def upperBoundBellman2DRotatedRectangle(self, x, y, phi, n_points=64):
        """Наилучшее соотношение сторон при фиксированной ориентации phi."""
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        scale = float(np.hypot(np.linalg.norm(x_rotated),
                               np.linalg.norm(y_rotated)))
        if scale < Const.EPS:
            return 0.0

        if (abs(x_rotated[0]) < Const.EPS * scale
                and abs(y_rotated[0]) < Const.EPS * scale):
            return float(pure_bellman_1d(x_rotated[1], y_rotated[1]))

        if (abs(x_rotated[1]) < Const.EPS * scale
                and abs(y_rotated[1]) < Const.EPS * scale):
            return float(pure_bellman_1d(x_rotated[0], y_rotated[0]))

        X, F = self._adaptive_grid_search_xi(x_rotated, y_rotated, n_points)
        maxind = int(np.argmax(F))
        h_step = X[1] - X[0]

        def func_xi(xi_param):
            return self.upperBoundBellman2DAngleRectangle(
                x_rotated, y_rotated, 0, xi_param)

        f_fine = OU.golden_section_search(
            func=func_xi, a=X[maxind] - h_step, b=X[maxind] + h_step,
            tol=1e-10, maximize=True)

        return max(f_fine, float(F[maxind]))

    def upperBoundBellman2DRectangle(self, x, y, n_points=64, return_arg=False):
        """Оптимизация по ориентации прямоугольника."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        deg = degenerate_case(x, y)
        if deg is not None:
            return (0.0, deg) if return_arg else deg

        def func_phi(phi_param):
            return self.upperBoundBellman2DRotatedRectangle(
                x, y, phi_param, n_points)

        return OU.two_stage_optimization(
            func=func_phi, search_range=(0, np.pi / 2), n_points=n_points,
            tol=1e-10, maximize=True, return_arg=return_arg)
    