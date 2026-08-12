import numpy as np

from bounds.common import pure_bellman_1d
from utils.utils import OptimizationUtils as OU


class LowerBoundBellmanFunction:
    """Граница через описанный квадрат.

    Диск D содержится в квадрате [-1,1]^2, повёрнутом на любой угол phi.
    Больше допустимых управлений означает меньшую стоимость, поэтому
    omega_square >= omega_2D. Минимизация по phi даёт самую тугую
    из границ этого семейства. Для квадрата задача расщепляется на две
    независимые одномерные задачи Фуллера, что позволяет вычислить
    значение в замкнутой форме.
    """

    def pureBellman1D(self, x, y):
        """Точная одномерная функция Беллмана, u in [-1, 1]."""
        return float(pure_bellman_1d(x, y))

    def lowerBoundBellman2DAngle(self, x, y, phi):
        """Значение для квадрата фиксированной ориентации phi."""
        x_rotated, y_rotated = OU.rotate_vectors(x, y, phi)
        return (self.pureBellman1D(x_rotated[0], y_rotated[0])
                + self.pureBellman1D(x_rotated[1], y_rotated[1]))

    def lowerBoundBellman2D(self, x, y, n_points=50, return_arg=False):
        """Оптимизация по ориентации квадрата.

        Квадрат инвариантен относительно поворота на pi/2, поэтому
        достаточно перебрать phi в [0, pi/2].
        """
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
    