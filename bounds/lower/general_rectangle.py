import numpy as np

from bounds.common import scaled_bellman_1d
from utils.utils import OptimizationUtils as OU


class GeneralRectangleBound:
    """Граница через произвольный описанный прямоугольник.

    Рассматривается семейство R(phi, a, b) = [-a, a] x [-b, b], повёрнутое
    на phi, при условии D в R, то есть a >= 1 и b >= 1. Задача расщепляется,
    и значение равно сумме одномерных функций с масштабами a и b.

    Теоретическое замечание: расширение сторон сверх единицы только
    увеличивает множество управлений и, следовательно, увеличивает omega,
    тогда как требуется минимум. Поэтому минимум семейства достигается
    ровно на квадрате a = b = 1, и метод служит проверкой согласованности
    с классом LowwerBoundBellmanFunction, а не источником улучшения.
    Реальное усиление даёт октагон, реализованный в модуле octagon.
    """

    def value(self, x, y, phi, a, b):
        """Значение границы для прямоугольника с параметрами (phi, a, b)."""
        if a < 1.0 or b < 1.0:
            return np.inf
        xr, yr = OU.rotate_vectors(x, y, phi)
        return float(scaled_bellman_1d(xr[0], yr[0], a)
                     + scaled_bellman_1d(xr[1], yr[1], b))

    def lower_bound_general_rectangle(self, x, y, n_points=48,
                                      a_max=3.0, return_arg=False):
        """Минимизация по (phi, a, b) при ограничении D в R.

        Внешняя оптимизация ведётся по углу phi, внутренняя -- по
        полуосям a и b, каждая на отрезке [1, a_max]. Ожидаемый оптимум
        a = b = 1 воспроизводит результат для квадрата.
        """
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
    