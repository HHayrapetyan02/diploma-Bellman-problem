import numpy as np
from scipy.optimize import fsolve

from bounds.common import degenerate_case
from utils.geometry import to_pq


class TimeOptimalBound:
    """Граница методом быстродействия.

    Рассматривается вспомогательная задача минимизации времени

        min (T - t0):  xdot = y, ydot = u in D, x(T) = y(T) = 0,

    имеющая то же множество допустимых траекторий, что и исходная задача,
    но другой функционал. Времяоптимальная траектория допустима для
    исходной задачи, поэтому подстановка её в функционал даёт верхнюю
    оценку стоимости, то есть нижнюю оценку для omega.

    По принципу максимума сопряжённая переменная psi аффинна по времени,
    оптимальное управление равно psi/||psi||, а условие трансверсальности
    даёт ||psi(T)|| = 1. После масштабирования времени переменной
    tau = alpha*t с beta = sqrt(1 - tau_bar^2) элементы грамиана и сам
    функционал выражаются в замкнутой форме. Обращение зависимости
    грамиана от параметров выполняется по паре инвариантов методом Ньютона.

    Метод даёт существенно более точную оценку, чем прямоугольные:
    относительный зазор составляет от долей процента до нескольких процентов
    против примерно двадцати процентов у прямоугольного семейства.
    """

    def __init__(self, tol=1e-12):
        """
        Args:
            tol: точность решения нелинейной системы для параметров.
        """
        self.tol = tol

    @staticmethod
    def _gram_terms(tau, tau_bar):
        """Элементы грамиана, нормированные степенями alpha.

        Реализуются выражения для alpha^4*||x||^2, alpha^3*<x,y> и
        alpha^2*||y||^2, полученные интегрированием динамики с
        оптимальным управлением.
        """
        beta = np.sqrt(max(1.0 - tau_bar**2, 0.0))
        s = np.sqrt(tau**2 + beta**2)
        at = np.arctanh(np.clip(tau_bar, -1 + 1e-15, 1 - 1e-15))
        ash = np.arcsinh(tau / beta) if beta > 1e-14 else 0.0

        nx = (0.25 * tau**4
              + (beta**2 * at**2 + 1.25 * beta**2 + 1.0) * tau**2
              - (beta**2 * at + tau_bar) * tau
              + 0.25 * (4 * beta**4 + 3 * beta**2 + 1
                        + beta**4 * at**2 - 2 * beta**2 * tau_bar * at)
              - tau**2 * s
              + 0.5 * (3 * beta**2 * at + tau_bar) * tau * s
              - 2 * beta**2 * s
              - 2 * beta**2 * tau**2 * at * ash
              + beta**2 * tau * ash
              - 0.5 * beta**2 * (beta**2 * at - tau_bar) * ash
              - 1.5 * beta**2 * tau * s * ash
              + beta**2 * tau**2 * ash**2
              + 0.25 * beta**4 * ash**2)

        xy = (0.5 * tau**3
              + (beta**2 * at**2 + 0.5 * beta**2 + 1.0) * tau
              - 0.5 * beta**2 * s * ash
              + 0.5 * (beta**2 * at + tau_bar) * s
              - 1.5 * tau * s
              + 0.5 * beta**2 * ash
              - 0.5 * (beta**2 * at + tau_bar)
              + beta**2 * tau * ash**2
              - 2 * beta**2 * tau * at * ash)

        ny = (tau**2 - 2 * s + beta**2 * ash**2
              - 2 * beta**2 * at * ash
              + (beta**2 * at**2 + beta**2 + 1))

        return nx, xy, ny

    def _residual(self, params, p_target, q_target):
        """Невязка по инвариантам для метода Ньютона."""
        tau_bar, tau0 = params
        tau_bar = np.clip(tau_bar, -0.999999, 0.999999)
        nx, xy, ny = self._gram_terms(tau0, tau_bar)
        if ny <= 1e-14:
            return [1e6, 1e6]
        return [nx / ny**2 - p_target, xy / ny**1.5 - q_target]

    def _fit_params(self, x, y):
        """Определение (alpha, tau_bar, tau0) по заданным начальным данным."""
        p_t, q_t = to_pq(np.asarray(x, float), np.asarray(y, float))

        best = None
        for tb0 in (0.9, 0.5, 0.0, -0.5, -0.9):
            for t00 in (-3.0, -2.0, -1.0, -0.5, 0.5):
                sol, info, ier, _ = fsolve(
                    self._residual, [tb0, t00], args=(p_t, q_t),
                    full_output=True, xtol=self.tol)
                if ier == 1:
                    r = np.linalg.norm(self._residual(sol, p_t, q_t))
                    if best is None or r < best[1]:
                        best = (sol, r)
        if best is None or best[1] > 1e-6:
            return None

        tau_bar, tau0 = best[0]
        _, _, ny = self._gram_terms(tau0, tau_bar)
        alpha = np.sqrt(ny) / np.linalg.norm(y)
        return float(alpha), float(tau_bar), float(tau0)

    @staticmethod
    def _objective(tau0, tau_bar):
        """Значение исходного функционала на времяоптимальной траектории.

        Вычисляется как определённый интеграл от нормированного ||x||^2
        по масштабированному времени между начальной и конечной точками.
        """
        from scipy.integrate import quad

        def integrand(t):
            nx, _, _ = TimeOptimalBound._gram_terms(t, tau_bar)
            return nx

        val, _ = quad(integrand, tau0, tau_bar, limit=200)
        return 0.5 * val

    def upper_bound_time_optimal(self, x, y, return_arg=False):
        """Верхняя оценка через подстановку времяоптимальной траектории.

        Args:
            x, y: начальные данные задачи.
            return_arg: возвращать ли найденные параметры траектории.

        Returns:
            Значение границы либо пара (параметры, значение).
        """
        deg = degenerate_case(x, y)
        if deg is not None:
            return (None, deg) if return_arg else deg

        fit = self._fit_params(x, y)
        if fit is None:
            return (None, -np.inf) if return_arg else -np.inf

        alpha, tau_bar, tau0 = fit
        cost = self._objective(tau0, tau_bar) / alpha**5
        val = -float(cost)
        return ((alpha, tau_bar, tau0), val) if return_arg else val
    