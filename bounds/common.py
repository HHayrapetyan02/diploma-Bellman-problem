import numpy as np

from utils.constants import Constants as Const


def pure_bellman_1d(x, y):
    """Точная функция Беллмана одномерной задачи Фуллера, u in [-1, 1].

    Возвращает omega = -J*, то есть неположительную величину.
    Обе ветви формулы объединены знаком s: s = +1 при x >= -beta*y*|y|,
    иначе s = -1. Подкоренное выражение на выбранной ветви не меньше
    y^2 * (1 - 2*beta) > 0, клиппинг защищает от ошибок округления.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    s = np.where(x >= -Const.BETA * y * np.abs(y), 1.0, -1.0)
    base = np.maximum(y * y + 2.0 * s * x, 0.0)

    return (s * (-0.5 * x * x * y - y**5 / 15.0)
            - x * y**3 / 3.0
            - Const.GAMMA * base**2.5)


def scaled_bellman_1d(x, y, a):
    """Функция Беллмана одномерной задачи для управления u in [-a, a].

    Замена времени и координат даёт omega_a(x, y) = omega(a*x, y) / a^3.
    Эквивалентная форма a^2 * omega(x/a, y/a) годится как независимая проверка.
    """
    a = np.asarray(a, dtype=float)
    return pure_bellman_1d(a * x, y) / a**3


def degenerate_case(x, y):
    """Обработка вырожденных конфигураций, общая для всех оценок.

    Возвращает точное значение omega, если пара (x, y) вырождена,
    и None иначе. Все проверки относительные, так как задача однородна.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    scale = float(np.hypot(np.linalg.norm(x), np.linalg.norm(y)))
    if scale < Const.EPS:
        return 0.0

    nx = float(np.linalg.norm(x))
    if nx < Const.EPS * scale:
        return float(pure_bellman_1d(0.0, np.linalg.norm(y)))

    det_val = float(np.linalg.det(np.column_stack((x, y))))
    if abs(det_val) / scale**2 < Const.EPS:
        return float(pure_bellman_1d(nx, float(np.dot(x, y)) / nx))

    return None
