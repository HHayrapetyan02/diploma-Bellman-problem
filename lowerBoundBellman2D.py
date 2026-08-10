import numpy as np
from lowerBoundBellman2DAngle import lowerBoundBellman2DAngle


def lowerBoundBellman2D(x, y):
    """
    Lower bound on optimal value (upper bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in a unit square [-1,1]
    The best square rotated by an angle phi is taken into account
    The minimum is sought with the golden ratio method
    The value has to be minimized over the angle
    x, y are the 2D arguments
    """
    
    # Создаем массив углов от 0 до 0.5*pi с шагом 0.01*pi
    P = np.arange(0, 0.5 * np.pi + 0.01 * np.pi, 0.01 * np.pi)
    F = np.zeros(len(P))
    
    # Вычисляем значения функции для каждого угла
    for k in range(len(P)):
        phi = P[k]
        F[k] = lowerBoundBellman2DAngle(x, y, phi)
    
    # Находим индекс минимального значения
    minind = np.argmin(F)
    h = P[1] - P[0]  # Шаг сетки
    
    # Определяем интервал для уточнения методом золотого сечения
    phimin = P[minind] - h
    phimax = P[minind] + h
    
    # Золотое сечение
    gr = (np.sqrt(5) - 1) / 2
    grp = 1 - gr
    
    # Инициализация точек для метода золотого сечения
    phil = phimin * gr + phimax * grp
    phir = phimax * gr + phimin * grp
    
    # Вычисляем значения в четырех точках
    f = np.array([
        lowerBoundBellman2DAngle(x, y, phimin),
        lowerBoundBellman2DAngle(x, y, phil),
        lowerBoundBellman2DAngle(x, y, phir),
        lowerBoundBellman2DAngle(x, y, phimax)
    ])
    
    # Метод золотого сечения для уточнения минимума
    while phimax - phimin > 1e-10:
        # Проверка условий выхода (условия унимодальности)
        if (f[0] <= f[1]) or (f[2] >= f[3]):
            break
        
        if f[1] > f[2]:
            # Сдвигаем левую границу
            f = np.array([f[1], f[2], f[2], f[3]])
            phimin = phil
            phil = phir
            phir = phimax * gr + phimin * grp
            f[2] = lowerBoundBellman2DAngle(x, y, phir)
        else:
            # Сдвигаем правую границу
            f = np.array([f[0], f[1], f[1], f[2]])
            phimax = phir
            phir = phil
            phil = phimin * gr + phimax * grp
            f[1] = lowerBoundBellman2DAngle(x, y, phil)
    
    # Возвращаем минимальное значение
    return np.min(f)
