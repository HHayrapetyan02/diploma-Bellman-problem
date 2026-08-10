import numpy as np
from pureBellman1D import pureBellman1D
from upperBoundBellman2DRotatedRectangle import upperBoundBellman2DRotatedRectangle

def upperBoundBellman2DRectangle(x, y):
    """
    Upper bound on optimal value (lower bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in the best rotated centered rectangle
    x, y are the 2D arguments
    The best rectangle is sought by a two-level algorithm
    On the lower level the best side ratio is sought by the golden ratio method
    On the upper level the best angle is sought by the golden ratio method
    The value has to be maximized over the side ratio
    """
    tol = 1e-13
    
    # Проверка коллинеарности и особых случаев
    nxy = np.linalg.norm([x, y])
    
    if nxy < tol:
        # x = y = 0
        return 0.0
    
    nx = np.linalg.norm(x)
    if nx < tol:
        # x = 0
        return pureBellman1D(0, np.linalg.norm(y))
    
    # Проверка коллинеарности x и y
    # Создаем матрицу из векторов x и y
    det_val = np.linalg.det(np.column_stack((x, y)))
    if abs(det_val) / (nxy**2) < tol:
        # x и y коллинеарны
        return pureBellman1D(nx, np.dot(x, y) / nx)
    
    # Грубый поиск на сетке для начального приближения
    h = 2**(-6)
    P = np.arange(0, 1 + h, h) * np.pi/2
    F = np.zeros(len(P))
    
    for k in range(len(P)):
        phi = P[k]
        F[k] = upperBoundBellman2DRotatedRectangle(x, y, phi)
    
    # Находим индекс максимального значения
    maxind = np.argmax(F)
    h = P[1] - P[0]  # Шаг сетки
    
    # Определяем интервал для уточнения методом золотого сечения
    phimin = P[maxind] - h
    phimax = P[maxind] + h
    
    # Золотое сечение
    gr = (np.sqrt(5) - 1) / 2
    grp = 1 - gr
    
    # Инициализация точек для метода золотого сечения
    phil = phimin * gr + phimax * grp
    phir = phimax * gr + phimin * grp
    
    # Вычисляем значения в четырех точках
    f = np.array([
        upperBoundBellman2DRotatedRectangle(x, y, phimin),
        upperBoundBellman2DRotatedRectangle(x, y, phil),
        upperBoundBellman2DRotatedRectangle(x, y, phir),
        upperBoundBellman2DRotatedRectangle(x, y, phimax)
    ])
    
    # Метод золотого сечения для уточнения максимума
    while phimax - phimin > 1e-10:
        # Проверка условий выхода
        if (f[0] >= f[1]) or (f[2] <= f[3]):
            break
        
        if f[1] < f[2]:
            # Сдвигаем левую границу
            f = np.array([f[1], f[2], f[2], f[3]])
            phimin = phil
            phil = phir
            phir = phimax * gr + phimin * grp
            f[2] = upperBoundBellman2DRotatedRectangle(x, y, phir)
        else:
            # Сдвигаем правую границу
            f = np.array([f[0], f[1], f[1], f[2]])
            phimax = phir
            phir = phil
            phil = phimin * gr + phimax * grp
            f[1] = upperBoundBellman2DRotatedRectangle(x, y, phil)
    
    # Возвращаем максимальное значение
    return np.max(f)
