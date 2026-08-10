import numpy as np
from scaled1D import scaledBellman1D


def upperBoundBellman2DAngleRectangle(x, y, phi, xi):
    """
    Upper bound on optimal value (lower bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in a rotated by angle phi
    centered rectangle with corner (cos(xi), sin(xi))
    x, y are the 2D arguments (should be numpy arrays of length 2)
    """
    # Проверка условия на xi (должен быть в пределах [0, pi/2])
    assert 0 <= xi <= np.pi/2, f"xi must be in [0, pi/2], got {xi}"
    
    c = np.cos(phi)
    s = np.sin(phi)
    
    # Матрица поворота
    O = np.array([[c, -s], [s, c]])
    
    # Поворачиваем векторы x и y
    x_rotated = O @ x
    y_rotated = O @ y
    
    # Обработка граничных случаев
    if xi == 0:
        # Прямоугольник вырождается в отрезок вдоль первой оси
        # Проверяем, что вторая компонента равна нулю
        if np.allclose([x_rotated[1], y_rotated[1]], [0, 0]):
            v = scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi))
        else:
            v = float('inf')
    
    elif np.isclose(xi, np.pi/2):
        # Прямоугольник вырождается в отрезок вдоль второй оси
        # Проверяем, что первая компонента равна нулю
        if np.allclose([x_rotated[0], y_rotated[0]], [0, 0]):
            v = scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi))
        else:
            v = float('inf')
    
    else:
        # Общий случай: прямоугольник не вырожден
        v = (scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi)) + 
             scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi)))
    
    return v


# Заглушка для функции scaledBellman1D
# def scaledBellman1D(x_component, y_component, scale):
#     """
#     Вычисляет масштабированную Bellman функцию для 1D случая
#     Это заглушка - нужно реализовать реальную логику
#     """
#     # TODO: Реализовать вычисление scaledBellman1D
#     # В MATLAB это был отдельный файл scaledBellman1D.m
#     # Пример заглушки:
#     return abs(x_component) * scale + abs(y_component) * scale**2  # Пример, замените на реальную формулу