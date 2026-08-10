import numpy as np

def pureBellman1D(x, y):
    """
    Computes 1D Bellman function for Fuller problem with interval [-1,+1]
    x, y are the arguments of the Bellman function
    The function is the negative of the cost, hence nonpositive
    """
    # Константы, вычисленные из уравнений:
    # 36*β^4 + 3*β^2 - 2 = 0
    # γ = (-β^2 + 2β - 2/3) / (10*(1-2β)^(3/2))
    bet = 0.444623560185937
    gam = 6.753024861778741e-02
    
    # Условие ветвления
    if x >= -bet * y * abs(y):
        # Первая ветка (x >= -β*y*|y|)
        v = -x**2 * y / 2 - x * y**3 / 3 - y**5 / 15 - gam * (y**2 + 2*x)**(5/2)
    else:
        # Вторая ветка (x < -β*y*|y|)
        v = x**2 * y / 2 - x * y**3 / 3 + y**5 / 15 - gam * (y**2 - 2*x)**(5/2)
    
    return v