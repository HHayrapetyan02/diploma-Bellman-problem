import numpy as np


# 36*β^4 + 3*β^2 - 2 = 0
# γ = (-β^2 + 2β - 2/3) / (10*(1-2β)^(3/2))    
BETTA = 0.444623560185937
GAMMA = 6.753024861778741e-02
    
def pureBellman1D(x, y):
    """
    Computes 1D Bellman function for Fuller problem with interval [-1,+1]
    x, y are the arguments of the Bellman function
    The function is the negative of the cost, hence nonpositive
    """
    if x >= -BETTA * y * abs(y):
        v = -x**2 * y / 2 - x * y**3 / 3 - y**5 / 15 - GAMMA * (y**2 + 2*x)**(5/2)
    else:
        v = x**2 * y / 2 - x * y**3 / 3 + y**5 / 15 - GAMMA * (y**2 - 2*x)**(5/2)
    
    return v