import numpy as np
from core.lowerBoundBellman2DAngle import lowerBoundBellman2DAngle

GR = (np.sqrt(5) - 1) / 2
GRp = 1 - GR
    
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
    
    P = np.arange(0, 0.5 * np.pi + 0.01 * np.pi, 0.01 * np.pi)
    F = np.zeros(len(P))
    
    for k in range(len(P)):
        phi = P[k]
        F[k] = lowerBoundBellman2DAngle(x, y, phi)
    
    minind = np.argmin(F)
    h = P[1] - P[0]  # Шаг сетки
    
    phimin = P[minind] - h
    phimax = P[minind] + h
    
    GR = (np.sqrt(5) - 1) / 2
    GRp = 1 - GR
    
    phil = phimin * GR + phimax * GRp
    phir = phimax * GR + phimin * GRp
    
    f = np.array([
        lowerBoundBellman2DAngle(x, y, phimin),
        lowerBoundBellman2DAngle(x, y, phil),
        lowerBoundBellman2DAngle(x, y, phir),
        lowerBoundBellman2DAngle(x, y, phimax)
    ])
    
    while phimax - phimin > 1e-10:
        if (f[0] <= f[1]) or (f[2] >= f[3]):
            break
        
        if f[1] > f[2]:
            f = np.array([f[1], f[2], f[2], f[3]])
            phimin = phil
            phil = phir
            phir = phimax * GR + phimin * GRp
            f[2] = lowerBoundBellman2DAngle(x, y, phir)
        else:
            f = np.array([f[0], f[1], f[1], f[2]])
            phimax = phir
            phir = phil
            phil = phimin * GR + phimax * GRp
            f[1] = lowerBoundBellman2DAngle(x, y, phil)
    
    return np.min(f)
