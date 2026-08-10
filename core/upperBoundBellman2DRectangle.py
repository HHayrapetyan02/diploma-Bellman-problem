import numpy as np

from core.pureBellman1D import pureBellman1D
from core.upperBoundBellman2DRotatedRectangle import upperBoundBellman2DRotatedRectangle


GR = (np.sqrt(5) - 1) / 2
GRp = 1 - GR
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
    
    nxy = np.linalg.norm([x, y])
    
    # x = y = 0
    if nxy < tol:
        return 0.0
    
    # x = 0
    nx = np.linalg.norm(x)
    if nx < tol:
        return pureBellman1D(0, np.linalg.norm(y))
    
    # x || y
    det_val = np.linalg.det(np.column_stack((x, y)))
    if abs(det_val) / (nxy**2) < tol:
        return pureBellman1D(nx, np.dot(x, y) / nx)
    
    h = 2**(-6)
    P = np.arange(0, 1 + h, h) * np.pi/2
    F = np.zeros(len(P))
    
    for k in range(len(P)):
        phi = P[k]
        F[k] = upperBoundBellman2DRotatedRectangle(x, y, phi)
    
    maxind = np.argmax(F)
    h = P[1] - P[0]  
    
    phimin = P[maxind] - h
    phimax = P[maxind] + h
    
    phil = phimin * GR + phimax * GRp
    phir = phimax * GR + phimin * GRp
    
    f = np.array([
        upperBoundBellman2DRotatedRectangle(x, y, phimin),
        upperBoundBellman2DRotatedRectangle(x, y, phil),
        upperBoundBellman2DRotatedRectangle(x, y, phir),
        upperBoundBellman2DRotatedRectangle(x, y, phimax)
    ])
    
    while phimax - phimin > 1e-10:
        if (f[0] >= f[1]) or (f[2] <= f[3]):
            break
        
        if f[1] < f[2]:
            f = np.array([f[1], f[2], f[2], f[3]])
            phimin = phil
            phil = phir
            phir = phimax * GR + phimin * GRp
            f[2] = upperBoundBellman2DRotatedRectangle(x, y, phir)
        else:
            f = np.array([f[0], f[1], f[1], f[2]])
            phimax = phir
            phir = phil
            phil = phimin * GR + phimax * GRp
            f[1] = upperBoundBellman2DRotatedRectangle(x, y, phil)
    
    return np.max(f)
