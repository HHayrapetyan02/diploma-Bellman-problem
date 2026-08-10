import numpy as np
from core.upperBoundBellman2DAngleRectangle import upperBoundBellman2DAngleRectangle
from core.pureBellman1D import pureBellman1D

def upperBoundBellman2DRotatedRectangle(x, y, phi):
    """
    Upper bound on optimal value (lower bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in the centered rectangle
    rotated by phi and with best side ratio
    x, y are the 2D arguments
    The best side ratio is sought by the golden ratio method
    The value has to be maximized over the side ratio
    """
    tol = 1e-14
    
    c = np.cos(phi)
    s = np.sin(phi)
    O = np.array([[c, -s], [s, c]])
    
    x_rotated = O @ x
    y_rotated = O @ y
    
    if (abs(x_rotated[0]) < tol) and (abs(y_rotated[0]) < tol):
        return pureBellman1D(x_rotated[1], y_rotated[1])
    
    if (abs(x_rotated[1]) < tol) and (abs(y_rotated[1]) < tol):
        return pureBellman1D(x_rotated[0], y_rotated[0])
    
    h = 2**(-6)
    
    while True:
        X = np.arange(0, 1 + h, h) * np.pi/2
        X = X[1:-1]
        
        F = np.zeros(len(X))
        
        for k in range(len(X)):
            xi = X[k]
            F[k] = upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, xi)
        if (len(F) >= 2) and (F[0] < F[1]) and (F[-2] > F[-1]):
            break
        
        h = h / 2
    
    maxind = np.argmax(F)
    h = X[1] - X[0]  
    
    ximin = X[maxind] - h
    ximax = X[maxind] + h
    
    gr = (np.sqrt(5) - 1) / 2
    grp = 1 - gr
    
    xil = ximin * gr + ximax * grp
    xir = ximax * gr + ximin * grp
    
    f = np.array([
        upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, ximin),
        upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, xil),
        upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, xir),
        upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, ximax)
    ])
    
    while ximax - ximin > 1e-10:
        if (f[0] >= f[1]) or (f[2] <= f[3]):
            break
        
        if f[1] < f[2]:
            f = np.array([f[1], f[2], f[2], f[3]])
            ximin = xil
            xil = xir
            xir = ximax * gr + ximin * grp
            f[2] = upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, xir)
        else:
            f = np.array([f[0], f[1], f[1], f[2]])
            ximax = xir
            xir = xil
            xil = ximin * gr + ximax * grp
            f[1] = upperBoundBellman2DAngleRectangle(x_rotated, y_rotated, 0, xil)
    
    return np.max(f)

