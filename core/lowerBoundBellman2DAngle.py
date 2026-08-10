import numpy as np
from core.pureBellman1D import pureBellman1D


def lowerBoundBellman2DAngle(x, y, phi):
    """
    Lower bound on optimal value (upper bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in a unit square [-1,1]
    rotated by an angle phi
    x, y are the 2D arguments (should be numpy arrays of length 2)
    """
    c = np.cos(phi)
    s = np.sin(phi)
    
    O = np.array([[c, -s], [s, c]])
    x_rotated = O @ x
    y_rotated = O @ y
    
    return pureBellman1D(x_rotated[0], y_rotated[0]) + pureBellman1D(x_rotated[1], y_rotated[1])
    