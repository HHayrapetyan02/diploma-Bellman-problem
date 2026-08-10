import numpy as np
from core.scaled1D import scaledBellman1D


def upperBoundBellman2DAngleRectangle(x, y, phi, xi):
    """
    Upper bound on optimal value (lower bound on Bellman function)
    for 2D problem with control in the disc
    Computes value of Bellman function for control in a rotated by angle phi
    centered rectangle with corner (cos(xi), sin(xi))
    x, y are the 2D arguments (should be numpy arrays of length 2)
    """
    assert 0 <= xi <= np.pi/2, f"xi must be in [0, pi/2], got {xi}"
    
    c = np.cos(phi)
    s = np.sin(phi)
    
    O = np.array([[c, -s], [s, c]])
    x_rotated = O @ x
    y_rotated = O @ y
    
    if xi == 0:
        if np.allclose([x_rotated[1], y_rotated[1]], [0, 0]):
            v = scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi))
        else:
            v = float('inf')
    
    elif np.isclose(xi, np.pi/2):
        if np.allclose([x_rotated[0], y_rotated[0]], [0, 0]):
            v = scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi))
        else:
            v = float('inf')
    
    else:
        v = (scaledBellman1D(x_rotated[0], y_rotated[0], np.cos(xi)) + 
             scaledBellman1D(x_rotated[1], y_rotated[1], np.sin(xi)))
    
    return v

