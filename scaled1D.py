from pureBellman1D import pureBellman1D

def scaledBellman1D(x, y, a):
    """
    computes 1D Bellman function for Fuller problem with symmetric interval [-a,+a]
    x,y are the arguments of the Bellman function
    the function is the negative of the cost, hence nonpositive
    """
    assert a > 0, 'scaling factor must be positive'
    return pureBellman1D(a * x, y) / (a ** 3)