import numpy as np


class Constants:
    BETA = 0.444623560185937                   # 36*β^4 + 3*β^2 - 2 = 0
    GAMMA = 6.753024861778741e-02               # γ = (-β^2 + 2β - 2/3) / (10*(1-2β)^(3/2))               
    GR = (np.sqrt(5.0) - 1.0) / 2.0
    GRP = 1.0 - GR
    P_SELF_SIMILAR = 2.0 / 3.0
    Q_SELF_SIMILAR = -2.0 * np.sqrt(6.0) / 9.0
    EPS = 1e-13
