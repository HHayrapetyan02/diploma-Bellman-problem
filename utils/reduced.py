import numpy as np

from utils.geometry import to_pq

P_SELF_SIMILAR = 2.0 / 3.0
Q_SELF_SIMILAR = -2.0 * np.sqrt(6.0) / 9.0


def reduced_rates(x, y, u):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    u = np.asarray(u, float)

    n = float(np.linalg.norm(y))
    e_y = y / n
    p, q = to_pq(x, y)

    x_perp = x - (x @ e_y) * e_y
    norm_perp = float(np.linalg.norm(x_perp))
    e_perp = x_perp / norm_perp if norm_perp > 1e-14 \
        else np.array([-e_y[1], e_y[0]])

    a = float(u @ e_y)
    b = float(u @ e_perp)

    p_dot = (2.0 * q - 4.0 * p * a) / n
    q_dot = (1.0 - 2.0 * q * a + np.sqrt(max(p - q * q, 0.0)) * b) / n
    return p_dot, q_dot, a


def hjb_residual(p, q, f, f_p, f_q):
    drift = 2.0 * q * f_p + f_q
    ctrl = np.hypot(5.0 * f - 4.0 * p * f_p - 2.0 * q * f_q,
                    np.sqrt(max(p - q * q, 0.0)) * f_q)
    return drift + ctrl - 0.5 * p