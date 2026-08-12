import numpy as np
from scipy.integrate import solve_ivp

from bounds.common import pure_bellman_1d
from utils.constants import Constants as Const

SELF_SIMILAR_COST = 1.0 / 540.0    


# --------------------------------------------------------------------- 1D
def cost_1d(X, Y):
    return -float(pure_bellman_1d(X, Y))


def grad_cost_1d(X, Y):
    s = 1.0 if X >= -Const.BETA * Y * abs(Y) else -1.0
    base = max(Y * Y + 2.0 * s * X, 0.0)
    w_x = -s * X * Y - Y**3 / 3.0 - 5.0 * Const.GAMMA * s * base**1.5
    w_y = (-s * (0.5 * X * X + Y**4 / 3.0) - X * Y * Y
           - 5.0 * Const.GAMMA * Y * base**1.5)
    return -w_x, -w_y


def embed(X, Y, theta):
    e = np.array([np.cos(theta), np.sin(theta)])
    return np.concatenate([X * e, Y * e])


# ------------------------------------------------------------- collinear
def collinear_sample(X, Y, theta):
    e = np.array([np.cos(theta), np.sin(theta)])
    g_x, g_y = grad_cost_1d(X, Y)
    return embed(X, Y, theta), cost_1d(X, Y), np.concatenate([g_x * e, g_y * e])


def collinear_samples(scale=1.0, n_state=13, n_dir=32, span=2.0):
    out = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False):
        for X in np.linspace(-span, span, n_state) * scale**2:
            for Y in np.linspace(-span, span, n_state) * scale:
                out.append(collinear_sample(X, Y, theta))
    return out


# --------------------------------------------------------- self-similar
def _self_similar_base(T=1.0, n=60):
    out = []
    for sigma in (1.0, -1.0):
        def rhs(t, z):
            tau = max(T - t, 1e-14)
            phase = sigma * np.sqrt(5) * np.log(tau)
            u = np.array([np.cos(phase), np.sin(phase)])
            return np.concatenate([z[2:4], u, [0.5 * (z[0:2] @ z[0:2])]])

        sol = solve_ivp(rhs, (T, 0.0), np.zeros(5),
                        t_eval=np.linspace(T, 0.0, n),
                        rtol=1e-12, atol=1e-14, method="DOP853")
        for k in range(1, n):
            z = np.concatenate([sol.y[0:2, k], sol.y[2:4, k]])
            out.append((z, -float(sol.y[4, k])))
    return out


_BASE = None


def self_similar_base():
    global _BASE
    if _BASE is None:
        _BASE = _self_similar_base()
    return _BASE


def rotate_state(z, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return np.concatenate([R @ z[:2], R @ z[2:]])


def self_similar_samples(z_query, n_rot=64, n_lambda=9, log_span=1.2):
    z_query = np.asarray(z_query, float)
    ny = float(np.linalg.norm(z_query[2:]))
    if ny <= 0.0:
        return []
    ang_y = np.arctan2(z_query[3], z_query[2])
    lam_q = np.sqrt(6.0) * ny

    out = []
    for lam in lam_q * np.exp(np.linspace(-log_span, log_span, n_lambda)):
        for z_b, J_b in self_similar_base():
            z_s = np.concatenate([z_b[:2] * lam**2, z_b[2:] * lam])
            J_s = J_b * lam**5
            th0 = ang_y - np.arctan2(z_s[3], z_s[2])
            for th in th0 + np.linspace(0.0, 2.0 * np.pi, n_rot, endpoint=False):
                out.append((rotate_state(z_s, th), J_s))
    return out


def exact_cost_if_known(z, tol=1e-9):
    z = np.asarray(z, float)
    x, y = z[:2], z[2:]
    scale = float(np.hypot(np.linalg.norm(x), np.linalg.norm(y)))
    if scale < tol:
        return 0.0

    det = float(x[0] * y[1] - x[1] * y[0])
    if abs(det) / scale**2 < tol:                      
        nx = float(np.linalg.norm(x))
        if nx < tol * scale:
            return cost_1d(0.0, float(np.linalg.norm(y)))
        return cost_1d(nx, float(x @ y) / nx)

    G = np.array([[x @ x, x @ y], [x @ y, y @ y]])     
    G_ref = np.array([[1 / 54, -1 / 27], [-1 / 27, 1 / 6]])
    for s in (1.0, -1.0):
        lam = np.sqrt(6.0) * float(np.linalg.norm(y))
        S = np.array([[lam**4 / 54, s * -lam**3 / 27],
                      [s * -lam**3 / 27, lam**2 / 6]])
        if np.max(np.abs(G - S)) <= tol * max(1.0, np.max(np.abs(S))):
            return SELF_SIMILAR_COST * lam**5
    del G_ref
    return None
