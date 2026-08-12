import numpy as np
from scipy.optimize import linprog

from bounds.convex.samples import collinear_sample


class AffineFunction:

    __slots__ = ("g", "c", "certified", "origin")

    def __init__(self, g, c, certified=True, origin=None):
        self.g = np.asarray(g, dtype=float)
        self.c = float(c)
        self.certified = bool(certified)
        self.origin = origin

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        return float(self.g @ z + self.c)

    def __repr__(self):
        return "AffineFunction(g=%s, c=%.6g, certified=%s)" % (
            np.round(self.g, 6), self.c, self.certified)


def cut_from_sample(z0, J0, grad):
    g = np.asarray(grad, dtype=float)
    return AffineFunction(g, J0 - g @ np.asarray(z0, float),
                          certified=True, origin="tangent")


def cut_at_collinear(X, Y, theta):
    return cut_from_sample(*collinear_sample(X, Y, theta))


def best_collinear_cut(z, n_restarts=20, seed=0, scale=None):
    from scipy.optimize import minimize

    z = np.asarray(z, dtype=float)
    if scale is None:
        scale = max(float(np.linalg.norm(z[2:])),
                    float(np.linalg.norm(z[:2])) ** 0.5, 1e-6)

    rng = np.random.default_rng(seed)
    best, best_cut = -np.inf, None
    for _ in range(n_restarts):
        p0 = (rng.uniform(-2, 2) * scale**2,
              rng.uniform(-2, 2) * scale,
              rng.uniform(0.0, 2.0 * np.pi))
        res = minimize(lambda p: -cut_at_collinear(*p)(z), p0,
                       method="Nelder-Mead",
                       options={"xatol": 1e-11, "fatol": 1e-16,
                                "maxiter": 2000})
        if -res.fun > best:
            best, best_cut = -res.fun, cut_at_collinear(*res.x)
    return best_cut


def verify_cuts(cuts, reference_upper, points):
    worst = -np.inf
    for z in points:
        ub = reference_upper(z)
        for l in cuts:
            worst = max(worst, l(z) - ub)
    return worst <= 0.0, float(worst)


def compute_lower_affine(S, f_values, current_lower=None, targets=None,
                         grid=None, gradients=None):
    S = np.atleast_2d(np.asarray(S, dtype=float))
    f_values = np.asarray(f_values, dtype=float).ravel()

    if gradients is not None:
        gradients = np.atleast_2d(np.asarray(gradients, dtype=float))
        return [cut_from_sample(S[k], f_values[k], gradients[k])
                for k in range(len(S))]

    targets = S if targets is None else np.atleast_2d(
        np.asarray(targets, dtype=float))

    rows = [np.concatenate([S[k], [1.0]]) for k in range(len(S))]
    rhs = list(f_values)
    if grid is not None and current_lower is not None:
        grid = np.atleast_2d(np.asarray(grid, dtype=float))
        for z in grid:
            rows.append(np.concatenate([z, [1.0]]))
            rhs.append(float(current_lower(z)))

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    cuts = []
    for z_star in targets:
        c_obj = -np.concatenate([z_star, [1.0]])          
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * 5, method="highs")
        if res.success:
            cuts.append(AffineFunction(res.x[:4], res.x[4],
                                       certified=False, origin="lp"))
    return cuts


class AffineCutLowerBound:

    def __init__(self, cuts=(), n_restarts=20, seed=0, adaptive=True):
        self.cuts = list(cuts)
        self.n_restarts = n_restarts
        self.seed = seed
        self.adaptive = adaptive

    def add(self, cut):
        self.cuts.append(cut)
        return self

    def cost(self, z):
        z = np.asarray(z, dtype=float)
        vals = [l(z) for l in self.cuts]
        if self.adaptive:
            vals.append(best_collinear_cut(
                z, n_restarts=self.n_restarts, seed=self.seed)(z))
        return max(vals) if vals else 0.0

    def omega(self, x, y):
        return -self.cost(np.concatenate([np.asarray(x, float),
                                          np.asarray(y, float)]))
    