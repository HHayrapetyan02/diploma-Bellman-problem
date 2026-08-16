import warnings

import numpy as np
from scipy.optimize import linprog, minimize

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
        return "AffineFunction(g=%s, c=%.6g, certified=%s, origin=%s)" % (
            np.round(self.g, 6), self.c, self.certified, self.origin)


class AffineEnvelope:
    def __init__(self, functions=(), mode="max", require_certified=True):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min', got %r" % mode)
        self.mode = mode
        self.require_certified = require_certified
        self.functions = list(functions)

    def add(self, fn):
        self.functions.append(fn)
        return self

    def active(self):
        if not self.require_certified:
            return self.functions
        return [f for f in self.functions if f.certified]

    def __call__(self, z):
        vals = [f(z) for f in self.active()]
        if not vals:
            return 0.0 if self.mode == "max" else np.inf
        return max(vals) if self.mode == "max" else min(vals)


def cut_from_sample(z0, J0, grad):
    g = np.asarray(grad, dtype=float)
    return AffineFunction(g, J0 - g @ np.asarray(z0, float),
                          certified=True, origin="tangent")


def cut_at_collinear(X, Y, theta):
    return cut_from_sample(*collinear_sample(X, Y, theta))


def best_collinear_cut(z, n_restarts=20, seed=0, scale=None):
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
        try:
            res = minimize(lambda p: -cut_at_collinear(*p)(z), p0,
                           method="Nelder-Mead",
                           options={"xatol": 1e-11, "fatol": 1e-16,
                                    "maxiter": 2000})
        except Exception:
            continue
        if np.isfinite(res.fun) and -res.fun > best:
            best, best_cut = -res.fun, cut_at_collinear(*res.x)
    if best_cut is None:                      
        return AffineFunction(np.zeros(4), 0.0, certified=True,
                              origin="trivial")   
    return best_cut


def verify_cuts(cuts, reference_upper, points):
    worst = -np.inf
    for z in points:
        ub = reference_upper(z)
        for l in cuts:
            worst = max(worst, l(z) - ub)
    return worst <= 0.0, float(worst)


def compute_lower_affine(S, f_values, current_lower=None, targets=None,
                         grid=None, gradients=None, verbose=False):
    S = np.atleast_2d(np.asarray(S, dtype=float))
    f_values = np.asarray(f_values, dtype=float).ravel()
    if len(S) != len(f_values):
        raise ValueError("S and f_values must have the same length")

    if gradients is not None:
        gradients = np.atleast_2d(np.asarray(gradients, dtype=float))
        return [cut_from_sample(S[k], f_values[k], gradients[k])
                for k in range(len(S))]

    if grid is None or current_lower is None:
        warnings.warn(
            "без ограничений на всей области ЛП-режим не даёт корректной "
            "нижней оценки: по двойственности он вычисляет оценку Йенсена "
            "(сверху). Передайте gradients либо grid+current_lower.",
            RuntimeWarning, stacklevel=2)

    targets = S if targets is None else np.atleast_2d(
        np.asarray(targets, dtype=float))

    rows = [np.concatenate([S[k], [1.0]]) for k in range(len(S))]
    rhs = list(f_values)
    if grid is not None and current_lower is not None:
        for z in np.atleast_2d(np.asarray(grid, dtype=float)):
            rows.append(np.concatenate([z, [1.0]]))
            rhs.append(float(current_lower(z)))

    A_ub, b_ub = np.array(rows), np.array(rhs)

    cuts, failed = [], 0
    for z_star in targets:
        res = linprog(-np.concatenate([z_star, [1.0]]), A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * 5, method="highs")
        if res.success:
            cuts.append(AffineFunction(res.x[:4], res.x[4],
                                       certified=False, origin="lp"))
        else:
            failed += 1                      
    if failed and verbose:
        print("compute_lower_affine: %d/%d ЛП не решены (status: "
              "неограниченность — мало ограничений)" % (failed, len(targets)))
    return cuts


class AffineCutLowerBound:
    def __init__(self, cuts=(), n_restarts=20, seed=0, adaptive=True,
                 require_certified=True):
        self.envelope = AffineEnvelope(cuts, mode="max",
                                       require_certified=require_certified)
        self.n_restarts = n_restarts
        self.seed = seed
        self.adaptive = adaptive

    @property
    def cuts(self):
        return self.envelope.functions

    def add(self, cut):
        self.envelope.add(cut)
        return self

    def cost(self, z):
        z = np.asarray(z, dtype=float)
        val = self.envelope(z)
        if self.adaptive:
            val = max(val, best_collinear_cut(
                z, n_restarts=self.n_restarts, seed=self.seed)(z))
        return float(val)

    def omega(self, x, y):
        return -self.cost(np.concatenate([np.asarray(x, float),
                                          np.asarray(y, float)]))
    