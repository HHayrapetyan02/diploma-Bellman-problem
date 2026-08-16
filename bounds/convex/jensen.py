import numpy as np
from scipy.optimize import linprog

from bounds.convex.samples import collinear_samples, self_similar_samples


def build_dictionary(z, n_state=11, n_dir=32, n_rot=64, n_lambda=9,
                     use_self_similar=True, extra_atoms=()):
    z = np.asarray(z, dtype=float)
    scale = max(float(np.linalg.norm(z[2:])),
                float(np.linalg.norm(z[:2])) ** 0.5, 1e-6)

    atoms = [(np.zeros(4), 0.0)]                      
    atoms += [(zk, Jk) for zk, Jk, _ in
              collinear_samples(scale=scale, n_state=n_state, n_dir=n_dir)]
    if use_self_similar:
        atoms += self_similar_samples(z, n_rot=n_rot, n_lambda=n_lambda)
    atoms += [(np.asarray(zi, dtype=float), float(Ji)) for zi, Ji in extra_atoms]
    return atoms


def compute_upper_affine(S, f_values, z, current_upper=None,
                         return_affine=True):
    S = np.atleast_2d(np.asarray(S, dtype=float))
    f_values = np.asarray(f_values, dtype=float).ravel()
    if len(S) != len(f_values):
        raise ValueError("S and f_values must have the same length")
    z = np.asarray(z, dtype=float)
    if z.shape != (4,):
        raise ValueError("z must be a 4-vector (x_1, x_2, y_1, y_2)")

    A_eq = np.vstack([S.T, np.ones(len(f_values))])
    b_eq = np.concatenate([z, [1.0]])
    res = linprog(f_values, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0.0, None)] * len(f_values), method="highs")
    if not res.success:
        return np.inf, None, None

    value = float(res.fun)
    affine = None
    if return_affine and getattr(res, "eqlin", None) is not None:
        from bounds.convex.affine import AffineFunction
        duals = np.asarray(res.eqlin.marginals, dtype=float)
        affine = AffineFunction(duals[:4], duals[4], certified=False,
                                origin="jensen-dual")

    if current_upper is not None:
        value = min(value, float(current_upper(z)))
    return value, res.x, affine


class JensenUpperBound:
    def __init__(self, n_state=11, n_dir=32, n_rot=64, n_lambda=9,
                 use_self_similar=True, extra_atoms=()):
        self.kwargs = dict(n_state=n_state, n_dir=n_dir, n_rot=n_rot,
                           n_lambda=n_lambda, use_self_similar=use_self_similar,
                           extra_atoms=tuple(extra_atoms))

    def cost(self, z, return_weights=False, return_affine=False):
        z = np.asarray(z, dtype=float)
        atoms = build_dictionary(z, **self.kwargs)
        S = np.array([a for a, _ in atoms])
        f = np.array([v for _, v in atoms])
        value, weights, affine = compute_upper_affine(
            S, f, z, return_affine=return_affine)
        if return_weights and return_affine:
            return value, weights, S, affine
        if return_weights:
            return value, weights, S
        if return_affine:
            return value, affine
        return value

    def omega(self, x, y):
        return -self.cost(np.concatenate([np.asarray(x, float),
                                          np.asarray(y, float)]))
    