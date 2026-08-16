import numpy as np
import pytest

from bounds.convex.affine import (AffineCutLowerBound, AffineEnvelope,
                                  best_collinear_cut, compute_lower_affine,
                                  cut_at_collinear, verify_cuts)
from bounds.convex.jensen import (JensenUpperBound, build_dictionary,
                                  compute_upper_affine)
from bounds.convex.samples import (collinear_sample, cost_1d, exact_cost_if_known,
                                   grad_cost_1d, SELF_SIMILAR_COST)
from bounds.lower.square import LowerBoundBellmanFunction
from bounds.upper.rectangle import UpperBoundBellmanFunction

EXACT = SELF_SIMILAR_COST          


def self_similar_state():
    G = np.array([[1 / 54, -1 / 27], [-1 / 27, 1 / 6]])
    L = np.linalg.cholesky(G).T
    return np.concatenate([L[:, 0], L[:, 1]])


def cost_upper(z):
    return -UpperBoundBellmanFunction().upperBoundBellman2DRectangle(
        z[:2], z[2:], n_points=24)


def cost_lower(z):
    return -LowerBoundBellmanFunction().lowerBoundBellman2D(z[:2], z[2:])


def test_analytic_gradient_of_the_1d_cost():
    rng = np.random.default_rng(0)
    for _ in range(200):
        X, Y = rng.normal(size=2) * rng.choice([0.3, 1.0, 3.0])
        h = 1e-6
        num = ((cost_1d(X + h, Y) - cost_1d(X - h, Y)) / (2 * h),
               (cost_1d(X, Y + h) - cost_1d(X, Y - h)) / (2 * h))
        ana = grad_cost_1d(X, Y)
        assert np.allclose(num, ana, atol=1e-5, rtol=1e-5)


def test_cost_is_convex():
    rng = np.random.default_rng(1)
    for _ in range(50):
        z1, z2 = rng.normal(size=4), rng.normal(size=4)
        zm = 0.5 * (z1 + z2)
        assert cost_lower(zm) <= 0.5 * (cost_upper(z1) + cost_upper(z2)) + 1e-12


def test_exact_values_are_recognised():
    assert exact_cost_if_known(self_similar_state()) == pytest.approx(EXACT,
                                                                      rel=1e-9)
    e = np.array([np.cos(0.4), np.sin(0.4)])
    z = np.concatenate([0.7 * e, -1.3 * e])
    assert exact_cost_if_known(z) == pytest.approx(cost_1d(0.7, -1.3), rel=1e-9)
    assert exact_cost_if_known(np.array([1.0, 0.0, 0.0, 1.0])) is None


def test_tangent_cut_touches_its_own_point():
    z0, J0, g = collinear_sample(0.4, -0.9, 1.1)
    l = cut_at_collinear(0.4, -0.9, 1.1)
    assert l(z0) == pytest.approx(J0, rel=1e-12)
    assert np.allclose(l.g, g)


def test_cuts_never_exceed_a_valid_upper_bound():
    rng = np.random.default_rng(2)
    cuts = [cut_at_collinear(rng.normal(), rng.normal(),
                             rng.uniform(0, 2 * np.pi)) for _ in range(30)]
    pts = [rng.normal(size=4) * rng.choice([0.5, 1.0]) for _ in range(12)]
    ok, worst = verify_cuts(cuts, cost_upper, pts)
    assert ok, "supporting-hyperplane property violated by %.3e" % worst


def test_compute_lower_affine_with_gradients_is_certified():
    S, f, g = [], [], []
    for p in [(0.3, 0.5, 0.0), (-0.4, 1.1, 1.0), (0.8, -0.2, 2.2)]:
        z0, J0, gr = collinear_sample(*p)
        S.append(z0); f.append(J0); g.append(gr)
    cuts = compute_lower_affine(np.array(S), np.array(f), gradients=np.array(g))
    assert len(cuts) == 3 and all(c.certified for c in cuts)
    for z0, J0, c in zip(S, f, cuts):
        assert c(z0) == pytest.approx(J0, rel=1e-12)


def test_affine_cut_bound_is_a_valid_lower_bound():
    z = self_similar_state()
    v = AffineCutLowerBound(n_restarts=8).cost(z)
    assert 0.0 < v <= EXACT + 1e-12


def test_jensen_is_exact_on_the_self_similar_orbit():
    z = self_similar_state()
    v = JensenUpperBound(n_state=9, n_dir=24, n_rot=48, n_lambda=7).cost(z)
    assert v == pytest.approx(EXACT, rel=1e-6)
    assert v <= cost_upper(z) + 1e-12          


def test_jensen_is_an_upper_bound():
    rng = np.random.default_rng(3)
    for _ in range(3):
        z = rng.normal(size=4) * 0.7
        v = JensenUpperBound(n_state=9, n_dir=24, n_rot=32, n_lambda=5).cost(z)
        assert v >= cost_lower(z) - 1e-12


def test_compute_upper_affine_duality():
    z = self_similar_state()
    atoms = build_dictionary(z, n_state=7, n_dir=16, n_rot=24, n_lambda=5)
    S = np.array([a for a, _ in atoms])
    f = np.array([v for _, v in atoms])
    value, weights, affine = compute_upper_affine(S, f, z=z)
    assert value == pytest.approx(EXACT, rel=1e-6)
    assert np.all(weights >= -1e-12)
    if affine is not None:
        assert affine(z) == pytest.approx(value, rel=1e-6, abs=1e-9)
        assert np.max(np.array([affine(s) for s in S]) - f) <= 1e-8


def test_uncertified_lp_cuts_are_excluded_by_default():
    import warnings
    from bounds.convex.samples import collinear_sample

    S, f = [], []
    for prm in [(0.3, 0.5, 0.0), (-0.4, 1.1, 1.0), (0.8, -0.2, 2.2),
                (0.1, 0.9, 3.0), (1.2, 0.4, 4.5)]:
        z0, J0, _ = collinear_sample(*prm)
        S.append(z0); f.append(J0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp_cuts = compute_lower_affine(np.array(S), np.array(f))
    assert lp_cuts and all(not c.certified for c in lp_cuts)

    z = self_similar_state()
    safe = AffineCutLowerBound(cuts=lp_cuts, adaptive=False)
    assert safe.cost(z) <= EXACT + 1e-12            
    unsafe = AffineCutLowerBound(cuts=lp_cuts, adaptive=False,
                                 require_certified=False)
    assert unsafe.cost(z) > EXACT                   


def test_affine_envelope_modes():
    z = self_similar_state()
    fns = [cut_at_collinear(0.3, 0.5, 0.0), cut_at_collinear(-0.4, 1.1, 1.0)]
    assert AffineEnvelope(fns, mode="max")(z) == pytest.approx(
        max(f(z) for f in fns))
    assert AffineEnvelope(fns, mode="min")(z) == pytest.approx(
        min(f(z) for f in fns))
    with pytest.raises(ValueError):
        AffineEnvelope(fns, mode="avg")


def test_user_supplied_exact_points_are_used():
    z = self_similar_state()
    je = JensenUpperBound(n_state=5, n_dir=8, n_rot=12, n_lambda=3,
                          use_self_similar=False, extra_atoms=[(z, EXACT)])
    assert je.cost(z) == pytest.approx(EXACT, rel=1e-9)


def test_jensen_dual_affine_is_not_marked_certified():
    z = self_similar_state()
    atoms = build_dictionary(z, n_state=5, n_dir=8, n_rot=12, n_lambda=3)
    S = np.array([a for a, _ in atoms])
    f = np.array([v for _, v in atoms])
    _, _, affine = compute_upper_affine(S, f, z)
    assert affine is not None and not affine.certified
    assert np.max(np.array([affine(s) for s in S]) - f) <= 1e-8
