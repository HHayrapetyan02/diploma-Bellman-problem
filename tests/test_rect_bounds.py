import numpy as np
import pytest

from bounds.common import pure_bellman_1d
from bounds.lower.square import LowerBoundBellmanFunction
from bounds.upper.policy_improvement import PolicyImprovementBound
from bounds.upper.rectangle import UpperBoundBellmanFunction
from bounds.upper.self_similar import SelfSimilarControlBound
from bounds.upper.time_optimal import TimeOptimalBound
from utils.geometry import gramian, to_pq
from utils.reduced import reduced_rates
from utils.constants import Constants as Const

OMEGA_EXACT = -1.0 / 540.0            
TIME_OPTIMAL_REF = -0.0019779902706   


def self_similar_state():
    G = np.array([[1 / 54, -1 / 27], [-1 / 27, 1 / 6]])
    L = np.linalg.cholesky(G).T
    return L[:, 0], L[:, 1]


def test_self_similar_gramian_and_invariants():
    x, y = self_similar_state()
    G = np.array([[1 / 54, -1 / 27], [-1 / 27, 1 / 6]])
    assert np.allclose(gramian(x, y), G)
    p, q = to_pq(x, y)
    assert p == pytest.approx(Const.P_SELF_SIMILAR, abs=1e-12)
    assert q == pytest.approx(Const.Q_SELF_SIMILAR, abs=1e-12)


def test_reduced_dynamics_matches_full_system():
    rng = np.random.default_rng(0)
    for _ in range(100):
        x, y = rng.normal(size=2), rng.normal(size=2)
        u = rng.normal(size=2)
        u /= np.linalg.norm(u)
        h = 1e-6
        p0, q0 = to_pq(x, y)
        p1, q1 = to_pq(x + h * y, y + h * u)
        n0, n1 = np.linalg.norm(y), np.linalg.norm(y + h * u)
        num = np.array([(p1 - p0) / h, (q1 - q0) / h, (n1 - n0) / h])
        assert np.allclose(num, reduced_rates(x, y, u), atol=1e-3, rtol=1e-3)


def test_self_similar_point_is_an_equilibrium_with_unit_control():
    p, q = Const.P_SELF_SIMILAR, Const.Q_SELF_SIMILAR
    a = q / (2 * p)                                   
    b = -(1 - 2 * q * a) / np.sqrt(p - q * q)         
    assert a * a + b * b == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- collinear
@pytest.mark.parametrize("x0,y0", [(1.0, 0.3), (-0.7, 1.2), (0.2, -2.0)])
def test_collinear_reduces_to_1d(x0, y0):
    e = np.array([np.cos(0.7), np.sin(0.7)])
    x, y = x0 * e, y0 * e
    ref = float(pure_bellman_1d(x0, y0))
    assert UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y) \
        == pytest.approx(ref, rel=1e-10)
    assert LowerBoundBellmanFunction().lowerBoundBellman2D(x, y) \
        == pytest.approx(ref, rel=1e-10)


# ------------------------------------------------------------- benchmarks
def test_bounds_sandwich_the_exact_value():
    x, y = self_similar_state()
    lo = UpperBoundBellmanFunction().upperBoundBellman2DRectangle(x, y)
    hi = LowerBoundBellmanFunction().lowerBoundBellman2D(x, y)
    assert lo <= OMEGA_EXACT <= hi


def test_time_optimal_reproduces_the_paper():
    x, y = self_similar_state()
    arg, v = TimeOptimalBound().upper_bound_time_optimal(x, y, return_arg=True)
    alpha, tau_bar, tau0 = arg
    assert v == pytest.approx(TIME_OPTIMAL_REF, rel=1e-8)
    assert alpha == pytest.approx(4.13415835032, rel=1e-8)
    assert tau_bar == pytest.approx(0.97116420999, rel=1e-8)
    assert tau0 == pytest.approx(-2.17695799429, rel=1e-8)
    assert v <= OMEGA_EXACT


def test_self_similar_control_is_exact_on_its_own_orbit():
    x, y = self_similar_state()
    v = SelfSimilarControlBound().upper_bound_self_similar(x, y)
    assert v == pytest.approx(OMEGA_EXACT, rel=1e-8)      
    off = SelfSimilarControlBound().upper_bound_self_similar(
        np.array([1.0, 0.3]), np.array([0.1, 0.7]))
    assert off == -np.inf


def test_policy_improvement_is_valid_and_not_worse():
    x, y = self_similar_state()
    pi = PolicyImprovementBound(n_controls=12, h_factors=(0.1, 0.3),
                                rect_points=24)
    base = pi.base(x, y)
    improved = pi.upper_bound_policy_improvement(x, y, n_steps=1)
    assert base <= improved <= OMEGA_EXACT
    