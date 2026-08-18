"""
Unit tests for propel.propensity (PropensityModel).
"""

import pytest
import numpy as np
from propel.propensity import PropensityModel


def test_propensity_zero_bias():
    """When all bias coefficients are 0, probabilities should be uniform and weights should be 1.0."""
    model = PropensityModel(B_prim=0.0, B_rec=0.0, B_mid=0.0)
    N = 20
    probs = model.compute_probabilities(N)
    assert len(probs) == N
    assert np.allclose(probs, 1.0 / N)

    norm_prop = model.get_normalized_propensities(N)
    for p in range(1, N + 1):
        assert pytest.approx(norm_prop[p], rel=1e-5) == 1.0

    weights = model.get_inverse_propensity_weights(N)
    for p in range(1, N + 1):
        assert pytest.approx(weights[p], rel=1e-5) == 1.0


def test_propensity_primacy_bias():
    """Under primacy bias, position 1 should have high probability, S_hat > 1, and w(1) < 1."""
    model = PropensityModel(B_prim=2.0, B_rec=-0.8, B_mid=-0.5)
    N = 20
    probs = model.compute_probabilities(N)
    assert pytest.approx(sum(probs), rel=1e-6) == 1.0
    assert probs[0] > probs[-1]  # Top position is favored over bottom

    weights = model.get_inverse_propensity_weights(N)
    assert weights[1] < 1.0       # Favored position is down-weighted
    assert weights[N] > 1.0       # Disfavored position is up-weighted


def test_propensity_curve_data():
    """Verify propensity curve generation for explainability."""
    model = PropensityModel(B_prim=1.8, B_rec=-0.7, B_mid=-0.5)
    N = 20
    curve = model.get_propensity_curve_data(N, K=10)
    assert len(curve) == N
    positions = [p for p, _ in curve]
    assert positions == list(range(1, N + 1))
    freqs = [f for _, f in curve]
    assert pytest.approx(sum(freqs), rel=1e-5) == 10.0


def test_propensity_single_item():
    """Edge case: N=1."""
    model = PropensityModel(B_prim=2.0, B_rec=-1.0, B_mid=0.0)
    probs = model.compute_probabilities(1)
    assert len(probs) == 1
    assert probs[0] == 1.0
    weights = model.get_inverse_propensity_weights(1)
    assert weights[1] == 1.0
