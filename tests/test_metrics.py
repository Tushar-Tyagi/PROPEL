"""
Unit tests for propel.metrics.
"""

import pytest
import numpy as np
from propel.metrics import (
    compute_ndcg,
    compute_hit,
    compute_mrr,
    kendall_tau_distance,
)


def test_compute_ndcg():
    """Verify NDCG calculation at rank 0, 1, 2..."""
    ranked = ["A", "B", "C", "D"]
    # Target item A at rank 0 -> 1 / log2(0 + 2) = 1.0
    assert compute_ndcg("A", ranked, k=10) == 1.0
    # Target item B at rank 1 -> 1 / log2(1 + 2) = 1 / log2(3) ≈ 0.6309
    assert pytest.approx(compute_ndcg("B", ranked, k=10), rel=1e-4) == 1.0 / np.log2(3)
    # Target item not in top-2 -> 0.0
    assert compute_ndcg("C", ranked, k=2) == 0.0
    # Target item not present at all
    assert compute_ndcg("Z", ranked, k=10) == 0.0


def test_compute_hit():
    """Verify Hit@K."""
    ranked = ["A", "B", "C"]
    assert compute_hit("A", ranked, k=1) == 1.0
    assert compute_hit("B", ranked, k=1) == 0.0
    assert compute_hit("B", ranked, k=2) == 1.0


def test_compute_mrr():
    """Verify Mean Reciprocal Rank."""
    ranked = ["A", "B", "C"]
    assert compute_mrr("A", ranked) == 1.0
    assert pytest.approx(compute_mrr("B", ranked)) == 0.5
    assert pytest.approx(compute_mrr("C", ranked)) == 1.0 / 3.0


def test_kendall_tau_distance():
    """Verify Kendall-Tau distance between rankings."""
    r1 = ["A", "B", "C"]
    r2 = ["C", "B", "A"]
    # 3 discordant pairs: (A,B), (A,C), (B,C)
    assert kendall_tau_distance(r1, r2) == 3
    # Same ranking: 0 discordant pairs
    assert kendall_tau_distance(r1, r1) == 0
