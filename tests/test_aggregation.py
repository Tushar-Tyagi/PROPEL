"""
Unit tests for propel.aggregation (ConsensusAggregator).
"""

import pytest
from propel.aggregation import (
    ConsensusAggregator,
    borda_initialization,
    asymmetric_local_search,
    compute_consistency_clipping_bound,
)


def test_borda_initialization():
    """Verify standard Borda count initialization ranking."""
    rankings = [
        ["A", "B", "C", "D"],
        ["A", "C", "B", "D"],
        ["B", "A", "C", "D"],
    ]
    # A has scores: 3 + 3 + 2 = 8
    # B has scores: 2 + 1 + 3 = 6
    # C has scores: 1 + 2 + 1 = 4
    # D has scores: 0 + 0 + 0 = 0
    consensus = borda_initialization(rankings)
    assert consensus == ["A", "B", "C", "D"]


def test_consistency_clipping_bound():
    """Verify consistency clip computation C = clip(2 * p_bar / (1 - p_bar), 1, 15)."""
    # 1. Unanimous agreement: p_bar = 1.0 -> should clip at max_c (15.0)
    unanimous = [
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "B", "C"],
    ]
    c_val, p_bar = compute_consistency_clipping_bound(unanimous)
    assert p_bar == 1.0
    assert c_val == 15.0

    # 2. Maximum noise: p_bar = 0.5 -> 2 * (0.5 / 0.5) = 2.0
    split = [
        ["A", "B"],
        ["B", "A"],
    ]
    c_val, p_bar = compute_consistency_clipping_bound(split)
    assert pytest.approx(p_bar, rel=1e-5) == 0.5
    assert pytest.approx(c_val, rel=1e-5) == 2.0


def test_asymmetric_local_search_swaps():
    """Verify asymmetric propensity ratio local search."""
    # Suppose B was presented at position 20 (w=4.0) and A at position 1 (w=0.25).
    # If LLM ranked B above A, the vote for B is weighted by 4.0 / 0.25 = 16.0 (clipped to 15.0).
    initial_consensus = ["A", "B"]
    rankings = [["B", "A"]]
    original_candidate_orders = [["A", "B"]]  # A was at prompt pos 1, B at prompt pos 2
    propensity_weights = {1: 0.25, 2: 4.0}

    final_consensus, num_swaps = asymmetric_local_search(
        initial_consensus=initial_consensus,
        rankings=rankings,
        original_candidate_orders=original_candidate_orders,
        propensity_weights=propensity_weights,
        clip_val=15.0,
    )
    assert final_consensus == ["B", "A"]
    assert num_swaps == 1


def test_consensus_aggregator_end_to_end():
    """Test full ConsensusAggregator aggregate method."""
    aggregator = ConsensusAggregator()
    rankings = [
        ["A", "B", "C"],
        ["A", "C", "B"],
    ]
    original_candidate_orders = [
        ["A", "B", "C"],
        ["C", "B", "A"],
    ]
    propensity_weights = {1: 0.5, 2: 1.0, 3: 2.0}

    consensus, metadata = aggregator.aggregate(
        rankings=rankings,
        original_candidate_orders=original_candidate_orders,
        propensity_weights=propensity_weights,
    )
    assert len(consensus) == 3
    assert "clip_bound" in metadata
    assert "consistency_rate" in metadata
    assert "num_swaps" in metadata
