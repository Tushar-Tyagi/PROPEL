"""
Unit tests for propel.profiler (BiasProfiler).
"""

import pytest
from propel.profiler import BiasProfiler, compute_segment_boundaries


def test_segment_boundaries():
    """Verify segment boundary calculations for N=20."""
    prim, mid, rec = compute_segment_boundaries(20)
    # Primacy: positions 1..5 (25%)
    assert prim == (1, 5)
    # Middle: positions 6..15 (50%)
    assert mid == (6, 15)
    # Recency: positions 16..20 (25%)
    assert rec == (16, 20)


def test_bias_profiler_uniform_simulation():
    """Simulating uniform item placement across many trials should yield near-zero bias."""
    profiler = BiasProfiler(N=20, top_k=10)
    cands = [f"Item_{i}" for i in range(1, 21)]

    # 1000 uniform trials
    for i in range(1000):
        # Deterministic cyclic shift so each position is selected equally
        top_k_items = [cands[(i + j) % 20] for j in range(10)]
        profiler.record_trial(prompt_candidate_order=cands, llm_ranked_items=top_k_items)

    b_prim, b_rec, b_mid = profiler.estimate_bias_coefficients()
    assert pytest.approx(b_prim, abs=0.05) == 0.0
    assert pytest.approx(b_rec, abs=0.05) == 0.0
    assert pytest.approx(b_mid, abs=0.05) == 0.0


def test_bias_profiler_primacy_simulation():
    """Simulating severe primacy bias where top 5 positions are always selected."""
    profiler = BiasProfiler(N=20, top_k=5)
    cands = [f"Item_{i}" for i in range(1, 21)]

    for _ in range(500):
        # LLM always selects items from prompt positions 1..5
        ranked = cands[:5] + cands[5:]
        profiler.record_trial(prompt_candidate_order=cands, llm_ranked_items=ranked)

    b_prim, b_rec, b_mid = profiler.estimate_bias_coefficients()
    # Primacy should be strongly positive
    assert b_prim > 1.0
    # Recency and middle should be negative (neglected)
    assert b_rec < 0.0
    assert b_mid < 0.0
