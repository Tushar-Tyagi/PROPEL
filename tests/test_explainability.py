"""
Unit tests for propel.explainability (ExplainabilityReport).
"""

import json
from propel.explainability import generate_explainability_report


def test_explainability_report_structure():
    """Verify structured explainability report generation and JSON serialization."""
    report = generate_explainability_report(
        recommendation_id="rec_test_123",
        dataset="ml-1m",
        model="gpt-4o-mini",
        bias_coefficients={"B_prim": 1.834, "B_rec": -0.782, "B_mid": -0.526},
        propensity_curve=[(1, 2.5), (2, 2.0), (3, 1.0)],
        rankings=[["A", "B", "C"], ["B", "A", "C"]],
        original_candidate_orders=[["A", "B", "C"], ["C", "B", "A"]],
        propensity_weights={1: 0.4, 2: 1.0, 3: 2.5},
        initial_borda_ranking=["A", "B", "C"],
        final_consensus_ranking=["B", "A", "C"],
        aggregation_metadata={"clip_bound": 2.0, "consistency_rate": 0.6, "num_swaps": 1},
    )

    d = report.to_dict()
    assert d["recommendation_id"] == "rec_test_123"
    assert d["final_ranking"] == ["B", "A", "C"]
    assert len(d["item_adjustments"]) == 3

    # Check rank shift: B was at borda pos 2, final pos 1 -> shift = +1
    b_adj = next(a for a in d["item_adjustments"] if a["item_title"] == "B")
    assert b_adj["rank_shift"] == 1

    # Check JSON export
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["recommendation_id"] == "rec_test_123"

    # Check human-readable summary
    summary = report.summary()
    assert "PROPEL Explainability Report" in summary
    assert "Top-5 Final Recommendations" in summary
