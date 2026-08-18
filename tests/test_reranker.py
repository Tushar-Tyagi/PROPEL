"""
Integration tests for propel.reranker (PropelReranker / PROPEL).
"""

import pytest
from propel import PROPEL, LLMClient, UserProfile


def test_propel_rerank_end_to_end():
    """End-to-end mocked rerank test."""
    # Deterministic mock LLM: preserves prompt order (which exhibits primacy bias)
    def mock_llm_fn(prompt: str) -> str:
        return '{"ranked_movies": [1, 2, 3, 4, 5]}'

    client = LLMClient(custom_caller=mock_llm_fn)
    reranker = PROPEL(
        model="gpt-4o-mini",
        dataset="ml-1m",
        llm_client=client,
    )

    history = [{"title": "Film Past", "genres": ["Action"], "rating": 5.0}]
    candidates = [
        {"title": f"Candidate_{i}", "genres": ["Action"]} for i in range(1, 6)
    ]

    final_ranking, report = reranker.rerank(
        user_history=history,
        candidates=candidates,
        num_shuffles=5,
        seed=42,
    )

    assert len(final_ranking) == 5
    assert report.model == "gpt-4o-mini"
    assert report.dataset == "ml-1m"
    assert "clip_bound" in report.aggregation_metadata
    assert len(report.item_adjustments) == 5


def test_propel_evaluate_profiles():
    """Test batch profile evaluation."""
    def mock_llm_fn(prompt: str) -> str:
        return '{"ranked_movies": [1, 2, 3, 4, 5]}'

    client = LLMClient(custom_caller=mock_llm_fn)
    reranker = PROPEL(model="gpt-4o-mini", dataset="ml-1m", llm_client=client)

    profiles = [
        UserProfile(
            user_id="user_1",
            history=[{"title": "H1"}],
            candidates=[{"title": f"C_{i}"} for i in range(1, 6)],
            ground_truth={"title": "C_1"},
        ),
        UserProfile(
            user_id="user_2",
            history=[{"title": "H2"}],
            candidates=[{"title": f"C_{i}"} for i in range(1, 6)],
            ground_truth={"title": "C_5"},
        ),
    ]

    metrics = reranker.evaluate_profiles(profiles, num_shuffles=3)
    assert "hit@1" in metrics
    assert "ndcg@5" in metrics
    assert "ndcg@20" in metrics
    assert "mrr" in metrics
    assert metrics["num_evaluated_users"] == 2
