"""
LLM_debias.py - PROPEL Compatibility Layer.

Provides backward-compatible interface and aliases mapping to the modular `propel` package.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from propel import (
    PROPEL,
    PropelReranker,
    PropensityModel,
    BiasProfiler,
    ConsensusAggregator,
    ExplainabilityReport,
    LLMClient,
    build_ranking_prompt,
    parse_ranking_response,
    compute_ndcg,
    compute_hit,
    compute_mrr,
    DEFAULT_BIAS_PARAMS,
)


def get_data_columns(data_name: str) -> Tuple[str, List[str], List[str], List[str]]:
    """Get dataset column mappings."""
    ds = data_name.lower()
    if ds in ["movie_lens", "movielens", "ml-1m"]:
        return "Title", ["Genres"], ["Gender", "Age", "Occupation"], ["Rating"]
    elif ds in ["books", "amazon_books"]:
        return "Title", [], [], []
    elif ds in ["music", "cds_vinyl", "amazon_music"]:
        return "Title", [], [], []
    elif ds in ["beauty", "amazon_beauty"]:
        return "Title", [], [], []
    elif ds in ["steam", "video_games"]:
        return "Title", [], [], []
    else:
        return "Title", [], [], []


def get_api_config(model_name: str, tier: str = "tier_1") -> Dict[str, Any]:
    """Get API rate limit configuration."""
    configs = {
        "basic": {"rpm": 500, "tpm": 200000, "max_workers": 5},
        "tier_1": {"rpm": 3500, "tpm": 1000000, "max_workers": 15},
        "tier_2": {"rpm": 5000, "tpm": 2000000, "max_workers": 30},
    }
    if "gpt" in model_name.lower():
        return configs.get(tier, configs["basic"])
    return {"rpm": 60, "tpm": 10000, "max_workers": 2}


class LLMPositionBiasAnalyzer:
    """
    Backward-compatible analyzer wrapper delegating to PROPEL.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_name: str = "movielens",
        model: str = "gpt-4o-mini",
        backend: str = "openai",
        num_bias_users: int = 5,
        num_eval_users: int = 50,
        num_shuffles_bias: int = 20,
        list_size: int = 20,
        api_tier: str = "tier_1",
    ):
        if data.empty:
            raise ValueError("Input data DataFrame is empty.")
        if "UserID" not in data.columns or "Title" not in data.columns:
            raise ValueError("DataFrame must contain 'UserID' and 'Title' columns.")

        self.data = data
        self.data_name = data_name
        self.model = model
        self.backend = backend
        self.num_bias_users = num_bias_users
        self.num_eval_users = num_eval_users
        self.num_shuffles = num_shuffles_bias
        self.list_size = list_size
        self.api_tier = api_tier
        self.api_config = get_api_config(model, api_tier)

        self.middle_start = int(0.25 * list_size)
        self.middle_end = int(0.75 * list_size)

        # Users with sufficient data
        user_counts = self.data["UserID"].value_counts()
        eligible_users = list(user_counts[user_counts >= 5].index)
        self.bias_users = eligible_users[:num_bias_users]
        self.eval_users = eligible_users[num_bias_users : num_bias_users + num_eval_users]

        self.reranker = PropelReranker(
            model=model,
            dataset=data_name,
            num_shuffles=num_shuffles_bias,
            backend=backend,
        )

    def calculate_propensity_scores(self, N: int, experiment_results: Optional[Dict] = None) -> Dict[int, float]:
        """Compute propensity weights for N candidates."""
        return self.reranker.propensity_model.get_inverse_propensity_weights(N)

    def debias_ranking(
        self,
        rankings: List[List[str]],
        original_candidate_orders: List[List[str]],
    ) -> List[str]:
        """Debias rankings via PROPEL consensus aggregation."""
        weights = self.calculate_propensity_scores(len(rankings[0]))
        consensus, _ = self.reranker.aggregator.aggregate(
            rankings=rankings,
            original_candidate_orders=original_candidate_orders,
            propensity_weights=weights,
        )
        return consensus


def main():
    print("PROPEL: PROpensity-based-Position-bias-Elimination-for-LLMs")
    print("Use `from propel import PROPEL, PropelReranker` for modern Python integration.")


if __name__ == "__main__":
    main()