"""
PROPEL: PROpensity-based-Position-bias-Elimination-for-LLMs

A model-agnostic, training-free framework for detecting and eliminating position bias
in Large Language Model (LLM) based recommender systems via empirical bias profiling,
parametric inverse propensity weighting, and bias-aware consensus aggregation.
"""

__version__ = "1.0.0"

from propel.reranker import PropelReranker
from propel.propensity import PropensityModel
from propel.profiler import BiasProfiler
from propel.aggregation import (
    ConsensusAggregator,
    borda_initialization,
    asymmetric_local_search,
    compute_consistency_clipping_bound,
)
from propel.explainability import (
    ExplainabilityReport,
    ItemAuditRecord,
    generate_explainability_report,
)
from propel.llm import (
    LLMClient,
    build_ranking_prompt,
    parse_ranking_response,
    format_item_string,
)
from propel.metrics import (
    compute_ndcg,
    compute_hit,
    compute_mrr,
    kendall_tau_distance,
)
from propel.data import (
    UserProfile,
    load_dataset_cohort,
    create_user_profile_from_records,
)
from propel.defaults import DEFAULT_BIAS_PARAMS, DOMAIN_PROMPT_CONFIG

# Top-level convenient alias
PROPEL = PropelReranker

__all__ = [
    "PROPEL",
    "PropelReranker",
    "PropensityModel",
    "BiasProfiler",
    "ConsensusAggregator",
    "borda_initialization",
    "asymmetric_local_search",
    "compute_consistency_clipping_bound",
    "ExplainabilityReport",
    "ItemAuditRecord",
    "generate_explainability_report",
    "LLMClient",
    "build_ranking_prompt",
    "parse_ranking_response",
    "format_item_string",
    "compute_ndcg",
    "compute_hit",
    "compute_mrr",
    "kendall_tau_distance",
    "UserProfile",
    "load_dataset_cohort",
    "create_user_profile_from_records",
    "DEFAULT_BIAS_PARAMS",
    "DOMAIN_PROMPT_CONFIG",
    "__version__",
]
