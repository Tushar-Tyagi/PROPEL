"""
PROPEL Explainability Interface Module.

Implements the Explainability Interface from Section 4.3 of the PROPEL paper:
  - Bias Coefficients: Primacy (B_prim), Recency (B_rec), Middle (B_mid)
  - Propensity Curve Data: [position p, propensity S_total(p)] pairs
  - Item Adjustment Table: Detailed per-item audit trails showing input positions,
    inverse propensity weights, raw LLM ranks, and final de-biased ranks
  - JSON payload export for auditability and dashboard visualization
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union


@dataclass
class ItemAuditRecord:
    """Audit record for an individual candidate item."""
    item_id: str
    item_title: str
    prompt_positions: List[int]
    inverse_propensity_weights: List[float]
    raw_llm_ranks: List[int]
    initial_borda_rank: int
    final_debiased_rank: int
    rank_shift: int  # initial_borda_rank - final_debiased_rank (>0 means promoted, <0 means demoted)


@dataclass
class ExplainabilityReport:
    """
    Structured Explainability Report for a PROPEL Recommendation Instance.
    """
    recommendation_id: str
    dataset: str
    model: str
    bias_coefficients: Dict[str, float]
    propensity_curve: List[List[Union[int, float]]]
    aggregation_metadata: Dict[str, Any]
    item_adjustments: List[Dict[str, Any]]
    final_ranking: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard Python dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate a concise human-readable summary of the debiasing adjustments."""
        lines = [
            f"=== PROPEL Explainability Report [ID: {self.recommendation_id}] ===",
            f"Model: {self.model} | Dataset: {self.dataset}",
            f"Bias Coefficients: Primacy={self.bias_coefficients.get('B_prim', 0.0):+.3f}, "
            f"Recency={self.bias_coefficients.get('B_rec', 0.0):+.3f}, "
            f"Middle={self.bias_coefficients.get('B_mid', 0.0):+.3f}",
            f"Consistency Clip: C={self.aggregation_metadata.get('clip_bound', 1.0):.2f} "
            f"(Agreement: {self.aggregation_metadata.get('consistency_rate', 0.5):.1%}) | "
            f"Local Swaps: {self.aggregation_metadata.get('num_swaps', 0)}",
            "",
            "Top-5 Final Recommendations (Rank Shift from Borda baseline):",
        ]
        for idx, item_title in enumerate(self.final_ranking[:5], start=1):
            adj = next((a for a in self.item_adjustments if a["item_title"] == item_title), None)
            shift_str = ""
            if adj:
                shift = adj["rank_shift"]
                if shift > 0:
                    shift_str = f" [▲ +{shift} ranks]"
                elif shift < 0:
                    shift_str = f" [▼ {shift} ranks]"
                else:
                    shift_str = " [= unchanged]"
            lines.append(f"  {idx}. {item_title}{shift_str}")
        return "\n".join(lines)


def generate_explainability_report(
    recommendation_id: str,
    dataset: str,
    model: str,
    bias_coefficients: Dict[str, float],
    propensity_curve: List[Tuple[int, float]],
    rankings: List[List[str]],
    original_candidate_orders: List[List[str]],
    propensity_weights: Dict[int, float],
    initial_borda_ranking: List[str],
    final_consensus_ranking: List[str],
    aggregation_metadata: Dict[str, Any],
) -> ExplainabilityReport:
    """
    Construct an ExplainabilityReport from a PROPEL reranking execution.
    """
    borda_pos_map = {item: i + 1 for i, item in enumerate(initial_borda_ranking)}
    final_pos_map = {item: i + 1 for i, item in enumerate(final_consensus_ranking)}

    item_adjustments = []
    candidates = list(initial_borda_ranking)

    for item in candidates:
        prompt_positions = []
        inv_weights = []
        raw_ranks = []

        for r, orig_order in zip(rankings, original_candidate_orders):
            if item in orig_order:
                p_pos = orig_order.index(item) + 1
                prompt_positions.append(p_pos)
                inv_weights.append(float(propensity_weights.get(p_pos, 1.0)))
            if item in r:
                r_pos = r.index(item) + 1
                raw_ranks.append(r_pos)

        b_rank = borda_pos_map.get(item, len(candidates))
        f_rank = final_pos_map.get(item, len(candidates))
        rank_shift = b_rank - f_rank  # Positive if promoted

        record = ItemAuditRecord(
            item_id=str(item),
            item_title=str(item),
            prompt_positions=prompt_positions,
            inverse_propensity_weights=inv_weights,
            raw_llm_ranks=raw_ranks,
            initial_borda_rank=b_rank,
            final_debiased_rank=f_rank,
            rank_shift=rank_shift,
        )
        item_adjustments.append(asdict(record))

    return ExplainabilityReport(
        recommendation_id=recommendation_id,
        dataset=dataset,
        model=model,
        bias_coefficients=bias_coefficients,
        propensity_curve=[[int(p), float(s)] for p, s in propensity_curve],
        aggregation_metadata=aggregation_metadata,
        item_adjustments=item_adjustments,
        final_ranking=final_consensus_ranking,
    )
