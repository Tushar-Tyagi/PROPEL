"""
PROPEL Consensus Aggregation Module.

Implements the two-phase Bias-Aware Consensus Aggregation procedure from
Section 4.2 (Equations 3 & 4) of the PROPEL paper:
  - Phase 1: Unweighted Borda count initialization
  - Consistency-based clipping bound: C = clip(2 * p_bar / (1 - p_bar), 1, 15)
  - Phase 2: Asymmetric propensity-ratio adjacent-swap local search
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_consistency_clipping_bound(
    rankings: List[List[Any]],
    min_c: float = 1.0,
    max_c: float = 15.0,
    multiplier: float = 2.0,
) -> Tuple[float, float]:
    """
    Compute the consistency-based clipping bound C (Section 4.2, Eq. 4).
    
    Parameters
    ----------
    rankings : List[List[Any]]
        List of rankings from N_S shuffles.
    min_c : float
        Lower clipping threshold (default 1.0).
    max_c : float
        Upper clipping threshold (default 15.0).
    multiplier : float
        Consistency multiplier (default 2.0).
        
    Returns
    -------
    C : float
        The derived clip bound in [min_c, max_c].
    p_bar : float
        The mean pairwise agreement probability across all item pairs.
    """
    if not rankings or len(rankings) <= 1:
        return min_c, 1.0

    items = list(rankings[0])
    num_items = len(items)
    if num_items <= 1:
        return min_c, 1.0

    # Build position lookups for fast comparison
    rank_pos = [{item: idx for idx, item in enumerate(r)} for r in rankings]

    pair_fractions = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            item_a = items[i]
            item_b = items[j]
            a_above_b = 0
            b_above_a = 0
            for pos_dict in rank_pos:
                if item_a in pos_dict and item_b in pos_dict:
                    if pos_dict[item_a] < pos_dict[item_b]:
                        a_above_b += 1
                    else:
                        b_above_a += 1
            total = a_above_b + b_above_a
            if total > 0:
                majority_fraction = max(a_above_b, b_above_a) / float(total)
                pair_fractions.append(majority_fraction)

    if not pair_fractions:
        p_bar = 0.5
    else:
        p_bar = float(np.mean(pair_fractions))

    # Guard bounds for p_bar to prevent division by zero
    p_safe = max(0.50, min(0.999, p_bar))
    raw_c = multiplier * (p_safe / (1.0 - p_safe))
    c_val = max(min_c, min(max_c, raw_c))

    return float(c_val), float(p_bar)


def borda_initialization(
    rankings: List[List[Any]],
    reference_order: Optional[Dict[Any, int]] = None,
) -> List[Any]:
    """
    Phase 1: Standard unweighted Borda count initialization.
    
    Assigns (M - 1 - rank) points per ranking and sorts descending.
    """
    if not rankings:
        return []
    if len(rankings) == 1:
        return list(rankings[0])

    M = len(rankings[0])
    if reference_order is None:
        reference_order = {item: i for i, item in enumerate(rankings[0])}

    borda_scores = defaultdict(float)
    for r in rankings:
        for idx, item in enumerate(r):
            borda_scores[item] += (M - 1 - idx)

    # Sort descending by score, tie-breaking by reference order
    consensus = sorted(
        borda_scores.keys(),
        key=lambda x: (borda_scores[x], -reference_order.get(x, 0)),
        reverse=True,
    )
    return consensus


def asymmetric_local_search(
    initial_consensus: List[Any],
    rankings: List[List[Any]],
    original_candidate_orders: List[List[Any]],
    propensity_weights: Dict[int, float],
    clip_val: Optional[float] = None,
    max_passes: int = 100,
) -> Tuple[List[Any], int]:
    """
    Phase 2: Kemeny-style adjacent-swap local search using asymmetric propensity ratios.
    
    Parameters
    ----------
    initial_consensus : List[Any]
        Starting permutation (typically from Phase 1 Borda init).
    rankings : List[List[Any]]
        The N_S LLM output rankings.
    original_candidate_orders : List[List[Any]]
        The presentation orders of candidates in the prompts (1-indexed mapping).
    propensity_weights : Dict[int, float]
        Mapping from prompt position p (1..M) to inverse propensity weight w(p).
    clip_val : float, optional
        Clipping threshold C for the propensity weight ratio w(p_A) / w(p_B).
    max_passes : int
        Maximum full passes over the list.
        
    Returns
    -------
    consensus : List[Any]
        The final de-biased consensus ranking.
    num_swaps : int
        Total number of adjacent swaps performed.
    """
    consensus = list(initial_consensus)
    M = len(consensus)
    if M <= 1 or not rankings:
        return consensus, 0

    ranking_positions = [{item: i for i, item in enumerate(r)} for r in rankings]
    prompt_positions = [
        {item: i + 1 for i, item in enumerate(orig)}
        for orig in original_candidate_orders
    ]

    total_swaps = 0
    passes = 0
    changed = True

    while changed and passes < max_passes:
        changed = False
        passes += 1

        for i in range(M - 1):
            item_a = consensus[i]
            item_b = consensus[i + 1]

            pref_a = 0.0
            pref_b = 0.0

            for rank_pos, prompt_pos in zip(ranking_positions, prompt_positions):
                if item_a not in rank_pos or item_b not in rank_pos:
                    continue
                if item_a not in prompt_pos or item_b not in prompt_pos:
                    continue

                pos_a = prompt_pos[item_a]
                pos_b = prompt_pos[item_b]

                w_a = propensity_weights.get(pos_a, 1.0)
                w_b = propensity_weights.get(pos_b, 1.0)

                # LLM ranked item_a above item_b
                if rank_pos[item_a] < rank_pos[item_b]:
                    denom = w_b if w_b > 1e-12 else 1e-12
                    ratio = w_a / denom
                    if clip_val is not None and clip_val >= 1.0:
                        ratio = min(clip_val, max(1.0 / clip_val, ratio))
                    pref_a += ratio
                else:
                    # LLM ranked item_b above item_a
                    denom = w_a if w_a > 1e-12 else 1e-12
                    ratio = w_b / denom
                    if clip_val is not None and clip_val >= 1.0:
                        ratio = min(clip_val, max(1.0 / clip_val, ratio))
                    pref_b += ratio

            # Swap if item_b has stronger debiased preference than item_a
            if pref_b > pref_a:
                consensus[i], consensus[i + 1] = consensus[i + 1], consensus[i]
                changed = True
                total_swaps += 1

    return consensus, total_swaps


class ConsensusAggregator:
    """
    Full Two-Phase PROPEL Bias-Aware Consensus Aggregator.
    """

    def __init__(
        self,
        propensity_weights: Optional[Dict[int, float]] = None,
        clip_val: Optional[float] = None,
        auto_clip: bool = True,
        min_clip: float = 1.0,
        max_clip: float = 15.0,
        multiplier: float = 2.0,
    ):
        self.propensity_weights = propensity_weights or {}
        self.clip_val = clip_val
        self.auto_clip = auto_clip
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.multiplier = multiplier

    def aggregate(
        self,
        rankings: List[List[Any]],
        original_candidate_orders: List[List[Any]],
        propensity_weights: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Run complete two-phase PROPEL consensus aggregation.
        
        Returns
        -------
        final_consensus : List[Any]
            The de-biased consensus ranking.
        metadata : Dict[str, Any]
            Aggregation metadata (clip value, consistency rate, number of swaps, etc.).
        """
        if not rankings:
            return [], {}
        if len(rankings) == 1:
            return list(rankings[0]), {"num_swaps": 0, "p_bar": 1.0, "c_val": 1.0}

        weights = propensity_weights if propensity_weights is not None else self.propensity_weights

        # Determine clipping threshold
        if self.auto_clip or self.clip_val is None:
            c_val, p_bar = compute_consistency_clipping_bound(
                rankings=rankings,
                min_c=self.min_clip,
                max_c=self.max_clip,
                multiplier=self.multiplier,
            )
        else:
            c_val = self.clip_val
            _, p_bar = compute_consistency_clipping_bound(
                rankings=rankings,
                min_c=self.min_clip,
                max_c=self.max_clip,
                multiplier=self.multiplier,
            )

        # Phase 1: Borda count initialization
        initial_consensus = borda_initialization(rankings)

        # Phase 2: Asymmetric Propensity-Ratio local search
        final_consensus, num_swaps = asymmetric_local_search(
            initial_consensus=initial_consensus,
            rankings=rankings,
            original_candidate_orders=original_candidate_orders,
            propensity_weights=weights,
            clip_val=c_val,
        )

        metadata = {
            "initial_borda_ranking": initial_consensus,
            "clip_bound": c_val,
            "consistency_rate": p_bar,
            "num_swaps": num_swaps,
            "num_shuffles": len(rankings),
            "num_candidates": len(initial_consensus),
        }

        return final_consensus, metadata
