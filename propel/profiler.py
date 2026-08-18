"""
PROPEL Probing and Bias Profiling Module.

Implements the offline Probing Stage from Section 4.1 & Appendix B of the PROPEL paper:
  - Empirical top-K appearance tracking across randomized candidate list shuffles
  - Laplace smoothing for position-wise selection frequencies: f(p) = (C(p) + 0.5) / (M + 1.0)
  - Regional segmentation: Primacy (first 25%), Middle (middle 50%), Recency (last 25%)
  - Normalized proportional deviations against hypergeometric random expectation:
      B_prim = (P_bar_prim - E_prim) / E_prim
      B_rec  = (P_bar_rec - E_rec) / E_rec
      B_mid  = (P_bar_mid - E_mid) / E_mid
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from propel.propensity import PropensityModel


def compute_segment_boundaries(N: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Compute 1-indexed (start, end) inclusive ranges for Primacy, Middle, and Recency segments.
    
    Default partitioning from Section 4.1:
      - Primacy: First 25% [1, floor(0.25 * N)] (min 1 position)
      - Recency: Last 25% [N - floor(0.25 * N) + 1, N] (min 1 position)
      - Middle: Middle 50% [primacy_end + 1, recency_start - 1]
    """
    if N < 4:
        # Edge case for very small lists
        return (1, 1), (2, max(1, N - 1)), (N, N)
    
    prim_count = max(1, int(round(0.25 * N)))
    rec_count = max(1, int(round(0.25 * N)))
    
    prim_range = (1, prim_count)
    rec_range = (N - rec_count + 1, N)
    mid_range = (prim_count + 1, N - rec_count)
    
    return prim_range, mid_range, rec_range


class BiasProfiler:
    """
    Profiles position bias from empirical candidate list shuffling and ranking observations.
    
    Attributes
    ----------
    N : int
        Candidate list size (e.g., 20 or 100).
    top_k : int
        Number of top ranked items considered per trial (e.g. 10).
    """

    def __init__(self, N: int = 20, top_k: int = 10):
        self.N = N
        self.top_k = min(top_k, N)
        self.position_counts: Dict[int, int] = {p: 0 for p in range(1, N + 1)}
        self.total_trials: int = 0
        self.per_user_frequencies: List[Dict[int, float]] = []

    def record_trial(self, prompt_candidate_order: List[Any], llm_ranked_items: List[Any]):
        """
        Record a single trial/shuffle result.
        
        Parameters
        ----------
        prompt_candidate_order : List[Any]
            The 1-indexed order of candidates presented to the LLM.
        llm_ranked_items : List[Any]
            The items returned by the LLM in ranked order.
        """
        selected_top_k = set(llm_ranked_items[: self.top_k])
        for prompt_idx, item in enumerate(prompt_candidate_order, start=1):
            if prompt_idx <= self.N and item in selected_top_k:
                self.position_counts[prompt_idx] += 1
        self.total_trials += 1

    def compute_empirical_frequencies(self, laplace: bool = True) -> Dict[int, float]:
        """
        Compute empirical selection frequency f(p) for each position.
        
        Laplace smoothed formula (Appendix B):
            f(p) = (C(p) + 0.5) / (M + 1.0)
        """
        M = max(1, self.total_trials)
        frequencies = {}
        for p in range(1, self.N + 1):
            count = self.position_counts.get(p, 0)
            if laplace:
                f_p = (count + 0.5) / (M + 1.0)
            else:
                f_p = count / float(M)
            frequencies[p] = f_p
        return frequencies

    def estimate_bias_coefficients(
        self,
        empirical_frequencies: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, float, float]:
        """
        Estimate normalized bias coefficients (B_prim, B_rec, B_mid).
        
        Calculates observed regional selection proportions against theoretical
        hypergeometric random expectation:
            E_prim = len(prim_segment) / N
            E_rec  = len(rec_segment) / N
            E_mid  = len(mid_segment) / N
            
            B_prim = (P_bar_prim - E_prim) / E_prim
            B_rec  = (P_bar_rec - E_rec) / E_rec
            B_mid  = (P_bar_mid - E_mid) / E_mid
        """
        if empirical_frequencies is None:
            empirical_frequencies = self.compute_empirical_frequencies(laplace=True)

        prim_range, mid_range, rec_range = compute_segment_boundaries(self.N)

        prim_positions = list(range(prim_range[0], prim_range[1] + 1))
        mid_positions = list(range(mid_range[0], mid_range[1] + 1))
        rec_positions = list(range(rec_range[0], rec_range[1] + 1))

        sum_total = sum(empirical_frequencies[p] for p in range(1, self.N + 1))
        if sum_total <= 0:
            sum_total = 1.0

        # Normalized observed regional proportions
        p_prim = sum(empirical_frequencies[p] for p in prim_positions) / sum_total
        p_mid = sum(empirical_frequencies[p] for p in mid_positions) / sum_total
        p_rec = sum(empirical_frequencies[p] for p in rec_positions) / sum_total

        # Theoretical expected proportions under uniform placement
        e_prim = len(prim_positions) / float(self.N)
        e_mid = len(mid_positions) / float(self.N)
        e_rec = len(rec_positions) / float(self.N)

        # Normalized bias coefficients
        b_prim = (p_prim - e_prim) / e_prim if e_prim > 0 else 0.0
        b_rec = (p_rec - e_rec) / e_rec if e_rec > 0 else 0.0
        b_mid = (p_mid - e_mid) / e_mid if e_mid > 0 else 0.0

        return float(b_prim), float(b_rec), float(b_mid)

    def create_propensity_model(self) -> PropensityModel:
        """Estimate bias coefficients and return a configured PropensityModel."""
        b_prim, b_rec, b_mid = self.estimate_bias_coefficients()
        return PropensityModel(B_prim=b_prim, B_rec=b_rec, B_mid=b_mid)
