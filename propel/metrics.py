"""
PROPEL Evaluation Metrics Module.

Implements standard ranking evaluation metrics:
  - NDCG@K (Normalized Discounted Cumulative Gain at position K)
  - Hit@K (Accuracy / Hit Rate at position K)
  - MRR (Mean Reciprocal Rank)
  - Kendall-Tau Distance
"""

import math
from typing import List, Sequence, Any, Union
import numpy as np


def compute_ndcg(target_item: Any, ranked_items: Sequence[Any], k: int = 20) -> float:
    """
    Compute NDCG@K for leave-one-out ranking evaluation.
    
    Parameters
    ----------
    target_item : Any
        The ground-truth positive item.
    ranked_items : Sequence[Any]
        The model's ranked list of items.
    k : int
        Cutoff rank.
        
    Returns
    -------
    float
        1 / log2(rank + 2) if target_item is in top-K (0-indexed rank), else 0.0.
    """
    for idx, item in enumerate(ranked_items[:k]):
        if item == target_item or (isinstance(item, dict) and item.get("title") == target_item):
            return float(1.0 / np.log2(idx + 2))
    return 0.0


def compute_hit(target_item: Any, ranked_items: Sequence[Any], k: int = 1) -> float:
    """
    Compute Hit@K (1.0 if target_item is within the top-K items, else 0.0).
    """
    for item in ranked_items[:k]:
        if item == target_item or (isinstance(item, dict) and item.get("title") == target_item):
            return 1.0
    return 0.0


def compute_mrr(target_item: Any, ranked_items: Sequence[Any]) -> float:
    """
    Compute Reciprocal Rank (1.0 / (1-indexed position)) of the target item.
    """
    for idx, item in enumerate(ranked_items):
        if item == target_item or (isinstance(item, dict) and item.get("title") == target_item):
            return float(1.0 / (idx + 1))
    return 0.0


def kendall_tau_distance(ranking_a: Sequence[Any], ranking_b: Sequence[Any]) -> int:
    """
    Compute Kendall-Tau distance (number of discordant pairs) between two permutations.
    """
    if len(ranking_a) != len(ranking_b):
        raise ValueError("Rankings must have identical lengths to compute Kendall-Tau distance.")

    pos_b = {item: i for i, item in enumerate(ranking_b)}
    items = list(ranking_a)
    n = len(items)
    discordant_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if items[i] in pos_b and items[j] in pos_b:
                if pos_b[items[i]] > pos_b[items[j]]:
                    discordant_pairs += 1

    return discordant_pairs
