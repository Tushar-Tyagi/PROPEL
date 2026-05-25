"""Tests for propensity score computation.

Validates that :meth:`LLMPositionBiasAnalyzer.calculate_propensity_scores`
produces strictly positive scores at every position and that the
inverse propensity weights are directionally correct (higher for
positions that the model under-represents).

All tests run without API calls.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)

from LLM_debias import LLMPositionBiasAnalyzer


class TestPropensityScores:
    """Tests for ``calculate_propensity_scores``.

    Notes
    -----
    The propensity function is defined in Eq. 2 of Tyagi & Madisetti
    (2026), §4.1, and implemented in ``LLM_debias.py`` lines
    1183–1204.  The function maps each position *p* to an inverse
    propensity weight w_p = 1 / exp(S_total(p)).
    """

    @pytest.fixture
    def analyzer(self) -> LLMPositionBiasAnalyzer:
        """Create a minimal analyzer instance.

        Returns
        -------
        LLMPositionBiasAnalyzer
        """
        data = pd.DataFrame(
            {
                "UserID": [1] * 10,
                "Title": [f"item_{i}" for i in range(10)],
                "Rating": [5] * 10,
                "Timestamp": list(range(10)),
                "Genres": ["Drama"] * 10,
                "Gender": ["M"] * 10,
                "Age": [25] * 10,
                "Occupation": [0] * 10,
            }
        )
        analyzer = object.__new__(LLMPositionBiasAnalyzer)
        analyzer.data = data
        analyzer.data_name = "movie_lens"
        analyzer.list_size = 20
        analyzer.model = "gpt-4o"
        analyzer.backend = "openai"
        return analyzer

    def test_propensity_strictly_positive(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """All propensity scores (inverse weights) must be > 0.

        Notes
        -----
        Since w_p = 1 / exp(S_total) and exp is always positive,
        w_p must be strictly positive regardless of the bias
        coefficient values.  This is a mathematical invariant of
        the exponential propensity model (Eq. 2).
        """
        list_size = 20
        # Strong primacy + moderate recency + negative middle
        experiment_results = {
            "avg_primacy": 6.0,
            "avg_recency": 3.0,
            "avg_middle": 0.5,
        }
        scores = analyzer.calculate_propensity_scores(
            list_size, experiment_results
        )

        assert len(scores) == list_size
        for pos in range(1, list_size + 1):
            assert scores[pos] > 0.0, (
                f"Propensity score at position {pos} must be "
                f"strictly positive, got {scores[pos]}"
            )

    def test_inverse_propensity_higher_for_underrepresented(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """Inverse weights should be higher for under-represented
        positions (middle/end) than for over-represented positions
        (beginning) when there is strong primacy bias.

        Notes
        -----
        With B_prim > 0, the model over-favours early positions.
        The propensity function assigns high S_total to early
        positions → low w_p (small weight).  Middle positions
        receive lower S_total → higher w_p (larger correction).
        """
        list_size = 20
        # Strong primacy bias, near-zero recency, slight
        # middle-under-representation
        experiment_results = {
            "avg_primacy": 6.0,
            "avg_recency": 0.5,
            "avg_middle": 0.5,
        }
        scores = analyzer.calculate_propensity_scores(
            list_size, experiment_results
        )

        # Early position (over-represented → low weight)
        early_weight = scores[1]
        # Middle position (under-represented → high weight)
        mid_pos = list_size // 2
        mid_weight = scores[mid_pos]

        assert mid_weight > early_weight, (
            f"Middle weight ({mid_weight:.4f}) should exceed "
            f"early weight ({early_weight:.4f}) under primacy bias"
        )

    def test_no_bias_uniform_weights(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """When all coefficients are zero, all weights should be 1.0.

        Notes
        -----
        B_prim = B_rec = B_mid = 0 → S_total = 0 → w_p = 1/exp(0)
        = 1.0 at every position.
        """
        list_size = 20
        expected_p = 0.025 * list_size  # 0.5
        expected_m = 0.05 * list_size   # 1.0
        experiment_results = {
            "avg_primacy": expected_p,
            "avg_recency": expected_p,
            "avg_middle": expected_m,
        }
        scores = analyzer.calculate_propensity_scores(
            list_size, experiment_results
        )

        for pos in range(1, list_size + 1):
            assert scores[pos] == pytest.approx(1.0, abs=1e-9), (
                f"Weight at position {pos} should be 1.0 under "
                f"no bias, got {scores[pos]}"
            )

    def test_recency_bias_end_overrepresented(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """Under strong recency bias, the last position should have
        a lower inverse weight (it's over-represented) than the
        first position.

        Notes
        -----
        B_rec > 0 means the model over-favours late positions.
        w_p for the last position should be small (high propensity
        = low correction weight).
        """
        list_size = 20
        experiment_results = {
            "avg_primacy": 0.5,   # neutral
            "avg_recency": 6.0,   # strong recency
            "avg_middle": 1.0,    # neutral
        }
        scores = analyzer.calculate_propensity_scores(
            list_size, experiment_results
        )

        first_weight = scores[1]
        last_weight = scores[list_size]

        assert first_weight > last_weight, (
            f"First weight ({first_weight:.4f}) should exceed "
            f"last weight ({last_weight:.4f}) under recency bias"
        )
