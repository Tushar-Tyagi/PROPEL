"""Tests for NDCG metric computation.

Validates the NDCG@k implementation in
:class:`LLM_debias.LLMPositionBiasAnalyzer` against hand-computed
reference values.  The tests import and exercise the existing
implementation rather than duplicating it.

All tests run without API calls.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2])
)

from LLM_debias import LLMPositionBiasAnalyzer


class TestNDCGAtK:
    """Tests for ``LLMPositionBiasAnalyzer._calculate_ndcg``.

    Notes
    -----
    ``_calculate_ndcg`` is a method on the analyzer class.  Because
    it does not depend on instance state (only on its arguments), we
    create a lightweight mock instance to call it.
    """

    @pytest.fixture
    def analyzer(self) -> LLMPositionBiasAnalyzer:
        """Create a minimal analyzer instance for calling NDCG.

        Returns
        -------
        LLMPositionBiasAnalyzer
            Instance with just enough state to avoid crashes in
            ``_calculate_ndcg``.
        """
        # Build a tiny synthetic DataFrame
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
        return analyzer

    def test_perfect_ranking(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """NDCG@k = 1.0 when the relevant item is ranked first.

        Notes
        -----
        DCG = 1 / log2(1 + 1) = 1.0; IDCG = 1.0.
        """
        target = "target_movie"
        ranked = [target, "a", "b", "c", "d"]
        for k in (1, 5, 10, 20):
            ndcg = analyzer._calculate_ndcg(target, ranked, k)
            assert ndcg == pytest.approx(1.0), (
                f"NDCG@{k} should be 1.0 for perfect ranking"
            )

    def test_relevant_item_last_beyond_k(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """NDCG@k = 0.0 when the relevant item is beyond position k.

        Notes
        -----
        The target does not appear in the first k items.
        """
        target = "target_movie"
        ranked = ["a", "b", "c", "d", target]
        assert analyzer._calculate_ndcg(target, ranked, 3) == 0.0
        assert analyzer._calculate_ndcg(target, ranked, 4) == 0.0

    def test_intermediate_ranking(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """NDCG@10 for the relevant item at position 3.

        Notes
        -----
        Position 3 (1-based) → DCG = 1 / log2(3 + 1) = 1 / 2 = 0.5.
        IDCG = 1.0 → NDCG = 0.5.
        """
        target = "target_movie"
        ranked = ["a", "b", target, "c", "d", "e", "f", "g", "h", "i"]
        ndcg_10 = analyzer._calculate_ndcg(target, ranked, 10)
        expected = 1.0 / math.log2(3 + 1)  # 0.5
        assert ndcg_10 == pytest.approx(expected, abs=1e-6)

    def test_position_4(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """NDCG@10 for the relevant item at position 4.

        Notes
        -----
        Position 4 (1-based) → DCG = 1 / log2(4 + 1) ≈ 0.4307.
        """
        target = "target_movie"
        ranked = ["a", "b", "c", target, "d", "e", "f", "g", "h", "i"]
        ndcg = analyzer._calculate_ndcg(target, ranked, 10)
        expected = 1.0 / math.log2(5)
        assert ndcg == pytest.approx(expected, abs=1e-6)

    def test_relevant_item_at_position_k(
        self, analyzer: LLMPositionBiasAnalyzer
    ) -> None:
        """NDCG@5 when the relevant item is exactly at position 5.

        Notes
        -----
        Position 5 → DCG = 1 / log2(6) ≈ 0.3869.
        """
        target = "target_movie"
        ranked = ["a", "b", "c", "d", target]
        ndcg = analyzer._calculate_ndcg(target, ranked, 5)
        expected = 1.0 / math.log2(6)
        assert ndcg == pytest.approx(expected, abs=1e-6)
