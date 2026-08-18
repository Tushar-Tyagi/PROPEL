"""
Basic functionality and legacy compatibility tests for PROPEL.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

from LLM_debias import LLMPositionBiasAnalyzer, get_data_columns, get_api_config
import propel


def test_propel_package_exports():
    """Verify core PROPEL symbols are exported at package level."""
    assert hasattr(propel, "PROPEL")
    assert hasattr(propel, "PropelReranker")
    assert hasattr(propel, "PropensityModel")
    assert hasattr(propel, "BiasProfiler")
    assert hasattr(propel, "ConsensusAggregator")
    assert hasattr(propel, "ExplainabilityReport")


def test_get_data_columns():
    """Test get_data_columns helper."""
    item_name, item_meta, user_meta, user_rating = get_data_columns("movie_lens")
    assert item_name == "Title"
    assert "Genres" in item_meta
    assert "Gender" in user_meta


def test_get_api_config():
    """Test API configuration helper."""
    config = get_api_config("gpt-4o-mini", "tier_1")
    assert config["rpm"] == 3500
    assert config["tpm"] == 1000000


def test_legacy_analyzer_init():
    """Test legacy LLMPositionBiasAnalyzer initialization."""
    data = pd.DataFrame({
        "UserID": ["u1", "u1", "u1", "u1", "u1", "u2", "u2", "u2", "u2", "u2"],
        "Title": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"],
        "Rating": [5, 4, 3, 5, 4, 5, 3, 4, 5, 4],
        "Genres": ["Action"] * 10,
    })

    analyzer = LLMPositionBiasAnalyzer(
        data=data,
        data_name="movielens",
        model="gpt-4o-mini",
        num_bias_users=1,
        num_eval_users=1,
        list_size=20,
    )
    assert analyzer.data_name == "movielens"
    assert analyzer.list_size == 20
    assert len(analyzer.bias_users) >= 1
