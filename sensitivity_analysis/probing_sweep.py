"""Probing sweep over N_b values for PROPEL sensitivity analysis.

This module implements the repeated-probing protocol described in §4.1 of
Tyagi & Madisetti (2026).  For each candidate value of N_b (the number of
probing users), R independent replications are run with distinct random
seeds.  Each replication instantiates the bias-detection stage of PROPEL,
computes the three bias coefficients (B_prim, B_rec, B_mid), and
optionally feeds those coefficients into the recommendation stage to
obtain downstream NDCG scores.

Functions
---------
run_probing_sweep
    Step 1 — repeated probing across N_b values.
run_ndcg_sweep
    Step 2 — downstream NDCG sensitivity.
compute_bias_coefficients
    Thin wrapper that extracts B_prim, B_rec, B_mid from raw averages.
load_movielens_data
    Load and preprocess the MovieLens-1M dataset.
download_movielens_data
    Download MovieLens-1M from GroupLens if not present locally.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so we can import from the repository top level.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from LLM_debias import LLMPositionBiasAnalyzer  # noqa: E402
from utilities.statistical_utils import (  # noqa: E402
    StatisticalSignificanceAnalyzer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named constants (mirroring LLM_debias.py:L1171-1173)
# ---------------------------------------------------------------------------
EXPECTED_PRIMACY_FRACTION = 0.025
"""Fraction of the candidate list constituting the primacy window
under the null hypothesis (no bias).  See §3.2 of the paper."""

EXPECTED_RECENCY_FRACTION = 0.025
"""Fraction of the candidate list constituting the recency window
under the null hypothesis."""

EXPECTED_MIDDLE_FRACTION = 0.05
"""Fraction of the candidate list constituting the middle window
under the null hypothesis."""


# ===================================================================
# Data loading
# ===================================================================

def download_movielens_data(dest_dir: str | Path) -> Path:
    """Download and extract MovieLens-1M if it is not already present.

    Parameters
    ----------
    dest_dir : str or Path
        Directory where the ``ml-1m`` folder should reside.  For example
        ``data/ml-1m`` relative to the project root.

    Returns
    -------
    Path
        Path to the directory containing the extracted ``.dat`` files.

    Notes
    -----
    Downloads from ``https://files.grouplens.org/datasets/movielens/
    ml-1m.zip`` (≈6 MB).  The archive is deleted after extraction.
    """
    dest = Path(dest_dir)
    ratings_path = dest / "ratings.dat"
    if ratings_path.exists():
        logger.info(
            "MovieLens-1M already present at %s — skipping download.",
            dest,
        )
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "ml-1m.zip"
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    logger.info("Downloading MovieLens-1M from %s …", url)
    subprocess.check_call(
        ["curl", "-L", "-o", str(zip_path), url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    logger.info("Extracting %s …", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # The zip contains a nested ``ml-1m/`` directory; move files up.
    nested = dest / "ml-1m"
    if nested.is_dir():
        for child in nested.iterdir():
            child.rename(dest / child.name)
        nested.rmdir()
    zip_path.unlink(missing_ok=True)

    logger.info("MovieLens-1M ready at %s", dest)
    return dest


def load_movielens_data(data_path: str | Path) -> pd.DataFrame:
    """Load and preprocess the MovieLens-1M dataset.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing ``ratings.dat``, ``movies.dat``
        and ``users.dat`` (the standard MovieLens-1M layout).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``UserID``, ``Title``, ``Rating``,
        ``Timestamp``, ``Genres``, ``Gender``, ``Age``, ``Occupation``
        — the format expected by
        :class:`LLM_debias.LLMPositionBiasAnalyzer`.

    Notes
    -----
    If the data files are not present, this function will attempt to
    download them via :func:`download_movielens_data`.
    """
    data_dir = Path(data_path)

    # Auto-download if needed
    if not (data_dir / "ratings.dat").exists():
        logger.info(
            "ratings.dat not found in %s — downloading …", data_dir
        )
        download_movielens_data(data_dir)

    logger.info("Loading MovieLens-1M from %s", data_dir)

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=["MovieID", "Title", "Genres"],
        engine="python",
        encoding="latin-1",
    )

    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        engine="python",
        encoding="latin-1",
    )

    df = ratings.merge(movies, on="MovieID").merge(users, on="UserID")
    logger.info(
        "Loaded %d interactions from %d users and %d movies.",
        len(df),
        df["UserID"].nunique(),
        df["MovieID"].nunique(),
    )
    return df


# ===================================================================
# Bias coefficient computation
# ===================================================================

def compute_bias_coefficients(
    avg_primacy: float,
    avg_recency: float,
    avg_middle: float,
    list_size: int,
) -> Tuple[float, float, float]:
    """Extract normalised bias coefficients from raw probing averages.

    Parameters
    ----------
    avg_primacy : float
        Mean number of primacy-window items appearing in the top-k
        across all shuffles and probing users.
    avg_recency : float
        Analogous count for the recency window.
    avg_middle : float
        Analogous count for the middle window.
    list_size : int
        Length of the candidate list used during probing.

    Returns
    -------
    tuple of (float, float, float)
        ``(B_prim, B_rec, B_mid)`` — the normalised bias coefficients.

    Notes
    -----
    Implements the bias-coefficient normalisation from
    ``LLM_debias.py`` lines 1179–1181, corresponding to Eq. 2 in
    Tyagi & Madisetti (2026), §4.1.  Positive values indicate that
    items in the corresponding region are over-represented in the
    top-k; negative values indicate under-representation relative to
    the uniform null.
    """
    # P_observed is the probability of an item appearing in the region
    top_k = max(1, int(0.10 * list_size))
    p_obs_prim = avg_primacy / top_k
    p_obs_rec = avg_recency / top_k
    p_obs_mid = avg_middle / top_k

    # E_expected is the expected probability based on window sizes
    e_exp_prim = 0.25  # primacy window is 25% of list
    e_exp_rec = 0.25   # recency window is 25% of list
    e_exp_mid = 0.50   # middle window is 50% of list

    b_prim = (p_obs_prim - e_exp_prim) / e_exp_prim
    b_rec = (p_obs_rec - e_exp_rec) / e_exp_rec
    b_mid = (p_obs_mid - e_exp_mid) / e_exp_mid

    # Bound coefficients to [-1, 1] as per §4.1
    b_prim = max(-1.0, min(1.0, b_prim))
    b_rec = max(-1.0, min(1.0, b_rec))
    b_mid = max(-1.0, min(1.0, b_mid))

    return b_prim, b_rec, b_mid


# ===================================================================
# Step 1 — Repeated probing sweep
# ===================================================================

def run_probing_sweep(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the repeated-probing protocol across N_b values.

    Parameters
    ----------
    config : dict
        Parsed contents of ``config.yaml``.

    Returns
    -------
    dict
        Nested dictionary keyed by N_b (int) → list of per-seed
        records, plus aggregated mean / 95 % CI for each coefficient.
        Structure::

            {
              "coefficients": {
                5:  {"seeds": [...], "B_prim_mean": ..., ...},
                10: { ... },
                ...
              }
            }

    Notes
    -----
    Each seed controls both user sampling and shuffle ordering via
    ``random.seed`` / ``numpy.random.seed``.  The probing logic is
    delegated to :class:`LLM_debias.LLMPositionBiasAnalyzer`.
    """
    probing_cfg = config["probing"]
    n_b_values: List[int] = probing_cfg["n_b_values"]
    seeds: List[int] = probing_cfg["seeds"]
    shuffles_per_user: int = probing_cfg["shuffles_per_user"]
    list_size: int = probing_cfg["list_size"]

    data_path = _resolve_path(config["data_path"])
    data = load_movielens_data(data_path)

    model = config["openai"]["model"]

    stat_analyzer = StatisticalSignificanceAnalyzer(
        confidence_level=0.95
    )

    results: Dict[str, Any] = {"coefficients": {}}

    for n_b in n_b_values:
        logger.info("=" * 60)
        logger.info("N_b = %d — starting %d seeds", n_b, len(seeds))
        logger.info("=" * 60)

        seed_records: List[Dict[str, Any]] = []

        for seed in seeds:
            logger.info(
                "  seed=%d  N_b=%d — setting RNG state", seed, n_b
            )
            random.seed(seed)
            np.random.seed(seed)

            # Instantiate the analyzer with this N_b and seed.
            # We need enough eval users to satisfy the constructor
            # but we only use bias_users for probing.
            analyzer = LLMPositionBiasAnalyzer(
                data=data,
                data_name="movie_lens",
                model=model,
                backend="openai",
                num_bias_users=n_b,
                num_eval_users=min(
                    250, len(data["UserID"].unique()) - n_b
                ),
                num_shuffles_bias=shuffles_per_user,
                list_size=list_size,
                api_tier="basic",
            )

            all_primacy: List[float] = []
            all_recency: List[float] = []
            all_middle: List[float] = []

            for user_id in analyzer.bias_users:
                logger.debug(
                    "    probing user %s (N_b=%d, seed=%d)",
                    user_id,
                    n_b,
                    seed,
                )
                try:
                    bias_result = (
                        analyzer.run_bias_detection_experiment(
                            user_id, use_parallel=True
                        )
                    )
                    if "error" not in bias_result:
                        all_primacy.append(
                            bias_result["avg_primacy"]
                        )
                        all_recency.append(
                            bias_result["avg_recency"]
                        )
                        all_middle.append(
                            bias_result["avg_middle"]
                        )
                except Exception:
                    logger.exception(
                        "    probing failed for user %s", user_id
                    )

            if not all_primacy:
                logger.warning(
                    "  seed=%d N_b=%d — no successful probing runs",
                    seed,
                    n_b,
                )
                continue

            avg_p = float(np.mean(all_primacy))
            avg_r = float(np.mean(all_recency))
            avg_m = float(np.mean(all_middle))

            b_prim, b_rec, b_mid = compute_bias_coefficients(
                avg_p, avg_r, avg_m, list_size
            )

            record = {
                "seed": seed,
                "n_b": n_b,
                "avg_primacy": avg_p,
                "avg_recency": avg_r,
                "avg_middle": avg_m,
                "B_prim": b_prim,
                "B_rec": b_rec,
                "B_mid": b_mid,
            }
            seed_records.append(record)
            logger.info(
                "  seed=%d  B_prim=%.4f  B_rec=%.4f  B_mid=%.4f",
                seed,
                b_prim,
                b_rec,
                b_mid,
            )

        # Aggregate across seeds for this N_b
        agg = _aggregate_across_seeds(
            seed_records, stat_analyzer
        )
        agg["seeds"] = seed_records
        results["coefficients"][n_b] = agg

        logger.info(
            "N_b=%d  B_prim=%.4f±%.4f  B_rec=%.4f±%.4f  "
            "B_mid=%.4f±%.4f",
            n_b,
            agg["B_prim_mean"],
            agg["B_prim_ci95"],
            agg["B_rec_mean"],
            agg["B_rec_ci95"],
            agg["B_mid_mean"],
            agg["B_mid_ci95"],
        )

    return results


# ===================================================================
# Step 2 — Downstream NDCG sweep
# ===================================================================

def run_ndcg_sweep(
    config: Dict[str, Any],
    coefficients: Dict[str, Any],
) -> Dict[str, Any]:
    """Run downstream recommendation for each (N_b, seed) pair.

    Parameters
    ----------
    config : dict
        Parsed ``config.yaml``.
    coefficients : dict
        Output of :func:`run_probing_sweep` (the ``"coefficients"``
        sub-dict keyed by N_b).

    Returns
    -------
    dict
        Nested dictionary keyed by N_b → per-seed NDCG records and
        aggregated mean / 95 % CI.

    Notes
    -----
    This step requires OpenRouter API calls and is expensive.  Use the
    ``--coefficients-only`` flag to skip it.
    """
    rec_cfg = config["recommendation"]
    n_test_users: int = rec_cfg["n_test_users"]
    test_seed: int = rec_cfg["test_seed"]
    candidate_list_size: int = rec_cfg["candidate_list_size"]
    n_shuffles: int = rec_cfg["n_shuffles"]
    list_size: int = config["probing"]["list_size"]
    model = config["openai"]["model"]

    data_path = _resolve_path(config["data_path"])
    data = load_movielens_data(data_path)

    stat_analyzer = StatisticalSignificanceAnalyzer(
        confidence_level=0.95
    )

    results: Dict[str, Any] = {"ndcg": {}}

    for n_b_str, coeff_data in coefficients.items():
        n_b = int(n_b_str)
        seed_records = coeff_data["seeds"]

        logger.info(
            "NDCG sweep — N_b=%d, %d seeds", n_b, len(seed_records)
        )

        ndcg_records: List[Dict[str, Any]] = []

        for rec in seed_records:
            seed = rec["seed"]
            b_prim = rec["B_prim"]
            b_rec = rec["B_rec"]
            b_mid = rec["B_mid"]

            logger.info(
                "  NDCG  N_b=%d  seed=%d — running recommendation",
                n_b,
                seed,
            )

            # Fix the test split with test_seed
            random.seed(test_seed)
            np.random.seed(test_seed)

            analyzer = LLMPositionBiasAnalyzer(
                data=data,
                data_name="movie_lens",
                model=model,
                backend="openai",
                num_bias_users=n_b,
                num_eval_users=n_test_users,
                num_shuffles_bias=n_shuffles,
                list_size=candidate_list_size,
                api_tier="basic",
            )

            # Build propensity scores from learned coefficients
            bias_result = {
                "avg_primacy": rec["avg_primacy"],
                "avg_recency": rec["avg_recency"],
                "avg_middle": rec["avg_middle"],
            }
            propensity_scores = (
                analyzer.calculate_propensity_scores(
                    candidate_list_size, bias_result
                )
            )

            # Evaluate each test user
            user_ndcg_1: List[float] = []
            user_ndcg_20: List[float] = []

            for user_id in analyzer.eval_users:
                try:
                    user_result = (
                        analyzer._evaluate_our_method_single_user(
                            user_id=user_id,
                            num_candidates=candidate_list_size,
                            num_trials=n_shuffles,
                            aggregation_method="mean",
                            propensity_scores=propensity_scores,
                            use_parallel=True,
                        )
                    )
                    if user_result is not None:
                        user_ndcg_1.append(user_result["ndcg_1"])
                        user_ndcg_20.append(
                            user_result["ndcg_20"]
                        )
                except Exception:
                    logger.exception(
                        "    NDCG eval failed for user %s",
                        user_id,
                    )

            if user_ndcg_1:
                mean_n1 = float(np.mean(user_ndcg_1))
                mean_n20 = float(np.mean(user_ndcg_20))
            else:
                mean_n1 = 0.0
                mean_n20 = 0.0

            ndcg_rec = {
                "seed": seed,
                "n_b": n_b,
                "NDCG_1": mean_n1,
                "NDCG_20": mean_n20,
                "n_users_evaluated": len(user_ndcg_1),
            }
            ndcg_records.append(ndcg_rec)
            logger.info(
                "  seed=%d  NDCG@1=%.4f  NDCG@20=%.4f "
                "(%d users)",
                seed,
                mean_n1,
                mean_n20,
                len(user_ndcg_1),
            )

        # Aggregate
        ndcg_agg = _aggregate_ndcg(ndcg_records, stat_analyzer)
        ndcg_agg["seeds"] = ndcg_records
        results["ndcg"][n_b] = ndcg_agg

    return results


# ===================================================================
# Internal helpers
# ===================================================================

def _resolve_path(rel_path: str) -> Path:
    """Resolve a path relative to ``sensitivity_analysis/``.

    Parameters
    ----------
    rel_path : str
        Path string from ``config.yaml``, potentially prefixed with
        ``../``.

    Returns
    -------
    Path
        Absolute resolved path.
    """
    return (_THIS_DIR / rel_path).resolve()


def _aggregate_across_seeds(
    seed_records: List[Dict[str, Any]],
    stat_analyzer: StatisticalSignificanceAnalyzer,
) -> Dict[str, Any]:
    """Compute mean and 95 % CI half-width for each coefficient.

    Parameters
    ----------
    seed_records : list of dict
        Per-seed records containing ``B_prim``, ``B_rec``, ``B_mid``.
    stat_analyzer : StatisticalSignificanceAnalyzer
        Reusable CI calculator from ``utilities/statistical_utils.py``.

    Returns
    -------
    dict
        Keys: ``B_prim_mean``, ``B_prim_ci95``, … for each coefficient.
    """
    agg: Dict[str, Any] = {}
    for key in ("B_prim", "B_rec", "B_mid"):
        values = [r[key] for r in seed_records]
        mean_val = float(np.mean(values))
        if len(values) >= 2:
            lo, hi = stat_analyzer.calculate_confidence_interval(
                values
            )
            ci_half = (hi - lo) / 2.0
        else:
            ci_half = 0.0
        agg[f"{key}_mean"] = mean_val
        agg[f"{key}_ci95"] = ci_half
    return agg


def _aggregate_ndcg(
    ndcg_records: List[Dict[str, Any]],
    stat_analyzer: StatisticalSignificanceAnalyzer,
) -> Dict[str, Any]:
    """Compute mean and 95 % CI for NDCG@1 and NDCG@20.

    Parameters
    ----------
    ndcg_records : list of dict
        Per-seed NDCG records.
    stat_analyzer : StatisticalSignificanceAnalyzer
        CI calculator.

    Returns
    -------
    dict
        Keys: ``NDCG_1_mean``, ``NDCG_1_ci95``, ``NDCG_20_mean``,
        ``NDCG_20_ci95``.
    """
    agg: Dict[str, Any] = {}
    for key in ("NDCG_1", "NDCG_20"):
        values = [r[key] for r in ndcg_records]
        mean_val = float(np.mean(values))
        if len(values) >= 2:
            lo, hi = stat_analyzer.calculate_confidence_interval(
                values
            )
            ci_half = (hi - lo) / 2.0
        else:
            ci_half = 0.0
        agg[f"{key}_mean"] = mean_val
        agg[f"{key}_ci95"] = ci_half
    return agg
