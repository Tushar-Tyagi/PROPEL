#!/usr/bin/env python3
"""Main entry point for the PROPEL probing sensitivity analysis.

This script orchestrates the N_b sweep described in §4.1 of
Tyagi & Madisetti (2026).  It supports three execution modes:

1. **Full run** (default) — probing + recommendation + figure + table.
2. ``--coefficients-only`` — probing sweep only, no API cost for the
   recommendation stage.  Produces the left panel of Fig. 7 and the
   bias-coefficient columns of the CSV table.
3. ``--from-cache`` — regenerate figure and table from a previously
   saved ``probing_sensitivity_raw.json`` with no API calls at all.

Usage
-----
::

    # Full pipeline (requires OPENROUTER_API_KEY)
    export OPENROUTER_API_KEY=sk-or-…
    python sensitivity_analysis/run_sensitivity.py

    # Coefficients only (still requires API for probing)
    python sensitivity_analysis/run_sensitivity.py --coefficients-only

    # From cached JSON (no API calls)
    python sensitivity_analysis/run_sensitivity.py --from-cache
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import LLM_debias.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sensitivity_analysis.plotting import generate_figure  # noqa: E402
from sensitivity_analysis.probing_sweep import (  # noqa: E402
    run_ndcg_sweep,
    run_probing_sweep,
)

logger = logging.getLogger("sensitivity_analysis")


# ===================================================================
# CLI
# ===================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=(
            "Probing sensitivity analysis for PROPEL — sweep over "
            "N_b (number of probing users) and measure the effect "
            "on bias coefficients and downstream NDCG."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_THIS_DIR / "config.yaml"),
        help="Path to config.yaml (default: %(default)s).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--coefficients-only",
        action="store_true",
        help=(
            "Run only Step 1 (probing sweep).  Skips the "
            "recommendation stage and produces the left panel of "
            "Fig. 7 without any additional API cost."
        ),
    )
    mode.add_argument(
        "--from-cache",
        action="store_true",
        help=(
            "Regenerate figure and table from a previously saved "
            "probing_sensitivity_raw.json.  No API calls."
        ),
    )
    return parser


# ===================================================================
# Logging setup
# ===================================================================

def _configure_logging(log_path: Path) -> None:
    """Set up logging to stdout (INFO) and a file (DEBUG).

    Parameters
    ----------
    log_path : Path
        Absolute path to the log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — DEBUG level
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s"
        )
    )
    root.addHandler(fh)

    # Stdout handler — INFO level
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(
        logging.Formatter("%(levelname)-5s  %(message)s")
    )
    root.addHandler(sh)


# ===================================================================
# OpenRouter environment setup
# ===================================================================

def _setup_openrouter_env() -> None:
    """Configure the OpenAI client to route through OpenRouter.

    Reads ``OPENROUTER_API_KEY`` from the environment and sets
    ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` so that the existing
    ``openai.OpenAI`` client in ``LLM_debias.py`` transparently
    sends requests to the OpenRouter endpoint.

    Raises
    ------
    RuntimeError
        If ``OPENROUTER_API_KEY`` is not set.

    Notes
    -----
    OpenRouter (https://openrouter.ai) provides a unified API that
    is wire-compatible with the OpenAI Chat Completions format.  By
    setting ``OPENAI_BASE_URL`` to
    ``https://openrouter.ai/api/v1``, the standard ``openai`` Python
    library will direct all requests there without code changes.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set.  "
            "Export it before running the sensitivity analysis:\n"
            "  export OPENROUTER_API_KEY=sk-or-…"
        )
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = (
        "https://openrouter.ai/api/v1"
    )
    logger.info(
        "OpenRouter environment configured "
        "(OPENAI_BASE_URL=%s).",
        os.environ["OPENAI_BASE_URL"],
    )


# ===================================================================
# Output helpers
# ===================================================================

def _save_raw_json(
    raw_data: Dict[str, Any], path: Path
) -> None:
    """Persist the complete raw results to JSON.

    Parameters
    ----------
    raw_data : dict
        Full results dictionary (coefficients + optional NDCG).
    path : Path
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert int keys to strings for JSON serialisation
    serialisable = _stringify_keys(raw_data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Raw results saved to %s", path)


def _save_table_csv(
    raw_data: Dict[str, Any], path: Path
) -> None:
    """Write the summary table as CSV.

    Parameters
    ----------
    raw_data : dict
        Full results dictionary.
    path : Path
        Destination CSV path.

    Notes
    -----
    Columns: ``N_b, B_prim_mean, B_prim_ci95, B_rec_mean,
    B_rec_ci95, B_mid_mean, B_mid_ci95, NDCG1_mean, NDCG1_ci95,
    NDCG20_mean, NDCG20_ci95``.  CI values are half-widths (±).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    coeff = raw_data["coefficients"]
    ndcg = raw_data.get("ndcg", {})

    fieldnames = [
        "N_b",
        "B_prim_mean",
        "B_prim_ci95",
        "B_rec_mean",
        "B_rec_ci95",
        "B_mid_mean",
        "B_mid_ci95",
        "NDCG1_mean",
        "NDCG1_ci95",
        "NDCG20_mean",
        "NDCG20_ci95",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_b in sorted(int(k) for k in coeff.keys()):
            c = coeff[str(n_b)]
            n = ndcg.get(str(n_b), {})
            row = {
                "N_b": n_b,
                "B_prim_mean": f"{c['B_prim_mean']:.6f}",
                "B_prim_ci95": f"{c['B_prim_ci95']:.6f}",
                "B_rec_mean": f"{c['B_rec_mean']:.6f}",
                "B_rec_ci95": f"{c['B_rec_ci95']:.6f}",
                "B_mid_mean": f"{c['B_mid_mean']:.6f}",
                "B_mid_ci95": f"{c['B_mid_ci95']:.6f}",
                "NDCG1_mean": (
                    f"{n['NDCG_1_mean']:.6f}" if n else ""
                ),
                "NDCG1_ci95": (
                    f"{n['NDCG_1_ci95']:.6f}" if n else ""
                ),
                "NDCG20_mean": (
                    f"{n['NDCG_20_mean']:.6f}" if n else ""
                ),
                "NDCG20_ci95": (
                    f"{n['NDCG_20_ci95']:.6f}" if n else ""
                ),
            }
            writer.writerow(row)

    logger.info("Summary table saved to %s", path)


def _stringify_keys(obj: Any) -> Any:
    """Recursively convert dict keys to strings for JSON.

    Parameters
    ----------
    obj : Any
        Arbitrary nested structure.

    Returns
    -------
    Any
        Same structure with all dict keys cast to ``str``.
    """
    if isinstance(obj, dict):
        return {
            str(k): _stringify_keys(v) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_stringify_keys(v) for v in obj]
    return obj


def _resolve(rel: str) -> Path:
    """Resolve a config-relative path to an absolute path.

    Parameters
    ----------
    rel : str
        Path string from ``config.yaml``.

    Returns
    -------
    Path
        Resolved absolute path.
    """
    return (_THIS_DIR / rel).resolve()


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    """Entry point for ``python run_sensitivity.py``."""
    args = _build_parser().parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Configure logging
    log_path = _resolve(config["output"]["log_file"])
    _configure_logging(log_path)

    logger.info("Probing sensitivity analysis — PROPEL")
    logger.info("Config: %s", args.config)

    # ----- Mode: --from-cache -----------------------------------
    if args.from_cache:
        cache_path = _resolve(config["output"]["cache_file"])
        if not cache_path.exists():
            logger.error(
                "Cache file not found: %s.  Run the full "
                "pipeline first.",
                cache_path,
            )
            sys.exit(1)

        logger.info("Loading cached results from %s", cache_path)
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        has_ndcg = "ndcg" in raw_data and raw_data["ndcg"]
        generate_figure(
            raw_data,
            config,
            coefficients_only=not has_ndcg,
        )
        _save_table_csv(
            raw_data,
            _resolve(config["output"]["table_csv"]),
        )
        logger.info("Done (from cache).")
        return

    # ----- Set up OpenRouter for modes that need API calls -------
    _setup_openrouter_env()

    # ----- Set global RNG seeds ----------------------------------
    random.seed(config["probing"]["seeds"][0])
    np.random.seed(config["probing"]["seeds"][0])

    # ----- Step 1: probing sweep ---------------------------------
    logger.info("Step 1 — Probing sweep")
    t0 = time.monotonic()
    probing_results = run_probing_sweep(config)
    t1 = time.monotonic()
    logger.info(
        "Step 1 completed in %.1f s", t1 - t0
    )

    raw_data: Dict[str, Any] = {
        "coefficients": probing_results["coefficients"],
    }

    # ----- Step 2: NDCG sweep (unless --coefficients-only) -------
    if not args.coefficients_only:
        logger.info("Step 2 — NDCG sweep")
        t2 = time.monotonic()
        ndcg_results = run_ndcg_sweep(
            config, probing_results["coefficients"]
        )
        t3 = time.monotonic()
        logger.info(
            "Step 2 completed in %.1f s", t3 - t2
        )
        raw_data["ndcg"] = ndcg_results["ndcg"]

    # ----- Save raw JSON -----------------------------------------
    raw_data = _stringify_keys(raw_data)
    cache_path = _resolve(config["output"]["cache_file"])
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)
    logger.info("Raw results saved to %s", cache_path)

    # ----- Step 3: Figure ----------------------------------------
    logger.info("Step 3 — Generating figure")
    generate_figure(
        raw_data,
        config,
        coefficients_only=args.coefficients_only,
    )

    # ----- Step 4: Table -----------------------------------------
    logger.info("Step 4 — Generating table")
    _save_table_csv(
        raw_data,
        _resolve(config["output"]["table_csv"]),
    )

    logger.info("All steps completed successfully.")


if __name__ == "__main__":
    main()
