"""Publication-quality figure generation for probing sensitivity analysis.

A two-panel figure showing how the bias coefficients (left) and downstream NDCG
(right) vary as a function of the number of probing users N_b.

All visual constants are collected in PLOT_STYLE for easy customisation.

The figure conforms to ACM double-column formatting (≈ 7 inches total width,
3.33 inches per panel, 9 pt font).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ===================================================================
# Style constants — ACM double-column, 9 pt, Black-compatible
# ===================================================================

PLOT_STYLE: Dict[str, Any] = {
    # --- Layout --------------------------------------------------
    "fig_width_inches": 7.0,
    "panel_width_inches": 3.33,
    "fig_height_inches": 2.8,
    # --- Typography ----------------------------------------------
    "font_size": 9,
    "font_family": "serif",
    "mathtext_fontset": "cm",
    # --- Axes ----------------------------------------------------
    "spine_visible_top": False,
    "spine_visible_right": False,
    "grid_axis": "y",
    "grid_alpha": 0.3,
    "grid_linestyle": "--",
    # --- Colours (curated palette) --------------------------------
    "color_B_prim": "#1f77b4",   # muted blue
    "color_B_rec": "#ff7f0e",    # muted orange
    "color_B_mid": "#2ca02c",    # muted green
    "color_NDCG_1": "#d62728",   # muted red
    "color_NDCG_20": "#9467bd",  # muted purple
    "ci_alpha": 0.20,            # fill transparency for CI bands
    # --- Reference lines -----------------------------------------
    "ref_linestyle": "--",
    "ref_linewidth": 0.8,
    "ref_alpha": 0.6,
    # --- Export --------------------------------------------------
    "dpi": 300,
}


def _apply_rcparams() -> None:
    """Configure matplotlib rcParams from :data:`PLOT_STYLE`."""
    matplotlib.rcParams.update(
        {
            "font.size": PLOT_STYLE["font_size"],
            "font.family": PLOT_STYLE["font_family"],
            "mathtext.fontset": PLOT_STYLE["mathtext_fontset"],
            "axes.spines.top": PLOT_STYLE["spine_visible_top"],
            "axes.spines.right": PLOT_STYLE["spine_visible_right"],
            "axes.grid": True,
            "axes.grid.axis": PLOT_STYLE["grid_axis"],
            "grid.alpha": PLOT_STYLE["grid_alpha"],
            "grid.linestyle": PLOT_STYLE["grid_linestyle"],
        }
    )


# ===================================================================
# Public API
# ===================================================================

def generate_figure(
    raw_data: Dict[str, Any],
    config: Dict[str, Any],
    coefficients_only: bool = False,
) -> None:
    """Generate Fig. 7 (probing sensitivity) and save to disk.

    Parameters
    ----------
    raw_data : dict
        The complete raw-results dictionary as stored in
        ``probing_sensitivity_raw.json``.  Must contain a
        ``"coefficients"`` key, and optionally an ``"ndcg"`` key.
    config : dict
        Parsed ``config.yaml``.
    coefficients_only : bool, optional
        If *True*, generate only the left panel (coefficients) and
        omit the NDCG panel.  Default is *False*.

    Notes
    -----
    Saves both a PDF (vector, for the paper) and a PNG (300 DPI, for
    slides / README) to the paths specified in ``config["output"]``.
    """
    _apply_rcparams()

    coeff_data = raw_data["coefficients"]
    ndcg_data = raw_data.get("ndcg")
    has_ndcg = ndcg_data is not None and not coefficients_only

    n_panels = 2 if has_ndcg else 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            PLOT_STYLE["fig_width_inches"]
            if has_ndcg
            else PLOT_STYLE["panel_width_inches"],
            PLOT_STYLE["fig_height_inches"],
        ),
    )
    if n_panels == 1:
        axes = [axes]

    # ---- Left panel: bias coefficients --------------------------
    ax_coeff = axes[0]
    _plot_coefficients(ax_coeff, coeff_data)

    # ---- Right panel: NDCG (if available) -----------------------
    if has_ndcg:
        ax_ndcg = axes[1]
        _plot_ndcg(ax_ndcg, ndcg_data, coeff_data)

    fig.tight_layout(pad=1.0)

    if "dataset" in config:
        dataset_name = config["dataset"]
        if dataset_name.lower() == "movielens-1m":
            dataset_name = "MovieLens-1M"
            
        fig.suptitle(
            f"Sensitivity of PROPEL to Probing Budget ($N_b$) on {dataset_name}",
            fontsize=PLOT_STYLE["font_size"] + 2,
            y=1.08,
        )

    # ---- Save ---------------------------------------------------
    out = config["output"]
    for key, fmt in [("figure_pdf", "pdf"), ("figure_png", "png")]:
        fpath = _resolve_output_path(out[key], config)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            fpath,
            format=fmt,
            dpi=PLOT_STYLE["dpi"],
            bbox_inches="tight",
        )
        logger.info("Saved %s", fpath)

    plt.close(fig)


# ===================================================================
# Panel renderers
# ===================================================================

def _plot_coefficients(
    ax: matplotlib.axes.Axes,
    coeff_data: Dict[str, Any],
) -> None:
    """Render the left panel (B_prim, B_rec, B_mid vs N_b).

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    coeff_data : dict
        Coefficient data keyed by N_b (int or str).
    """
    n_b_values = sorted(int(k) for k in coeff_data.keys())
    coeff_names = [
        ("B_prim", r"$B_{\mathrm{prim}}$", "color_B_prim"),
        ("B_rec", r"$B_{\mathrm{rec}}$", "color_B_rec"),
        ("B_mid", r"$B_{\mathrm{mid}}$", "color_B_mid"),
    ]

    for key, label, color_key in coeff_names:
        means = [
            coeff_data[str(n)][f"{key}_mean"]
            for n in n_b_values
        ]
        cis = [
            coeff_data[str(n)][f"{key}_ci95"]
            for n in n_b_values
        ]
        color = PLOT_STYLE[color_key]

        ax.plot(
            n_b_values,
            means,
            "o-",
            color=color,
            label=label,
            markersize=4,
            linewidth=1.2,
        )
        ax.fill_between(
            n_b_values,
            [m - c for m, c in zip(means, cis)],
            [m + c for m, c in zip(means, cis)],
            color=color,
            alpha=PLOT_STYLE["ci_alpha"],
        )

    ax.set_xlabel(r"$N_b$ (probing users)")
    ax.set_ylabel("Bias coefficient")
    ax.set_xticks(n_b_values)
    ax.legend(fontsize=PLOT_STYLE["font_size"] - 1)
    ax.set_title("(a) Bias coefficients", fontsize=PLOT_STYLE["font_size"])


def _plot_ndcg(
    ax: matplotlib.axes.Axes,
    ndcg_data: Dict[str, Any],
    coeff_data: Dict[str, Any],
) -> None:
    """Render the right panel (NDCG@1, NDCG@20 vs N_b).

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes.
    ndcg_data : dict
        NDCG data keyed by N_b.
    coeff_data : dict
        Coefficient data (used to determine the N_b=5 reference).
    """
    n_b_values = sorted(int(k) for k in ndcg_data.keys())
    ref_n_b = min(n_b_values)  # typically 5

    metrics = [
        ("NDCG_1", "NDCG@1", "color_NDCG_1"),
        ("NDCG_20", "NDCG@20", "color_NDCG_20"),
    ]

    for key, label, color_key in metrics:
        means = [
            ndcg_data[str(n)][f"{key}_mean"]
            for n in n_b_values
        ]
        cis = [
            ndcg_data[str(n)][f"{key}_ci95"]
            for n in n_b_values
        ]
        color = PLOT_STYLE[color_key]

        ax.plot(
            n_b_values,
            means,
            "o-",
            color=color,
            label=label,
            markersize=4,
            linewidth=1.2,
        )
        ax.fill_between(
            n_b_values,
            [m - c for m, c in zip(means, cis)],
            [m + c for m, c in zip(means, cis)],
            color=color,
            alpha=PLOT_STYLE["ci_alpha"],
        )

        # Reference line at N_b = 5 (paper's default)
        ref_val = ndcg_data["5"][f"{key}_mean"]
        ax.axhline(
            ref_val,
            color=color,
            linestyle=PLOT_STYLE["ref_linestyle"],
            linewidth=PLOT_STYLE["ref_linewidth"],
            alpha=PLOT_STYLE["ref_alpha"],
        )

    ax.set_xlabel(r"$N_b$ (probing users)")
    ax.set_ylabel("NDCG")
    ax.set_xticks(n_b_values)
    ax.legend(fontsize=PLOT_STYLE["font_size"] - 1)
    ax.set_title("(b) Downstream NDCG", fontsize=PLOT_STYLE["font_size"])


# ===================================================================
# Helpers
# ===================================================================

def _resolve_output_path(
    rel_path: str, config: Dict[str, Any]
) -> Path:
    """Resolve an output path relative to ``sensitivity_analysis/``.

    Parameters
    ----------
    rel_path : str
        Relative path string from ``config["output"]``.
    config : dict
        Full config (unused currently but kept for future flexibility).

    Returns
    -------
    Path
        Absolute path.
    """
    base = Path(__file__).resolve().parent
    return (base / rel_path).resolve()
