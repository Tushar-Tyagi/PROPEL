# Probing sensitivity analysis — PROPEL

This artifact accompanies the paper *PROPEL: Probabilistic Relaxation Of
Position Bias ELimination for Fair LLM-based Recommendations* (Tyagi &
Madisetti, 2026).  It investigates how the number of probing users N_b
affects the stability of the learned bias coefficients (B_prim, B_rec,
B_mid) and, consequently, the downstream recommendation quality measured
by NDCG@1 and NDCG@20.  The results address Reviewer 1 §1 ("How
sensitive are the bias estimates to the probing budget?") and Reviewer 3
§2 ("Would increasing N_b beyond the paper's default of 5 significantly
change the conclusions?").  The experimental protocol is detailed in §4.1
of the paper: for each candidate value of N_b ∈ {5, 10, 20, 50}, R = 10
independent replications are run with different random seeds; the mean
and 95 % confidence interval of each coefficient are reported.

## Requirements

- **Python ≥ 3.8**
- Dependencies (a superset of the main PROPEL requirements):

```bash
pip install numpy pandas scipy matplotlib pyyaml tqdm openai
```

Step 2 (downstream NDCG evaluation) requires an **OpenRouter API key**
to call GPT-4o.  Step 1 (coefficient estimation) also requires API calls
for the probing stage.  The `--from-cache` mode reproduces the figure and
table from pre-computed results with **no API calls**.

## Reproducing Fig. 7 from the paper

The full pipeline (Steps 1 + 2) runs the probing sweep *and* the
downstream recommendation evaluation.  This requires an OpenRouter API
key and incurs API cost.

```bash
export OPENROUTER_API_KEY=sk-or-…
cd /path/to/PROPEL
python sensitivity_analysis/run_sensitivity.py
```

**Expected wall-clock time:** 8–12 hours at Tier 1 rate limits (the
bottleneck is the 150 000 LLM calls in Step 2).

**Estimated API cost:** At current GPT-4o pricing through OpenRouter
(≈ $2.50 / 1 M input tokens, ≈ $10 / 1 M output tokens), the default
configuration costs approximately $50–100.  Step 1 alone costs < $5.

**Pre-cached results:** After a successful run, all raw per-seed values
are stored in `results/sensitivity/probing_sensitivity_raw.json`.  This
file is sufficient to regenerate both the figure and the table without
further API calls.

## Reproducing from cached results (no API calls)

If `probing_sensitivity_raw.json` already exists (e.g. from a previous
run or from the supplementary materials), the figure and table can be
regenerated in under 30 seconds with no network access:

```bash
python sensitivity_analysis/run_sensitivity.py --from-cache
```

## Coefficients only (no recommendation-stage API calls)

To estimate the bias coefficients without running the expensive
recommendation evaluation (Step 2), use:

```bash
export OPENROUTER_API_KEY=sk-or-…
python sensitivity_analysis/run_sensitivity.py --coefficients-only
```

This produces the **left panel** of Fig. 7 and the bias-coefficient
columns of the summary table.  The NDCG columns are left blank.  API
cost: < $5 for the default configuration.

## Configuration

All parameters are stored in `config.yaml`.  The table below maps each
key to its meaning and the value used in the published paper.

| Key | Description | Paper value |
|-----|-------------|-------------|
| `dataset` | Dataset identifier | `movielens-1m` |
| `data_path` | Path to MovieLens-1M data (relative to `sensitivity_analysis/`) | `../data/ml-1m` |
| `probing.n_b_values` | Candidate N_b values to sweep | `[5, 10, 20, 50]` |
| `probing.shuffles_per_user` | Number of random shuffles per probing user | `50` |
| `probing.top_k_percent` | Fraction of items treated as "relevant" in bias counting | `0.10` |
| `probing.primacy_window` | Fraction of list constituting the primacy region | `0.25` |
| `probing.recency_window` | Fraction of list constituting the recency region | `0.25` |
| `probing.list_size` | Candidate list size for probing | `20` |
| `probing.n_repeats` | Number of independent seeds per N_b | `10` |
| `probing.seeds` | Explicit seed list for reproducibility | `[0, 1, …, 9]` |
| `recommendation.n_test_users` | Number of test users for NDCG evaluation | `250` |
| `recommendation.test_seed` | Seed for the fixed test split | `42` |
| `recommendation.candidate_list_size` | Candidate list size for recommendation | `20` |
| `recommendation.n_shuffles` | Number of shuffles per test user (N_S) | `15` |
| `openai.model` | LLM model identifier | `gpt-4o` |
| `openai.max_retries` | Maximum retry attempts on API failure | `5` |
| `openai.backoff_base_seconds` | Base delay for exponential backoff | `2.0` |

## Output files

All output files are written to `results/sensitivity/`.

| File | Contents | Paper reference |
|------|----------|-----------------|
| `fig7_probing_sensitivity.pdf` | Vector figure (two panels) | Fig. 7 |
| `fig7_probing_sensitivity.png` | Raster figure at 300 DPI | Fig. 7 (slides) |
| `probing_sensitivity_table.csv` | Summary table with mean ± 95 % CI for all metrics | Table in §4.1 |
| `probing_sensitivity_raw.json` | Complete per-seed raw values for offline regeneration | Supplementary |
| `run.log` | Detailed execution log (DEBUG level) | — |

## Running tests

```bash
pytest sensitivity_analysis/tests/ -v
```

All tests run without API calls (LLM interactions are mocked via
`unittest.mock.patch`).

| Test file | Coverage |
|-----------|----------|
| `test_probing_sweep.py` | `compute_bias_coefficients()`: known-fixture verification, no-bias baseline, maximum-primacy edge case |
| `test_metrics.py` | `_calculate_ndcg()`: perfect ranking (1.0), target beyond k (0.0), hand-computed intermediate values |
| `test_propensity.py` | `calculate_propensity_scores()`: strict positivity, directional correctness under primacy/recency bias, uniform weights at zero bias |

## Citation

```bibtex
@article{tyagi2026propel,
  title   = {{PROPEL}: Probabilistic Relaxation Of Position Bias
             ELimination for Fair {LLM}-based Recommendations},
  author  = {Tyagi, Tushar and Madisetti, Vijay K.},
  journal = {ACM Transactions on Recommender Systems},
  year    = {2026},
  note    = {To appear}
}
```
