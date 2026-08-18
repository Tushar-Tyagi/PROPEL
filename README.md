# PROPEL: PROpensity-based-Position-bias-Elimination-for-LLMs

Official implementation of **PROPEL** (*PROpensity-based-Position-bias-Elimination-for-LLMs*), a principled, model-agnostic, and training-free framework for detecting, profiling, and mitigating position bias in Large Language Model (LLM) recommender systems.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

Large Language Models (LLMs) used for ranking exhibit severe position bias—consistently favoring items presented at the beginning (primacy bias) or end (recency bias) of the prompt, while neglecting items in the middle.

**PROPEL** addresses position bias through a two-stage pipeline:
1. **Offline Bias Probing**: Systematically profiles LLM position bias across randomized candidate list shuffles, measuring regional deviations (Primacy $B_{\text{prim}}$, Recency $B_{\text{rec}}$, Middle-ignoring $B_{\text{mid}}$) against theoretical hypergeometric expectations, and fitting a closed-form parametric Softmax propensity model:
   $$P(p) = \frac{\exp\left(B_{\text{prim}}(1-x) + B_{\text{rec}}x + B_{\text{mid}}[1 - 4(x-0.5)^2]\right)}{\sum_{j=1}^N \exp\left(B_{\text{prim}}(1-x_j) + B_{\text{rec}}x_j + B_{\text{mid}}[1 - 4(x_j-0.5)^2]\right)}$$
   yielding inverse propensity weights $w(p) = \frac{1}{N \cdot P(p)}$.
2. **Bias-Aware Consensus Aggregation**: Evaluates $N_S$ parallel shuffled rankings using a two-phase procedure:
   - **Phase 1 (Borda Initialization)**: Unweighted Borda count provides a stable, low-variance starting consensus.
   - **Phase 2 (Asymmetric Local Search with Consistency Clipping)**: Kemeny-style adjacent-swap local search where pairwise votes are weighted by propensity ratios $\frac{w(p_A)}{w(p_B)}$ and clipped to $[1/C, C]$ based on observed pairwise consistency:
     $$C = \text{clip}\left(2 \cdot \frac{\bar{p}}{1 - \bar{p}},\ 1,\ 15\right)$$
3. **Auditable Explainability**: Emits structured JSON audit reports detailing per-item position history, inverse propensity weights, raw LLM ranks, and final de-biased rank shifts.

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/Tushar-Tyagi/PROPEL.git
cd PROPEL

# Install package in editable mode
pip install -e .
```

### Environment Setup

Set your OpenAI or Anthropic API key:
```bash
export OPENAI_API_KEY="your-api-key"
# or create a .env file:
# OPENAI_API_KEY=your-api-key
```

---

## 🚀 Quickstart

### 1. End-to-End Reranking

```python
from propel import PROPEL

# Initialize PROPEL with pre-calibrated parameters for MovieLens-1M
reranker = PROPEL(model="gpt-4o-mini", dataset="ml-1m")

# User interaction history
user_history = [
    {"title": "The Matrix (1999)", "genres": ["Action", "Sci-Fi"], "rating": 5.0},
    {"title": "Inception (2010)", "genres": ["Action", "Sci-Fi", "Thriller"], "rating": 5.0},
]

# Candidate items to rerank
candidates = [
    {"title": "The Godfather (1972)", "genres": ["Crime", "Drama"]},
    {"title": "Terminator 2: Judgment Day (1991)", "genres": ["Action", "Sci-Fi"]},
    {"title": "Blade Runner (1982)", "genres": ["Sci-Fi", "Thriller"]},
]

# Rerank with N_S = 20 shuffles
final_ranking, report = reranker.rerank(
    user_history=user_history,
    candidates=candidates,
    num_shuffles=20,
)

print("De-biased Ranking:", final_ranking)
print("\n" + report.summary())
```

### 2. Offline Bias Probing (Calibrating a New Model / Prompt)

```python
from propel import BiasProfiler

profiler = BiasProfiler(N=20, top_k=10)

# Record trials (prompt presentation order vs LLM ranked output)
for trial in probing_trials:
    profiler.record_trial(
        prompt_candidate_order=trial["prompt_order"],
        llm_ranked_items=trial["ranked_items"]
    )

# Extract bias coefficients and propensity model
b_prim, b_rec, b_mid = profiler.estimate_bias_coefficients()
print(f"Calibrated: B_prim={b_prim:+.3f}, B_rec={b_rec:+.3f}, B_mid={b_mid:+.3f}")

propensity_model = profiler.create_propensity_model()
weights = propensity_model.get_inverse_propensity_weights(N=20)
```

---

## 📂 Repository Structure

```
PROPEL/
├── propel/                      # Core PROPEL library
│   ├── __init__.py              # Package entry point & exports
│   ├── reranker.py              # PropelReranker & pipeline engine
│   ├── propensity.py            # Parametric Softmax propensity model & weights (Eq. 2)
│   ├── aggregation.py           # Two-phase consensus aggregation & consistency clipping (Eq. 3 & 4)
│   ├── profiler.py              # Offline bias probing & Laplace smoothing (Sec 4.1, App B)
│   ├── llm.py                   # Parallel LLM interface & prompt templates (Table 1)
│   ├── explainability.py        # Structured JSON audit reports & item adjustment tables (Sec 4.3)
│   ├── metrics.py               # Evaluation metrics (NDCG@K, Hit@K, MRR, Kendall-Tau)
│   ├── data.py                  # Benchmark dataset loaders
│   └── defaults.py              # Pre-calibrated empirical bias parameters (Tables 2 & 3)
│
├── examples/                    # Runnable examples
│   ├── quickstart.py            # Basic reranking & explainability demo
│   ├── offline_probing.py       # Offline bias profiling demo
│   └── evaluate_dataset.py      # Benchmark evaluation script
│
├── tests/                       # Complete pytest suite
│   ├── test_propensity.py       # Propensity math & normalization tests
│   ├── test_aggregation.py      # Borda init, local search & clipping tests
│   ├── test_profiler.py         # Probing & coefficient estimation tests
│   ├── test_explainability.py   # Explainability payload & audit tests
│   ├── test_llm.py              # Prompt formatting & JSON parsing tests
│   ├── test_metrics.py          # Ranking metrics tests
│   └── test_reranker.py         # End-to-end reranker integration tests
│
├── data/                        # Dataset cohorts (MovieLens, Books, Beauty, Music, Steam)
│   └── create_datasets.py       # Dataset cohort generation script
│
├── pyproject.toml               # Package build configuration
├── setup.py                     # Setup script
└── requirements.txt             # Dependencies
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📄 Citation

```bibtex
@article{propel2026,
  title={De-biased LLM Re-Ranking Using Propensity-Aware Aggregation},
  author={PROPEL Authors},
  journal={ACM Transactions on Recommender Systems (TORS)},
  year={2026}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
