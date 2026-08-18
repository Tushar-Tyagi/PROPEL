# PROPEL Project Structure

This document outlines the architecture and organization of the lean PROPEL repository.

## 📁 Repository Overview

```
PROPEL/
├── propel/                      # Core PROPEL library
│   ├── __init__.py              # Package exports & version
│   ├── reranker.py              # PropelReranker & pipeline orchestration
│   ├── propensity.py            # Parametric Softmax propensity model & weights (Eq. 2)
│   ├── aggregation.py           # Two-phase consensus aggregation & consistency clipping (Eq. 3 & 4)
│   ├── profiler.py              # Offline bias probing & Laplace smoothing (Sec 4.1, App B)
│   ├── llm.py                   # Parallel LLM client & prompt templates (Table 1)
│   ├── explainability.py        # Explainability report generator & item adjustment table (Sec 4.3)
│   ├── metrics.py               # NDCG@K, Hit@K, MRR, and Kendall-Tau metrics
│   ├── data.py                  # UserProfile dataclass & cohort loaders
│   └── defaults.py              # Pre-calibrated empirical bias coefficients (Table 2 & 3)
│
├── examples/                    # Runnable examples
│   ├── quickstart.py            # End-to-end reranking & audit export demo
│   ├── offline_probing.py       # Offline bias calibration demo
│   └── evaluate_dataset.py      # Benchmark evaluation demo
│
├── tests/                       # Complete unit and integration test suite
│   ├── test_propensity.py       # Propensity modeling tests
│   ├── test_aggregation.py      # Borda init, local search, & clipping tests
│   ├── test_profiler.py         # Probing & coefficient estimation tests
│   ├── test_explainability.py   # Explainability JSON tests
│   ├── test_llm.py              # Prompt construction & JSON parser tests
│   ├── test_metrics.py          # Metric accuracy tests
│   └── test_reranker.py         # End-to-end integration tests
│
├── data/                        # Benchmark dataset cohorts
│   ├── ml-1m/                   # MovieLens-1M evaluation cohorts
│   ├── books/                   # Amazon Books cohorts
│   ├── beauty/                  # Amazon Beauty cohorts
│   ├── music/                   # Amazon Digital Music cohorts
│   ├── steam/                   # Steam Video Games cohorts
│   └── create_datasets.py       # Master dataset generation script
│
├── LLM_debias.py                # Backward compatibility layer
├── pyproject.toml               # Modern build configuration
├── setup.py                     # Python package setup
└── requirements.txt             # Minimal dependencies
```
