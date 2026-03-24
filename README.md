# PROPEL: PROpensity-based-Position-bias-Elimination-for-LLMs

LLM Position Bias Analysis Framework: A comprehensive framework for detecting and correcting position bias in Large Language Model (LLM) based recommender systems. This project provides tools to analyze how position bias affects recommendation quality and implements explainable debiasing strategy PROPEL.

##  Project Overview

Position bias occurs when users prefer items that appear earlier in recommendation lists, regardless of their actual relevance. Large Language Models (LLMs) used as recommender systems inherit this bias, favoring items in specific list positions (primacy, recency, and middle-position effects). 

PROPEL is a **model-agnostic, training-free framework** that combines empirical bias profiling, inverse propensity weighting, and bias-aware aggregation to correct these position-dependent distortions. 

By comparing observed top-k item frequencies against hypergeometric expectations across randomized candidate lists, PROPEL yields normalized bias coefficients fitted to a closed-form exponential propensity function. Multiple randomized LLM rerankings are then aggregated using inverse propensity weights to produce de-biased scores.

### Key Highlights
- **Detects & Quantifies position bias** in LLM-based recommender systems using hypergeometric expectations.
- **Explainable mitigation** via explicit bias coefficients (Primacy, Recency, Middle) for full auditability.
- **State-of-the-Art Performance**: Achieves up to **26.7% relative gains in NDCG@1** and **28% in NDCG@20**, significantly outperforming baselines like STELLA and standard Bootstrapping.
- **Model-Agnostic & Training-Free**: Works seamlessly with black-box LLM APIs (GPT-3.5, GPT-4, GPT-4o, Claude).

##  Features

- **Multi-dataset support**: MovieLens-1M, Amazon Books, Amazon Music, Amazon Beauty, and Steam.
- **Flexible LLM backends**: OpenAI GPT models, Claude, custom models.
- **Comprehensive evaluation**: NDCG@1, NDCG@20, Accuracy, Bias scores, Propensity analysis.
- **Rate limiting**: Built-in API rate limiting with configurable tiers.
- **Batch processing**: Efficient parallel processing for large-scale analysis.
- **Explainable Output**: Generates auditable inverse propensity scores and position-wise bias gradients.

##  Supported Datasets

| Dataset | Description | Format | Special Notes |
|---------|-------------|---------|---------------|
| **MovieLens** | Movie ratings and metadata | CSV with Title, Genres, Rating | Includes user demographics |
| **Books** | Book ratings and reviews | CSV with Title | Amazon product data |
| **Music** | Music ratings and reviews | CSV with Title | CD/Vinyl ratings |
| **News** | News article interactions | TSV with behaviors | MIND dataset format |
| **Beauty** | Beauty product reviews | JSONL with Title | Amazon product data |
| **Steam** | Game reviews and ratings | JSON with Title | Gaming platform data |

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (or other LLM provider)
- Sufficient API credits for your chosen model

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd debiased_ranking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   cp env.example .env
   
   # Edit .env with your API keys
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
   ```

4. **Download datasets** (optional)
   ```bash
   # See data/README.md for dataset download instructions
   # Datasets are not included in the repository due to size
   ```

5. **Verify installation**
   ```bash
   python -c "from LLM_debias import LLMPositionBiasAnalyzer; print('Installation successful!')"
   ```

##  Configuration

### API Configuration

The framework supports multiple API tiers for rate limiting:

```python
# Basic tier (conservative)
api_tier = 'basic'  # 500 RPM, 200K TPM

# Tier 1 (balanced)
api_tier = 'tier_1'  # 3500 RPM, 1M TPM

# Tier 2 (aggressive)
api_tier = 'tier_2'  # 5000 RPM, 2M TPM
```

### Model Configuration

Supported models and their configurations:

```python
# OpenAI Models
model = 'gpt-3.5-turbo'      # Fast, cost-effective
model = 'gpt-4'              # High quality, higher cost
model = 'gpt-4-turbo'        # Balanced performance

# Anthropic Models
model = 'claude-3-sonnet'    # High quality
model = 'claude-3-haiku'     # Fast, cost-effective
```

## 📖 Usage Examples

### Basic Position Bias Analysis

```python
import pandas as pd
import os
from LLM_debias import LLMPositionBiasAnalyzer

# Set up environment variables (required)
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'

# Load your dataset (download first - see data/README.md)
data = pd.read_csv('your_dataset.csv')

# Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=data,
    data_name='movie_lens',  # or 'books', 'music', etc.
    model='gpt-3.5-turbo',
    backend='openai',
    num_bias_users=5,
    num_eval_users=100,
    num_shuffles_bias=50,
    list_size=100,
    api_tier='tier_1'
)

# Run bias analysis
results = analyzer.run_bias_analysis()

# View results
print(f"Bias Score: {results['bias_score']:.4f}")
print(f"Propensity Score: {results['propensity_score']:.4f}")
```

### Debiasing with Propensity Scoring

```python
# Run debiased evaluation
debiased_results = analyzer.run_debiased_evaluation(
    bias_scores=results['bias_scores'],
    propensity_scores=results['propensity_scores']
)

# Compare original vs debiased
print("Original NDCG@10:", results['evaluation']['ndcg_10']['mean'])
print("Debiased NDCG@10:", debiased_results['ndcg_10']['mean'])
```

### Custom Dataset Integration

```python
# For custom datasets, modify get_data_columns function
def get_data_columns(data_name: str):
    if data_name == 'your_dataset':
        item_name = 'ItemTitle'      # Your item column name
        item_metadata = ['Category']  # Your metadata columns
        user_metadata = ['Age']       # Your user columns
        user_rating = ['Score']       # Your rating column
        return item_name, item_metadata, user_metadata, user_rating
```

## 📊 Output and Results

### Bias Analysis Results

```json
{
  "bias_score": 0.234,
  "propensity_score": 0.156,
  "shuffle_results": [...],
  "bias_scores": {...},
  "propensity_scores": {...}
}
```

### Evaluation Metrics

- **NDCG@k**: Normalized Discounted Cumulative Gain at position k
- **Accuracy**: Top-k recommendation accuracy
- **Bias Score**: Position bias magnitude (0 = no bias, 1 = maximum bias)
- **Propensity Score**: Item propensity to appear in top positions
- **Statistical Significance**: Performs paired t-tests and computes 95% Confidence Intervals comparing our method to STELLA and Bootstrapping baselines to validate performance gains.

## 🔧 Advanced Features

### Checkpoint System

Save and resume long-running experiments:

```python
# Save checkpoint
analyzer.save_checkpoint('experiment_checkpoint.json')

# Load and resume
analyzer.load_checkpoint('experiment_checkpoint.json')
results = analyzer.run_bias_analysis()
```

### Custom Prompts

Override default prompts for specific use cases:

```python
custom_prompt = "You are a specialized recommendation system..."
results = analyzer.run_bias_analysis(custom_prompt=custom_prompt)
```

### Batch Processing

Process multiple users in parallel:

```python
analyzer = LLMPositionBiasAnalyzer(
    # ... other params ...
    max_workers=10,  # Parallel processing
    batch_size=20    # Batch size for API calls
)
```

### Ablation Studies (No Re-computation)

You can analyze the effects of custom propensity scores and perform ablation studies without incurring additional LLM API costs by utilizing saved checkpoints:

```python
from utilities.ablation_utils import AblationAnalyzer

analyzer = AblationAnalyzer(data=data, data_name='movie_lens', model='gpt-3.5-turbo', backend='openai')

# Re-apply using uniform weights (ablation) or new bias values
results = analyzer.reapply_debiasing_with_new_bias(
    checkpoint_file='experiment_checkpoint.json',
    new_precalculated_bias={'avg_primacy': 0.0, 'avg_recency': 0.0, 'avg_middle': 0.0},
    aggregation_method='mean'
)
```

### Sensitivity Analysis and Hyperparameter Optimization (HPO)

To test the robustness of the framework against its configuration parameters (e.g. `num_bias_users`, `num_shuffles_bias`), you can perform a sensitivity analysis using the built-in utility:

```bash
python utilities/sensitivity_analysis.py --data_name movie_lens --data_path data/ml-1m/processed_ratings.csv --model gpt-3.5-turbo
```
You can modify the `param_grid` inside `utilities/sensitivity_analysis.py` to evaluate different configurations. Intermediate and final results are automatically exported to `results/sensitivity_analysis`.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 LLM_debias.py
black LLM_debias.py
```
##  Citation

If you use this framework in your research, please cite:

`To be updated`

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


##  Repository Notes

### Data and Checkpoints
- **Datasets**: Download instructions in `data/README.md`
- **Checkpoints**: Generate new ones or use small examples in `checkpoints/`
- **Results**: Outputs will be saved to `results/` directory

---

**Note**: This framework requires API credits for LLM providers. Monitor your usage to avoid unexpected costs.
