# Project Structure Documentation

This document provides an overview of the project structure and organization for the LLM Position Bias Analysis Framework.

## 📁 Directory Structure

```
debiased_ranking/
├── 📄 Core Python Files
│   ├── LLM_debias.py          # Main framework implementation
│   ├── LLM_debias2.py         # Alternative/experimental implementation
│   └── setup.py               # Package installation configuration
│
├── 📊 Data Directory
│   ├── beauty/                 # Beauty product dataset
│   ├── books/                  # Books dataset
│   ├── ml-1m/                 # MovieLens 1M dataset
│   ├── music/                  # Music (CDs & Vinyl) dataset
│   ├── news/                   # News (MIND) dataset
│   └── steam/                  # Steam games dataset
│
├── 📓 Jupyter Notebooks
│   ├── experiment_*.ipynb      # Dataset-specific experiments
│   ├── evaluation_*.ipynb      # Evaluation and analysis
│   ├── checkpoint_*.ipynb      # Checkpoint analysis
│   └── borda.ipynb            # Borda count analysis
│
├── 🛠️ Utility Scripts
│   ├── fix_*.py               # Data processing fixes
│   ├── test_*.py              # Testing scripts
│   ├── debug_*.py             # Debugging utilities
│   └── patch_*.py             # Method patches
│
├── 📈 Outputs & Results
│   ├── outputs/                # Generated output files
│   ├── img/                    # Generated plots and images
│   └── *.json, *.csv          # Analysis results
│
├── 📋 Configuration Files
│   ├── requirements.txt        # Python dependencies
│   ├── setup.py               # Package setup
│   ├── .gitignore             # Git ignore rules
│   └── env.example            # Environment variables template
│
└── 📚 Documentation
    ├── README.md               # Main project documentation
    ├── PROJECT_STRUCTURE.md    # This file
    └── LICENSE                 # Project license
```

## 🔧 Core Components

### 1. Main Framework (`LLM_debias.py`)

The primary implementation containing:

- **`LLMPositionBiasAnalyzer`** class: Main analysis framework
- **API rate limiting**: Configurable tiers for different usage levels
- **Multi-dataset support**: Handles various recommendation datasets
- **Bias detection**: Position bias analysis algorithms
- **Debiasing strategies**: Propensity scoring and correction methods
- **Evaluation metrics**: NDCG, Accuracy, and bias scores
- **Checkpoint system**: Save/resume long-running experiments

### 2. Alternative Implementation (`LLM_debias2.py`)

Experimental version with:
- Different approaches to bias analysis
- Alternative evaluation methods
- Extended functionality for research purposes

### 3. Dataset Support

The framework supports multiple recommendation datasets:

| Dataset | Format | Columns | Special Features |
|---------|--------|---------|------------------|
| **MovieLens** | CSV | UserID, Title, Genres, Rating | User demographics |
| **Books** | CSV | UserID, Title, Rating | Amazon product data |
| **Music** | CSV | UserID, Title, Rating | CD/Vinyl ratings |
| **News** | TSV | UserID, Title, Behaviors | MIND dataset format |
| **Beauty** | JSONL | UserID, Title, Rating | Product reviews |
| **Steam** | JSON | UserID, Title, Rating | Game reviews |

## 📊 Data Processing Pipeline

### 1. Data Loading
```python
# Load dataset
data = pd.read_csv('data/books/ratings_Books.csv')

# Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=data,
    data_name='books',
    model='gpt-3.5-turbo',
    backend='openai'
)
```

### 2. User Selection
- **Bias Users**: Small set for bias detection (default: 5)
- **Evaluation Users**: Larger set for performance evaluation (default: 200)
- **Filtering**: Users with sufficient interaction history (≥6 items)

### 3. Candidate List Generation
- **History-based**: Uses user's past interactions
- **Size control**: Configurable candidate list length
- **Metadata integration**: Includes item and user features

### 4. Bias Analysis
- **Position shuffling**: Random reordering of candidates
- **LLM evaluation**: Consistent ranking across positions
- **Bias measurement**: Statistical analysis of position effects

### 5. Debiasing
- **Propensity scoring**: Item position preference modeling
- **Correction methods**: Adjusting rankings for bias
- **Effectiveness evaluation**: Before/after comparison

## 🔄 Workflow Patterns

### Basic Analysis
```python
# 1. Initialize
analyzer = LLMPositionBiasAnalyzer(...)

# 2. Run bias analysis
results = analyzer.run_bias_analysis()

# 3. View results
print(f"Bias Score: {results['bias_score']:.4f}")
```

### Advanced Analysis
```python
# 1. Run bias analysis
bias_results = analyzer.run_bias_analysis()

# 2. Apply debiasing
debiased_results = analyzer.run_debiased_evaluation(
    bias_scores=bias_results['bias_scores'],
    propensity_scores=bias_results['propensity_scores']
)

# 3. Compare results
analyzer.compare_results(bias_results, debiased_results)
```

### Checkpoint Management
```python
# Save progress
analyzer.save_checkpoint('experiment_001.json')

# Resume later
analyzer.load_checkpoint('experiment_001.json')
results = analyzer.run_bias_analysis()
```

## 🧪 Testing and Development

### Test Files
- **`test_*.py`**: Unit tests for specific components
- **`test_beauty_fix.py`**: Beauty dataset specific tests
- **`test_fixed_news_dataset.py`**: News dataset tests
- **`test_patched_method.py`**: Method patch tests

### Debug Utilities
- **`debug_json_parsing.py`**: JSON parsing debugging
- **`fix_*.py`**: Data processing fixes and patches

### Development Workflow
1. **Setup**: Install dependencies with `pip install -r requirements.txt`
2. **Testing**: Run tests with `python -m pytest tests/`
3. **Linting**: Check code quality with `flake8` and `black`
4. **Documentation**: Update docstrings and README

## 📈 Output and Results

### Generated Files
- **Checkpoints**: JSON files with experiment state
- **Results**: CSV files with analysis metrics
- **Plots**: PNG files with visualizations
- **Logs**: Text files with execution details

### Key Metrics
- **Bias Score**: Position bias magnitude (0-1)
- **Propensity Score**: Item position preference
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Accuracy**: Top-k recommendation accuracy

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Model Settings
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_BACKEND=openai
DEFAULT_API_TIER=tier_1

# Analysis Parameters
DEFAULT_NUM_BIAS_USERS=5
DEFAULT_NUM_EVAL_USERS=200
DEFAULT_NUM_SHUFFLES=50
```

### API Tiers
- **Basic**: Conservative (500 RPM, 200K TPM)
- **Tier 1**: Balanced (3500 RPM, 1M TPM)
- **Tier 2**: Aggressive (5000 RPM, 2M TPM)

## 🚀 Getting Started

### 1. Installation
```bash
git clone <repository>
cd debiased_ranking
pip install -r requirements.txt
cp env.example .env
# Edit .env with your API keys
```

### 2. Quick Start
```python
from LLM_debias import LLMPositionBiasAnalyzer
import pandas as pd

# Load data
data = pd.read_csv('data/books/ratings_Books.csv')

# Run analysis
analyzer = LLMPositionBiasAnalyzer(
    data=data,
    data_name='books',
    model='gpt-3.5-turbo',
    backend='openai'
)

results = analyzer.run_bias_analysis()
print(f"Bias Score: {results['bias_score']:.4f}")
```

### 3. Explore Examples
- Check `experiment_*.ipynb` notebooks for dataset-specific examples
- Review `evaluation_*.ipynb` for analysis workflows
- Examine `checkpoint_*.ipynb` for checkpoint management

## 🔍 Troubleshooting

### Common Issues
1. **API Rate Limits**: Use appropriate tier and monitor usage
2. **Memory Issues**: Process large datasets in chunks
3. **Data Format**: Ensure dataset matches expected column structure
4. **API Keys**: Verify environment variables are set correctly

### Debug Tips
- Enable verbose logging with `VERBOSE=true`
- Use checkpoint system for long experiments
- Monitor API usage in provider dashboard
- Check data quality before analysis

## 📚 Further Reading

- **Main Documentation**: See `README.md` for comprehensive guide
- **API Reference**: Check docstrings in `LLM_debias.py`
- **Examples**: Review Jupyter notebooks for practical usage
- **Research**: Examine analysis notebooks for methodology details
