# LLM Position Bias Analysis Framework

A comprehensive framework for detecting and correcting position bias in Large Language Model (LLM) based recommender systems. This project provides tools to analyze how position bias affects recommendation quality and implements debiasing strategies.

## 🎯 Project Overview

Position bias occurs when users prefer items that appear earlier in recommendation lists, regardless of their actual relevance. This framework helps researchers and developers:

- **Detect position bias** in LLM-based recommender systems
- **Measure bias impact** on recommendation quality metrics (NDCG, Accuracy)
- **Implement debiasing strategies** using propensity scoring
- **Evaluate debiasing effectiveness** across multiple datasets

## 🚀 Features

- **Multi-dataset support**: MovieLens, Books, Music, News, Beauty, Steam
- **Flexible LLM backends**: OpenAI GPT models, Claude, custom models
- **Comprehensive evaluation**: NDCG@k, Accuracy, Bias scores, Propensity analysis
- **Rate limiting**: Built-in API rate limiting with configurable tiers
- **Batch processing**: Efficient parallel processing for large-scale analysis
- **Checkpoint system**: Save and resume long-running experiments

## 📊 Supported Datasets

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

## 🔑 Configuration

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

## 📈 Performance Optimization

### Rate Limiting

- Use appropriate API tier for your usage
- Monitor API usage in OpenAI dashboard
- Implement exponential backoff for errors

### Memory Management

- For large datasets, process in chunks
- Use checkpoint system for long experiments
- Monitor memory usage during processing

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_bias_analysis.py

# Run with coverage
python -m pytest --cov=LLM_debias tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

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

## 📚 Documentation

- **API Reference**: See docstrings in `LLM_debias.py`
- **Examples**: Check the `experiment_*.ipynb` notebooks
- **Research**: Review the analysis notebooks for methodology

## 🤝 Citation

If you use this framework in your research, please cite:

(Under publication)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact the maintainers directly

## 🔮 Roadmap

- [ ] Support for more LLM providers (Mistral, Cohere)
- [ ] Advanced debiasing algorithms
- [ ] Real-time bias monitoring
- [ ] Web interface for analysis
- [ ] Integration with popular ML frameworks

## 🧹 Repository Notes

This repository has been cleaned up for optimal developer experience:

- **🔒 Security**: Hardcoded API keys removed - use environment variables
- **📦 Size**: Large datasets and checkpoint files removed (24GB+ saved)
- **📁 Organization**: Files organized into logical directories
- **🚫 Git**: Large files excluded from version control
- **📚 Documentation**: Comprehensive guides for each component

### Data and Checkpoints
- **Datasets**: Download instructions in `data/README.md`
- **Checkpoints**: Generate new ones or use small examples in `checkpoints/`
- **Results**: Outputs will be saved to `results/` directory

---

**Note**: This framework requires API credits for LLM providers. Monitor your usage to avoid unexpected costs.
