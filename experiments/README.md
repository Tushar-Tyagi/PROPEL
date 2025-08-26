# Experiments Directory

This directory contains Jupyter notebooks for different experiments and analyses using the LLM Position Bias Analysis Framework.

## 📁 Contents

### Dataset-Specific Experiments
- **`experiment_beauty.ipynb`**: Beauty product dataset analysis
- **`experiment_books.ipynb`**: Books dataset analysis  
- **`experiment_cds_vinyl.ipynb`**: Music dataset analysis
- **`experiment_news_mind.ipynb`**: News (MIND) dataset analysis
- **`experiment_steam.ipynb`**: Steam games dataset analysis
- **`experiment_notebook_movielens.ipynb`**: MovieLens dataset analysis
- **`experiment_notebook.ipynb`**: General experiment template

### Evaluation and Analysis
- **`evaluation_results_analysis.ipynb`**: Analysis of evaluation results
- **`checkpoint_analysis.ipynb`**: Analysis of checkpoint data
- **`borda.ipynb`**: Borda count ranking analysis
- **`plot.ipynb`**: Visualization and plotting utilities

### Development
- **`Untitled.ipynb`**: Development scratchpad (can be removed)

## 🎯 Usage

### Running Experiments
1. **Choose dataset**: Select appropriate experiment notebook
2. **Configure parameters**: Set model, API keys, dataset paths
3. **Run analysis**: Execute cells sequentially
4. **Save results**: Use checkpoint system for long experiments

### Example Workflow
```python
# 1. Load dataset
data = pd.read_csv('data/books/ratings_Books.csv')

# 2. Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=data,
    data_name='books',
    model='gpt-3.5-turbo',
    backend='openai'
)

# 3. Run analysis
results = analyzer.run_bias_analysis()

# 4. Save checkpoint
analyzer.save_checkpoint('checkpoints/books_experiment.json')
```

## 📊 Experiment Types

### Bias Detection
- Position bias measurement
- User behavior analysis
- Statistical significance testing

### Debiasing
- Propensity scoring
- Ranking correction
- Effectiveness evaluation

### Comparison Studies
- Model performance comparison
- Dataset characteristics analysis
- Algorithm effectiveness testing

## 🔧 Customization

### Adding New Datasets
1. Copy existing experiment notebook
2. Modify data loading section
3. Update dataset configuration
4. Test with small sample first

### Parameter Tuning
- **API tiers**: Adjust rate limiting
- **User counts**: Balance bias vs evaluation users
- **Shuffle counts**: Trade-off between accuracy and speed
- **List sizes**: Consider computational constraints

## 📈 Results Interpretation

### Key Metrics
- **Bias Score**: Position bias magnitude (0-1)
- **Propensity Score**: Item position preference
- **NDCG@k**: Ranking quality at position k
- **Accuracy**: Top-k recommendation accuracy

### Statistical Significance
- Use provided significance tests
- Consider multiple comparison corrections
- Report confidence intervals

## 🚨 Best Practices

### Performance
- Start with small datasets for testing
- Use appropriate API tiers
- Monitor API usage and costs
- Use checkpoint system for long runs

### Reproducibility
- Set random seeds
- Document all parameters
- Save intermediate results
- Version control notebooks

### Data Management
- Clean data before analysis
- Handle missing values appropriately
- Validate data format requirements
- Backup important results

## 🧹 Maintenance

### Regular Cleanup
- Remove failed experiments
- Archive completed analyses
- Update documentation
- Validate results

### Version Control
- Commit working experiments
- Tag successful runs
- Document parameter changes
- Track performance improvements
