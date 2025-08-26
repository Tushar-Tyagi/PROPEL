# Results Directory

This directory contains generated outputs, analysis results, and visualizations from the LLM Position Bias Analysis Framework.

## 📁 Contents

### Generated Outputs
- **Analysis Results**: CSV files with evaluation metrics
- **Visualizations**: Plots and charts from experiments
- **Processed Data**: Cleaned and transformed datasets
- **Model Outputs**: LLM response summaries

### File Types
- **`.csv`**: Tabular data and metrics
- **`.json`**: Structured results and configurations
- **`.png`**: Generated plots and visualizations
- **`.txt`**: Text-based outputs and logs

## 🎯 Usage

### Accessing Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load analysis results
results = pd.read_csv('results/analysis_results.csv')

# Load generated plots
plot = plt.imread('results/plots/bias_analysis.png')
```

### Results Structure
- **User-level Results**: Individual user bias scores and preferences
- **Dataset-level Results**: Aggregate statistics and comparisons
- **Model-level Results**: Performance across different LLM models
- **Time-series Results**: Results over multiple experimental runs

## 📊 Output Categories

### Bias Analysis Results
- **Position Bias Scores**: Magnitude of position bias (0-1)
- **Propensity Scores**: Item position preference metrics
- **Statistical Tests**: P-values and confidence intervals
- **User Segments**: Bias patterns across user groups

### Evaluation Metrics
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Accuracy@k**: Top-k recommendation accuracy
- **Precision@k**: Precision at different positions
- **Recall@k**: Recall at different positions

### Comparative Analysis
- **Before/After**: Original vs. debiased rankings
- **Model Comparison**: Performance across different LLMs
- **Dataset Comparison**: Bias patterns across domains
- **Algorithm Comparison**: Different debiasing methods

## 🔍 File Naming Convention

### Standard Format
```
{experiment_type}_{dataset}_{model}_{timestamp}_{metric}.{extension}
```

### Examples
- `bias_analysis_books_gpt35_20240115_bias_scores.csv`
- `evaluation_movielens_gpt4_20240115_ndcg_results.csv`
- `comparison_music_books_20240115_model_performance.png`

### Components
- **Experiment Type**: bias_analysis, evaluation, comparison
- **Dataset**: books, movielens, music, news, beauty, steam
- **Model**: gpt35, gpt4, claude, custom
- **Timestamp**: YYYYMMDD format
- **Metric**: bias_scores, ndcg_results, performance

## 📈 Visualization Outputs

### Plot Types
- **Bias Distribution**: Histograms of bias scores
- **Performance Comparison**: Bar charts of metrics
- **Time Series**: Results over experimental runs
- **User Analysis**: Individual user behavior patterns

### Plot Formats
- **High Resolution**: 300+ DPI for publications
- **Vector Graphics**: SVG for web and scaling
- **Standard Sizes**: Consistent dimensions across plots
- **Color Schemes**: Accessible and publication-ready

## 🗂️ Organization

### Directory Structure
```
results/
├── analysis/           # Tabular results and metrics
├── plots/             # Generated visualizations
├── processed_data/    # Cleaned and transformed data
├── model_outputs/     # LLM response summaries
├── logs/              # Execution logs and timestamps
└── archives/          # Historical results and backups
```

### File Management
- **Active Results**: Current experimental outputs
- **Archived Results**: Historical data and backups
- **Temporary Files**: Intermediate processing results
- **Final Results**: Clean, validated outputs

## 🔄 Results Lifecycle

### Generation
1. **Experiment Execution**: Run bias analysis experiments
2. **Data Processing**: Clean and transform raw results
3. **Analysis**: Calculate metrics and statistics
4. **Visualization**: Create plots and charts
5. **Validation**: Verify result quality and accuracy

### Storage
1. **Immediate**: Save to results directory
2. **Backup**: Copy to archive location
3. **Version Control**: Track changes in git
4. **Documentation**: Update result summaries

### Cleanup
1. **Temporary Files**: Remove intermediate results
2. **Old Results**: Archive or delete outdated outputs
3. **Storage Optimization**: Compress large result files
4. **Quality Control**: Validate result integrity

## 🚨 Best Practices

### Result Management
- **Naming**: Use consistent, descriptive file names
- **Organization**: Group related results in subdirectories
- **Documentation**: Include metadata and context
- **Backup**: Regular backups of important results

### Quality Assurance
- **Validation**: Verify result accuracy and completeness
- **Reproducibility**: Ensure results can be regenerated
- **Documentation**: Clear explanation of methodology
- **Version Control**: Track result evolution over time

### Performance
- **Storage**: Monitor disk space usage
- **Compression**: Use appropriate compression for large files
- **Cleanup**: Regular removal of temporary files
- **Archiving**: Move old results to long-term storage

## 📊 Result Analysis

### Key Metrics
- **Bias Magnitude**: Overall position bias in dataset
- **User Variation**: Individual differences in bias
- **Model Performance**: LLM effectiveness in bias detection
- **Algorithm Effectiveness**: Debiasing method performance

### Statistical Significance
- **P-values**: Statistical significance of results
- **Confidence Intervals**: Uncertainty in estimates
- **Effect Sizes**: Practical significance of findings
- **Multiple Comparisons**: Correction for multiple tests

## 🔮 Future Enhancements

### Planned Features
- **Automated Analysis**: Scripts for result processing
- **Interactive Dashboards**: Web-based result exploration
- **Result Comparison**: Tools for cross-experiment analysis
- **Export Formats**: Multiple output formats (Excel, LaTeX)

### Integration
- **Database Storage**: Structured result storage
- **API Access**: Programmatic result retrieval
- **Visualization Tools**: Enhanced plotting capabilities
- **Report Generation**: Automated result summaries

---

**📊 The results directory provides a comprehensive view of all experimental outputs and analysis results from the LLM Position Bias Analysis Framework.**
