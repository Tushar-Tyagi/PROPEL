# Checkpoints Directory

This directory contains experiment checkpoints and results from the LLM Position Bias Analysis Framework.

## 📁 Contents

### Evaluation Checkpoints
- **`evaluation_checkpoint_*.json`**: Large checkpoint files containing complete experiment state
- **`checkpoint_analysis_results_*.csv`**: Analysis results from checkpoint processing

### Usage

These files are used to:
- Resume interrupted experiments
- Analyze previous results
- Compare different experimental runs
- Debug analysis issues

### File Sizes

⚠️ **Note**: Large checkpoint files (>10MB) have been removed from this repository to save space and improve git performance. The remaining files are smaller examples and can be used as templates.

Original large files contained:
- Complete experiment state (50MB-155MB each)
- Raw LLM responses
- Intermediate calculations
- User interaction data

### Management

- **Keep**: Recent checkpoints for active research
- **Archive**: Old checkpoints to external storage
- **Delete**: Failed or obsolete experiments
- **Compress**: Use gzip for long-term storage

## 🔄 Loading Checkpoints

```python
from LLM_debias import LLMPositionBiasAnalyzer

# Load a checkpoint
analyzer = LLMPositionBiasAnalyzer(...)
analyzer.load_checkpoint('checkpoints/evaluation_checkpoint_experiment.json')

# Resume analysis
results = analyzer.run_bias_analysis()
```

## 📊 Analysis Results

The CSV files contain:
- Bias scores across different conditions
- Statistical significance tests
- Performance metrics (NDCG, Accuracy)
- User-level analysis results

## 🧹 Cleanup

To free up disk space:
```bash
# List large files
ls -lh *.json | sort -k5 -hr

# Archive old checkpoints
tar -czf old_checkpoints.tar.gz evaluation_checkpoint_old_*.json

# Remove archived files
rm evaluation_checkpoint_old_*.json
```
