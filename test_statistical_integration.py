import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM_debias import LLMPositionBiasAnalyzer
import pandas as pd
import numpy as np

# Create stub data
data = {
    'UserID': [1,1,1,1,1,1, 2,2,2,2,2,2],
    'Title': ['A','B','C','D','E','F', 'G','H','I','J','K','L'],
    'Rating': [5,4,3,4,5,5, 5,5,5,4,4,5],
    'Timestamp': [1,2,3,4,5,6, 1,2,3,4,5,6]
}
df = pd.DataFrame(data)

# Create mock analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=df,
    data_name='movie_lens',
    model='gpt-3.5-turbo',
    backend='openai'
)

# Mock _parallel_evaluate_users to return predefined structured results instead of hitting API
def mock_evaluate(eval_users, num_candidates, num_trials, aggregation_method, propensity_scores, w, w2):
    return [
        {'accuracy': 0.3, 'ndcg_1': 0.3, 'ndcg_5': 0.4, 'ndcg_10': 0.45, 'ndcg_20': 0.5},
        {'accuracy': 0.35, 'ndcg_1': 0.35, 'ndcg_5': 0.42, 'ndcg_10': 0.48, 'ndcg_20': 0.51},
        {'accuracy': 0.4, 'ndcg_1': 0.4, 'ndcg_5': 0.48, 'ndcg_10': 0.52, 'ndcg_20': 0.55},
        {'accuracy': 0.25, 'ndcg_1': 0.25, 'ndcg_5': 0.38, 'ndcg_10': 0.41, 'ndcg_20': 0.45},
        {'accuracy': 0.32, 'ndcg_1': 0.32, 'ndcg_5': 0.41, 'ndcg_10': 0.46, 'ndcg_20': 0.52}
    ]
    
analyzer._parallel_evaluate_users = mock_evaluate

# Test execution with precalculated bias (skips detection step)
bias_dict = {'avg_primacy': 5.0, 'avg_recency': 2.0, 'avg_middle': 0.5}

print("Running test evaluation method...")
results = analyzer.evaluate_our_method(
    num_eval_users=5,
    num_candidates=20,
    precalculated_bias=bias_dict,
    use_parallel=True
)

print("Test finished successfully!")
