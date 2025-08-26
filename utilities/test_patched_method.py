
import pandas as pd
from LLM_debias import LLMPositionBiasAnalyzer

# Load fixed dataset
fixed_df = pd.read_csv('./data/news/processed_df_fixed.csv')

# Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=fixed_df,
    data_name='news',
    model='gpt-3.5-turbo',
    backend='openai',
    num_bias_users=3,
    num_eval_users=10,
    num_shuffles_bias=10,
    list_size=20,
    api_tier='tier_2'
)

# Apply patch
from patch_create_candidate_list import patch_create_candidate_list
patch_create_candidate_list(analyzer)

# Test candidate list creation
test_user = analyzer.eval_users[0]
candidate_list, user_history, target_item, actual_size = analyzer.create_candidate_list(test_user)

print(f"✅ Patched method test:")
print(f"  Candidates: {actual_size}")
print(f"  History items: {len(user_history)}")
print(f"  Target item: {target_item[:80]}...")

# Check if target is in candidates
candidate_titles = [item['title'] for item in candidate_list]
target_in_candidates = target_item in candidate_titles
print(f"  Target in candidates: {target_in_candidates}")

if target_in_candidates:
    target_position = candidate_titles.index(target_item)
    print(f"  Target position: {target_position + 1}")
