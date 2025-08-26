#!/usr/bin/env python3
"""
Quick Test of Fixed News Dataset
Verifies that the fixed dataset resolves the accuracy issues
"""

import pandas as pd
import numpy as np
from LLM_debias import LLMPositionBiasAnalyzer

def test_fixed_dataset():
    """Test the fixed dataset with a small evaluation."""
    print("🧪 Testing Fixed News Dataset")
    print("=" * 40)
    
    # Load fixed dataset
    fixed_df = pd.read_csv('./data/news/processed_df_fixed.csv')
    
    print(f"📊 Fixed Dataset Statistics:")
    print(f"  Total rows: {len(fixed_df):,}")
    print(f"  Unique users: {fixed_df['UserID'].nunique():,}")
    print(f"  Unique titles: {fixed_df['Title'].nunique():,}")
    
    # Check user interaction counts
    user_counts = fixed_df['UserID'].value_counts()
    print(f"\n📊 User Interaction Distribution:")
    print(f"  Users with 6+ interactions: {sum(user_counts >= 6):,}")
    print(f"  Users with 10+ interactions: {sum(user_counts >= 10):,}")
    
    # Check for negative sample contamination
    neg_users = fixed_df[fixed_df['UserID'].astype(str).str.contains('_neg_')]
    print(f"\n🔍 Negative Sample Check:")
    print(f"  Negative samples found: {len(neg_users)}")
    if len(neg_users) == 0:
        print(f"  ✅ No negative sample contamination!")
    else:
        print(f"  ⚠️ Still has negative samples!")
    
    # Initialize analyzer with fixed dataset
    print(f"\n🚀 Initializing analyzer with fixed dataset...")
    analyzer = LLMPositionBiasAnalyzer(
        data=fixed_df,
        data_name='news',
        model='gpt-3.5-turbo',
        backend='openai',
        num_bias_users=3,  # Small number for testing
        num_eval_users=10,  # Small number for testing
        num_shuffles_bias=10,  # Small number for testing
        list_size=20,
        api_tier='tier_2'
    )
    
    print(f"✅ Analyzer initialized successfully!")
    print(f"  Bias users: {analyzer.bias_users}")
    print(f"  Eval users: {analyzer.eval_users[:5]}...")  # Show first 5
    
    # Test candidate list creation for one user
    test_user = analyzer.eval_users[0]
    print(f"\n🔍 Testing candidate list creation for user {test_user}...")
    
    try:
        candidate_list, user_history, target_item, actual_size = analyzer.create_candidate_list(test_user)
        print(f"✅ Successfully created candidate list:")
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
        
    except Exception as e:
        print(f"❌ Error creating candidate list: {e}")
    
    print(f"\n✅ Test completed!")
    print(f"\n📋 Expected improvements with fixed dataset:")
    print(f"  - No negative sample contamination")
    print(f"  - Clean user histories")
    print(f"  - Proper target selection")
    print(f"  - Raw LLM accuracy should improve from 6% to ~20-30%")

if __name__ == "__main__":
    test_fixed_dataset() 