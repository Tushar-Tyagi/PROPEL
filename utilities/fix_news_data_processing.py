#!/usr/bin/env python3
"""
Fix News Dataset Processing Issues
Addresses the problems causing low raw LLM accuracy (6%)
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Tuple
import os

def analyze_current_issues():
    """Analyze the current data processing issues."""
    print("🔍 Analyzing current news dataset issues...")
    
    # Load current processed data
    processed_df = pd.read_csv('./data/news/processed_df3.csv')
    
    print(f"📊 Current Dataset Statistics:")
    print(f"  Total rows: {len(processed_df):,}")
    print(f"  Unique UserIDs: {processed_df['UserID'].nunique():,}")
    
    # Analyze user types
    real_users = processed_df[~processed_df['UserID'].astype(str).str.contains('_neg_')]
    neg_users = processed_df[processed_df['UserID'].astype(str).str.contains('_neg_')]
    
    print(f"  Real users: {len(real_users):,} rows")
    print(f"  Negative samples: {len(neg_users):,} rows")
    print(f"  Real unique users: {real_users['UserID'].nunique():,}")
    print(f"  Negative unique users: {neg_users['UserID'].nunique():,}")
    
    # Check user interaction counts
    user_counts = real_users['UserID'].value_counts()
    print(f"\n📊 Real User Interaction Distribution:")
    print(f"  Users with 6+ interactions: {sum(user_counts >= 6):,}")
    print(f"  Users with 10+ interactions: {sum(user_counts >= 10):,}")
    print(f"  Users with 20+ interactions: {sum(user_counts >= 20):,}")
    
    # Check for target items
    print(f"\n🎯 Target Item Analysis:")
    target_items = real_users[real_users['Timestamp'] >= 0]['Title'].nunique()
    print(f"  Unique target items: {target_items:,}")
    
    return processed_df, real_users, neg_users

def create_fixed_news_dataset():
    """Create a properly processed news dataset without negative sample contamination."""
    print("\n🔧 Creating fixed news dataset...")
    
    # Load original data
    behaviors_file = "data/news/behaviors.tsv"
    news_file = "data/news/news.tsv"
    
    # Load news data
    news_columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
    news_df = pd.read_csv(news_file, sep='\t', header=None, names=news_columns)
    
    # Create news ID to title mapping
    news_id_to_title = {}
    for _, row in news_df.iterrows():
        news_id = row['NewsID']
        title = row['Title']
        category = row['Category']
        subcategory = row['SubCategory']
        formatted_title = f"{title}[Category:{category},Subcategory:{subcategory}]"
        news_id_to_title[news_id] = formatted_title
    
    # Load behaviors data
    behaviors_columns = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
    behaviors_df = pd.read_csv(behaviors_file, sep='\t', header=None, names=behaviors_columns)
    
    print(f"📊 Original data:")
    print(f"  Behaviors: {len(behaviors_df):,} rows")
    print(f"  News articles: {len(news_df):,} rows")
    
    # Process data properly
    processed_data = []
    processed_users = set()
    
    print(f"\n🔄 Processing behaviors...")
    
    for idx, row in behaviors_df.iterrows():
        user_id = row['UserID']
        
        # Skip if user already processed
        if user_id in processed_users:
            continue
            
        history = row['History']
        impressions = row['Impressions']
        
        # Skip if no history or impressions
        if pd.isna(history) or pd.isna(impressions):
            continue
        
        # Parse history
        history_news_ids = history.split()
        
        # Skip if insufficient history (need at least 5 for history + 1 for target)
        if len(history_news_ids) < 6:
            continue
        
        # Parse impressions
        impression_items = impressions.split()
        clicked_news = []
        unclicked_news = []
        
        for item in impression_items:
            if '-' in item:
                news_id, click = item.split('-')
                if click == '1':
                    clicked_news.append(news_id)
                else:
                    unclicked_news.append(news_id)
        
        # Skip if no clicked news
        if not clicked_news:
            continue
        
        # Take one clicked news as target
        target_news_id = clicked_news[0]  # Take first clicked
        
        # Get title for target
        target_title = news_id_to_title.get(target_news_id, f"Unknown_{target_news_id}")
        
        # Get last 5 history items as titles
        history_titles = []
        for nid in history_news_ids[-5:]:
            title = news_id_to_title.get(nid, f"Unknown_{nid}")
            history_titles.append(title)
        
        # Create entries for history items
        current_time = idx
        
        # Add history items
        for i, title in enumerate(history_titles):
            processed_data.append({
                'UserID': user_id,
                'Title': title,
                'Timestamp': current_time - len(history_titles) + i
            })
        
        # Add target item
        processed_data.append({
            'UserID': user_id,
            'Title': target_title,
            'Timestamp': current_time
        })
        
        processed_users.add(user_id)
        
        # Limit to first 1000 users for testing
        if len(processed_users) >= 1000:
            break
    
    fixed_df = pd.DataFrame(processed_data)
    
    print(f"\n✅ Fixed Dataset Statistics:")
    print(f"  Total rows: {len(fixed_df):,}")
    print(f"  Unique users: {fixed_df['UserID'].nunique():,}")
    print(f"  Unique titles: {fixed_df['Title'].nunique():,}")
    
    # Check user interaction counts
    user_counts = fixed_df['UserID'].value_counts()
    print(f"\n📊 User Interaction Distribution:")
    print(f"  Users with 6+ interactions: {sum(user_counts >= 6):,}")
    print(f"  Users with 10+ interactions: {sum(user_counts >= 10):,}")
    
    # Save fixed dataset
    output_path = './data/news/processed_df_fixed.csv'
    fixed_df.to_csv(output_path, index=False)
    print(f"\n💾 Fixed dataset saved to: {output_path}")
    
    return fixed_df

def test_fixed_dataset():
    """Test the fixed dataset with a simple evaluation."""
    print("\n🧪 Testing fixed dataset...")
    
    # Load fixed dataset
    fixed_df = pd.read_csv('./data/news/processed_df_fixed.csv')
    
    # Sample a few users for testing
    sample_users = fixed_df['UserID'].unique()[:5]
    
    print(f"📊 Testing with {len(sample_users)} sample users:")
    
    for user_id in sample_users:
        user_data = fixed_df[fixed_df['UserID'] == user_id].sort_values('Timestamp')
        user_items = user_data['Title'].tolist()
        
        print(f"\n👤 User {user_id}:")
        print(f"  Total interactions: {len(user_items)}")
        
        # Show history and target
        if len(user_items) >= 6:
            history = user_items[:-1]  # All but last
            target = user_items[-1]    # Last item
            
            print(f"  History (last 3): {history[-3:]}")
            print(f"  Target: {target[:80]}...")
        else:
            print(f"  ⚠️ Insufficient data")

def create_improved_processing_script():
    """Create an improved data processing script for the notebook."""
    
    script = '''
# Improved News Dataset Processing
# This fixes the negative sample contamination issue

def process_mind_dataset_improved(behaviors_df, news_df, min_history_length=5, max_users=1000):
    """
    Process MIND dataset properly without negative sample contamination.
    
    Args:
        behaviors_df: DataFrame with user behaviors
        news_df: DataFrame with news information
        min_history_length: Minimum required history length
        max_users: Maximum number of users to process
    
    Returns:
        DataFrame suitable for LLMPositionBiasAnalyzer
    """
    processed_data = []
    
    # Create news ID to title mapping
    news_id_to_title = {}
    for _, row in news_df.iterrows():
        news_id = row['NewsID']
        title = row['Title']
        category = row['Category']
        subcategory = row['SubCategory']
        formatted_title = f"{title}[Category:{category},Subcategory:{subcategory}]"
        news_id_to_title[news_id] = formatted_title
    
    processed_users = set()
    
    print(f"🔄 Processing up to {max_users} users...")
    
    for idx, row in behaviors_df.iterrows():
        user_id = row['UserID']
        
        # Skip if user already processed or limit reached
        if user_id in processed_users or len(processed_users) >= max_users:
            continue
            
        history = row['History']
        impressions = row['Impressions']
        
        # Skip if no history or impressions
        if pd.isna(history) or pd.isna(impressions):
            continue
        
        # Parse history
        history_news_ids = history.split()
        
        # Skip if insufficient history
        if len(history_news_ids) < min_history_length + 1:  # +1 for target
            continue
        
        # Parse impressions
        impression_items = impressions.split()
        clicked_news = []
        
        for item in impression_items:
            if '-' in item:
                news_id, click = item.split('-')
                if click == '1':
                    clicked_news.append(news_id)
        
        # Skip if no clicked news
        if not clicked_news:
            continue
        
        # Take first clicked news as target
        target_news_id = clicked_news[0]
        target_title = news_id_to_title.get(target_news_id, f"Unknown_{target_news_id}")
        
        # Get last N history items as titles
        history_titles = []
        for nid in history_news_ids[-(min_history_length):]:
            title = news_id_to_title.get(nid, f"Unknown_{nid}")
            history_titles.append(title)
        
        # Create entries
        current_time = idx
        
        # Add history items
        for i, title in enumerate(history_titles):
            processed_data.append({
                'UserID': user_id,
                'Title': title,
                'Timestamp': current_time - len(history_titles) + i
            })
        
        # Add target item
        processed_data.append({
            'UserID': user_id,
            'Title': target_title,
            'Timestamp': current_time
        })
        
        processed_users.add(user_id)
    
    print(f"✅ Processed {len(processed_users)} unique users")
    print(f"✅ Generated {len(processed_data)} total data points")
    
    return pd.DataFrame(processed_data)

# Usage:
# processed_df_improved = process_mind_dataset_improved(behaviors_df, news_df, min_history_length=5, max_users=1000)
# processed_df_improved.to_csv('./data/news/processed_df_improved.csv', index=False)
'''
    
    with open('improved_news_processing.py', 'w') as f:
        f.write(script)
    
    print(f"\n💾 Improved processing script saved to: improved_news_processing.py")

if __name__ == "__main__":
    print("🚀 News Dataset Processing Fix")
    print("=" * 50)
    
    # Analyze current issues
    processed_df, real_users, neg_users = analyze_current_issues()
    
    # Create fixed dataset
    fixed_df = create_fixed_news_dataset()
    
    # Test fixed dataset
    test_fixed_dataset()
    
    # Create improved processing script
    create_improved_processing_script()
    
    print(f"\n✅ Fix complete!")
    print(f"\n📋 Next steps:")
    print(f"  1. Use the fixed dataset: data/news/processed_df_fixed.csv")
    print(f"  2. Or use the improved processing script: improved_news_processing.py")
    print(f"  3. Re-run your evaluation with the fixed data")
    print(f"  4. Expected improvement: Raw LLM accuracy should increase from 6% to ~20-30%") 