
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
