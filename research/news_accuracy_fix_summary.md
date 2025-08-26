# News Dataset Accuracy Fix Summary

## Problem Identified
Your raw LLM accuracy was only **6%** on the news dataset, which is significantly lower than the expected **~26%** from benchmarks.

## Root Cause Analysis

### 1. **Data Processing Issue - Negative Sample Contamination**
- **Original dataset**: 808,923 total rows
- **Real users**: 241,986 rows (30%)
- **Negative samples**: 566,937 rows (70%) with user IDs like `U13740_neg_0`, `U13740_neg_1`, etc.

### 2. **Candidate List Creation Problem**
The `create_candidate_list` method in `LLM_debias.py` was designed to work with negative samples:
```python
# Original problematic code
neg_pattern = f"{user_id}_neg_"
neg_users = [uid for uid in self.data['UserID'].unique() if str(uid).startswith(neg_pattern)]
available_items = self.data[self.data['UserID'].isin(neg_users)].drop_duplicates(item_name)
```

When the fixed dataset (without negative samples) was used, this resulted in:
- **Only 1 candidate** instead of 20
- **No proper negative sampling**
- **Flawed evaluation setup**

### 3. **Evaluation Quality Issues**
- LLM was being asked to rank among artificially created negative samples
- Target selection was contaminated by negative sample processing
- User histories were mixed with negative samples

## Solutions Applied

### 1. **Fixed Dataset Processing**
Created `data/news/processed_df_fixed.csv` with:
- **6,000 total rows** (clean, no negative samples)
- **1,000 unique users** with 6+ interactions each
- **2,601 unique news titles**
- **Proper target selection** from clicked news items
- **Clean user histories** from actual user interactions

### 2. **Patched Candidate List Method**
Created `patch_create_candidate_list.py` that:
- **Removes dependency on negative samples**
- **Samples from all available news items** (excluding user's seen items)
- **Creates proper 20-candidate lists** (19 negatives + 1 target)
- **Maintains evaluation integrity**

### 3. **Improved Data Processing Script**
Created `improved_news_processing.py` with:
- **Proper MIND dataset processing**
- **No negative sample contamination**
- **Clean user-target pairs**
- **Configurable user limits**

## Files Created

1. **`fix_news_data_processing.py`** - Main fix script
2. **`data/news/processed_df_fixed.csv`** - Fixed dataset
3. **`patch_create_candidate_list.py`** - Method patch
4. **`test_fixed_news_dataset.py`** - Dataset test
5. **`test_patched_method.py`** - Method test
6. **`improved_news_processing.py`** - Improved processing script

## Expected Results

With the fixes applied:
- **Raw LLM accuracy**: Should improve from 6% to **~20-30%**
- **Proper candidate lists**: 20 candidates per evaluation
- **Clean evaluation**: No negative sample contamination
- **Better bias detection**: More meaningful position bias analysis

## Usage Instructions

### Option 1: Use Fixed Dataset + Patch
```python
# Load fixed dataset
fixed_df = pd.read_csv('./data/news/processed_df_fixed.csv')

# Initialize analyzer
analyzer = LLMPositionBiasAnalyzer(
    data=fixed_df,
    data_name='news',
    model='gpt-3.5-turbo',
    backend='openai',
    num_bias_users=5,
    num_eval_users=200,
    num_shuffles_bias=50,
    list_size=20,
    api_tier='tier_2'
)

# Apply patch
from patch_create_candidate_list import patch_create_candidate_list
patch_create_candidate_list(analyzer)

# Run evaluation
results = analyzer.evaluate_our_method_batched(
    num_candidates=20,
    num_trials=20,
    aggregation_method="mean",
    batch_size=20,
    use_parallel=True,
    checkpoint_file="evaluation_checkpoint_news_fixed.json"
)
```

### Option 2: Re-process with Improved Script
```python
# Use improved processing script
from improved_news_processing import process_mind_dataset_improved

# Load original data
behaviors_df = pd.read_csv('data/news/behaviors.tsv', sep='\t', header=None, 
                          names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
news_df = pd.read_csv('data/news/news.tsv', sep='\t', header=None,
                      names=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])

# Process with improved method
processed_df_improved = process_mind_dataset_improved(behaviors_df, news_df, min_history_length=5, max_users=1000)
processed_df_improved.to_csv('./data/news/processed_df_improved.csv', index=False)
```

## Verification Steps

1. **Test dataset**: `python test_fixed_news_dataset.py`
2. **Test method**: `python test_patched_method.py`
3. **Check candidate lists**: Should have 20 candidates per user
4. **Verify no negative samples**: Dataset should be clean
5. **Run small evaluation**: Test with 10-20 users first

## Key Improvements

- ✅ **No negative sample contamination**
- ✅ **Proper candidate list creation**
- ✅ **Clean user histories**
- ✅ **Realistic target selection**
- ✅ **Expected accuracy improvement**

The fixes address the fundamental data processing issues that were causing the low accuracy, and should result in much more realistic and meaningful evaluation results. 