#!/usr/bin/env python3
"""
Fix for create_candidate_list method
Makes it work with the fixed news dataset that doesn't have negative samples
"""

def create_fixed_candidate_list_method():
    """Create a fixed version of the create_candidate_list method."""
    
    fixed_method = '''
    def create_candidate_list_fixed(self, user_id: int = None) -> Tuple[List[Dict], List[str], str, int]:
        """
        Create candidate list for evaluation - FIXED VERSION for news dataset.
        
        Args:
            user_id: User ID to create candidates for
            
        Returns:
            Tuple of (candidate_list, user_history, target_item, actual_candidate_size)
        """
        item_name, _, _, _ = get_data_columns(self.data_name)
        
        if user_id:
            if self.data_name == 'news':
                # For news dataset, get user's actual interactions
                user_items = set(self.data[self.data['UserID'] == user_id][item_name].values)
                user_items_list = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')[item_name].tolist()
                
                # Get all available items excluding user's items
                all_items = self.data.drop_duplicates(item_name)
                available_items = all_items[~all_items[item_name].isin(user_items)]
                
                # For bias users, ensure we have exactly self.list_size candidates
                if user_id in self.bias_users and len(available_items) < self.list_size:
                    # If not enough items, use all available items
                    print(f"⚠️  User {user_id}: Only {len(available_items)} available items (need {self.list_size})")
                    # Use all available items
                    sampled_items = available_items
                else:
                    # Sample random items
                    items_to_sample = min(self.list_size - 1, len(available_items))  # -1 for target
                    if items_to_sample > 0:
                        sampled_items = available_items.sample(n=items_to_sample, random_state=42)
                    else:
                        sampled_items = available_items.head(items_to_sample)
            else:
                # Get user's rated movies to exclude
                user_items = set(self.data[self.data['UserID'] == user_id][item_name].values)
                user_items_list = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')[item_name].tolist()
                available_items = self.data[~self.data[item_name].isin(user_items)].drop_duplicates(item_name)
                
                # Sample random items
                items_to_sample = min(self.list_size - 1, len(available_items))  # -1 for target
                if items_to_sample > 0:
                    sampled_items = available_items.sample(n=items_to_sample, random_state=42)
                else:
                    sampled_items = available_items.head(items_to_sample)
        else:
            available_items = self.data.drop_duplicates(item_name)
            items_to_sample = min(self.list_size, len(available_items))
            if items_to_sample > 0:
                sampled_items = available_items.sample(n=items_to_sample, random_state=42)
            else:
                sampled_items = available_items.head(items_to_sample)
            user_items_list = []
        
        candidate_titles = sampled_items[item_name].tolist()
        
        # Extract and append last item from user_items_list as target
        if user_items_list:
            last_item = user_items_list.pop()  # remove and store last item
            candidate_titles.append(last_item)   # add it to candidate list
        else:
            last_item = None
        
        # Convert to dictionary format with original positions
        candidate_list = [
            {
                'title': title,
                'original_position': i
            }
            for i, title in enumerate(candidate_titles)
        ]
        
        # Return actual candidate list size for dynamic propensity score calculation
        actual_candidate_size = len(candidate_list)
        return candidate_list, user_items_list, last_item, actual_candidate_size
    '''
    
    return fixed_method

def create_patch_for_llm_debias():
    """Create a patch to fix the create_candidate_list method in LLM_debias.py."""
    
    patch_code = '''
# PATCH: Fix create_candidate_list method for news dataset
# Add this to your notebook or create a separate file

def patch_create_candidate_list(analyzer):
    """Patch the create_candidate_list method to work with fixed news dataset."""
    
    def create_candidate_list_fixed(self, user_id: int = None):
        """Fixed version of create_candidate_list for news dataset."""
        from LLM_debias import get_data_columns
        
        item_name, _, _, _ = get_data_columns(self.data_name)
        
        if user_id:
            if self.data_name == 'news':
                # For news dataset, get user's actual interactions
                user_items = set(self.data[self.data['UserID'] == user_id][item_name].values)
                user_items_list = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')[item_name].tolist()
                
                # Get all available items excluding user's items
                all_items = self.data.drop_duplicates(item_name)
                available_items = all_items[~all_items[item_name].isin(user_items)]
                
                # Sample random items
                items_to_sample = min(self.list_size - 1, len(available_items))  # -1 for target
                if items_to_sample > 0:
                    sampled_items = available_items.sample(n=items_to_sample, random_state=42)
                else:
                    sampled_items = available_items.head(items_to_sample)
            else:
                # Original logic for other datasets
                user_items = set(self.data[self.data['UserID'] == user_id][item_name].values)
                user_items_list = self.data[self.data['UserID'] == user_id].sort_values('Timestamp')[item_name].tolist()
                available_items = self.data[~self.data[item_name].isin(user_items)].drop_duplicates(item_name)
                
                items_to_sample = min(self.list_size - 1, len(available_items))
                if items_to_sample > 0:
                    sampled_items = available_items.sample(n=items_to_sample, random_state=42)
                else:
                    sampled_items = available_items.head(items_to_sample)
        else:
            available_items = self.data.drop_duplicates(item_name)
            items_to_sample = min(self.list_size, len(available_items))
            if items_to_sample > 0:
                sampled_items = available_items.sample(n=items_to_sample, random_state=42)
            else:
                sampled_items = available_items.head(items_to_sample)
            user_items_list = []
        
        candidate_titles = sampled_items[item_name].tolist()
        
        # Extract and append last item from user_items_list as target
        if user_items_list:
            last_item = user_items_list.pop()
            candidate_titles.append(last_item)
        else:
            last_item = None
        
        # Convert to dictionary format
        candidate_list = [
            {
                'title': title,
                'original_position': i
            }
            for i, title in enumerate(candidate_titles)
        ]
        
        actual_candidate_size = len(candidate_list)
        return candidate_list, user_items_list, last_item, actual_candidate_size
    
    # Replace the method
    analyzer.create_candidate_list = create_candidate_list_fixed.__get__(analyzer, type(analyzer))
    print("✅ Patched create_candidate_list method for fixed news dataset")

# Usage:
# patch_create_candidate_list(analyzer)
'''
    
    with open('patch_create_candidate_list.py', 'w') as f:
        f.write(patch_code)
    
    print("💾 Patch file saved to: patch_create_candidate_list.py")

def test_patched_method():
    """Test the patched method."""
    print("🧪 Testing patched create_candidate_list method...")
    
    # Import and test
    test_code = '''
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
'''
    
    with open('test_patched_method.py', 'w') as f:
        f.write(test_code)
    
    print("💾 Test file saved to: test_patched_method.py")

if __name__ == "__main__":
    print("🔧 Creating fixes for create_candidate_list method")
    print("=" * 50)
    
    # Create the fixed method
    fixed_method = create_fixed_candidate_list_method()
    print("✅ Fixed method created")
    
    # Create patch file
    create_patch_for_llm_debias()
    
    # Create test file
    test_patched_method()
    
    print(f"\n✅ All fixes created!")
    print(f"\n📋 Usage:")
    print(f"  1. Import the patch: from patch_create_candidate_list import patch_create_candidate_list")
    print(f"  2. Apply to analyzer: patch_create_candidate_list(analyzer)")
    print(f"  3. Test with: python test_patched_method.py")
    print(f"  4. Re-run evaluation with fixed dataset and patched method") 