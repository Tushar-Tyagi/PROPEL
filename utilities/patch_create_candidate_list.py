
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
