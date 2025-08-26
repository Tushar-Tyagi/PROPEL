"""
Basic functionality tests for LLM Position Bias Analysis Framework
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from LLM_debias import LLMPositionBiasAnalyzer, get_data_columns
except ImportError as e:
    pytest.skip(f"Could not import LLM_debias: {e}", allow_module_level=True)


class TestDataColumns:
    """Test the get_data_columns function"""
    
    def test_movie_lens_columns(self):
        """Test MovieLens dataset column configuration"""
        item_name, item_metadata, user_metadata, user_rating = get_data_columns('movie_lens')
        
        assert item_name == 'Title'
        assert 'Genres' in item_metadata
        assert 'Gender' in user_metadata
        assert 'Rating' in user_rating
    
    def test_books_columns(self):
        """Test Books dataset column configuration"""
        item_name, item_metadata, user_metadata, user_rating = get_data_columns('books')
        
        assert item_name == 'Title'
        assert item_metadata == []
        assert user_metadata == []
        assert user_rating == []
    
    def test_unknown_dataset(self):
        """Test handling of unknown dataset names"""
        with pytest.raises(UnboundLocalError):
            get_data_columns('unknown_dataset')


class TestLLMPositionBiasAnalyzer:
    """Test the main analyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_users = 50
        n_items = 100
        
        data = []
        for user_id in range(n_users):
            n_interactions = np.random.randint(5, 20)
            items = np.random.choice(n_items, n_interactions, replace=False)
            for item_id in items:
                data.append({
                    'UserID': f'user_{user_id}',
                    'Title': f'item_{item_id}',
                    'Rating': np.random.randint(1, 6)
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_analyzer(self, sample_data):
        """Create a mock analyzer instance"""
        with patch('LLM_debias.LLMPositionBiasAnalyzer._filter_users_with_sufficient_data'):
            analyzer = LLMPositionBiasAnalyzer(
                data=sample_data,
                data_name='books',
                model='gpt-3.5-turbo',
                backend='openai',
                num_bias_users=2,
                num_eval_users=5,
                num_shuffles_bias=10,
                list_size=20,
                api_tier='basic'
            )
            # Mock the user selection
            analyzer.bias_users = ['user_0', 'user_1']
            analyzer.eval_users = ['user_2', 'user_3', 'user_4', 'user_5', 'user_6']
            analyzer.num_bias_users = 2
            analyzer.num_eval_users = 5
            return analyzer
    
    def test_analyzer_initialization(self, mock_analyzer):
        """Test analyzer initialization"""
        assert mock_analyzer.data_name == 'books'
        assert mock_analyzer.model == 'gpt-3.5-turbo'
        assert mock_analyzer.backend == 'openai'
        assert mock_analyzer.list_size == 20
        assert mock_analyzer.num_shuffles == 10
    
    def test_api_config_loading(self, mock_analyzer):
        """Test API configuration loading"""
        assert 'rpm' in mock_analyzer.api_config
        assert 'tpm' in mock_analyzer.api_config
        assert 'max_workers' in mock_analyzer.api_config
    
    def test_middle_segment_calculation(self, mock_analyzer):
        """Test middle segment calculation"""
        # For list_size=20: middle_start=5, middle_end=15
        assert mock_analyzer.middle_start == 5
        assert mock_analyzer.middle_end == 15
    
    @patch('LLM_debias.LLMPositionBiasAnalyzer._filter_users_with_sufficient_data')
    def test_user_filtering(self, mock_filter, sample_data):
        """Test user filtering logic"""
        mock_filter.return_value = ['user_0', 'user_1', 'user_2']
        
        analyzer = LLMPositionBiasAnalyzer(
            data=sample_data,
            data_name='books',
            model='gpt-3.5-turbo',
            backend='openai',
            num_bias_users=1,
            num_eval_users=2
        )
        
        # Should adjust numbers when not enough users
        assert analyzer.num_bias_users == 1
        assert analyzer.num_eval_users == 2


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_api_config_tiers(self):
        """Test API configuration tier system"""
        from LLM_debias import get_api_config
        
        # Test basic tier
        basic_config = get_api_config('gpt-3.5-turbo', 'basic')
        assert basic_config['rpm'] == 500
        assert basic_config['tpm'] == 200000
        
        # Test tier_1
        tier1_config = get_api_config('gpt-3.5-turbo', 'tier_1')
        assert tier1_config['rpm'] == 3500
        assert tier1_config['tpm'] == 1000000
        
        # Test unknown model (should return default)
        default_config = get_api_config('unknown_model', 'tier_1')
        assert default_config['rpm'] == 60
        assert default_config['tpm'] == 10000


class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):
            LLMPositionBiasAnalyzer(
                data=empty_df,
                data_name='books',
                model='gpt-3.5-turbo',
                backend='openai'
            )
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        invalid_df = pd.DataFrame({
            'UserID': ['user1', 'user2'],
            'InvalidColumn': ['item1', 'item2']
        })
        
        with pytest.raises(Exception):
            LLMPositionBiasAnalyzer(
                data=invalid_df,
                data_name='books',
                model='gpt-3.5-turbo',
                backend='openai'
            )


if __name__ == "__main__":
    # Run tests if file is executed directly
    pytest.main([__file__, "-v"])
