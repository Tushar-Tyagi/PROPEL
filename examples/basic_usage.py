#!/usr/bin/env python3
"""
Basic Usage Example for LLM Position Bias Analysis Framework

This script demonstrates how to use the framework for basic position bias analysis.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from LLM_debias import LLMPositionBiasAnalyzer
    print("✅ Successfully imported LLMPositionBiasAnalyzer")
except ImportError as e:
    print(f"❌ Error importing LLMPositionBiasAnalyzer: {e}")
    print("Please ensure you have installed the requirements: pip install -r requirements.txt")
    sys.exit(1)


def create_sample_data():
    """Create sample recommendation data for demonstration"""
    print("📊 Creating sample recommendation data...")
    
    np.random.seed(42)
    n_users = 100
    n_items = 200
    
    data = []
    for user_id in range(n_users):
        # Each user has 5-15 interactions
        n_interactions = np.random.randint(5, 16)
        items = np.random.choice(n_items, n_interactions, replace=False)
        
        for item_id in items:
            data.append({
                'UserID': f'user_{user_id:03d}',
                'Title': f'item_{item_id:03d}',
                'Rating': np.random.randint(1, 6)  # 1-5 rating
            })
    
    df = pd.DataFrame(data)
    print(f"   Created dataset with {len(df)} interactions from {n_users} users and {n_items} items")
    return df


def run_basic_analysis(data):
    """Run basic position bias analysis"""
    print("\n🔍 Running basic position bias analysis...")
    
    try:
        # Initialize the analyzer
        analyzer = LLMPositionBiasAnalyzer(
            data=data,
            data_name='books',  # Using books format
            model='gpt-3.5-turbo',
            backend='openai',
            num_bias_users=3,      # Small number for demo
            num_eval_users=10,     # Small number for demo
            num_shuffles_bias=20,  # Reduced for demo
            list_size=50,          # Smaller list for demo
            api_tier='basic'       # Conservative API usage
        )
        
        print("✅ Analyzer initialized successfully")
        print(f"   Selected {len(analyzer.bias_users)} bias users and {len(analyzer.eval_users)} evaluation users")
        
        # Note: In a real scenario, you would run the full analysis
        # For this demo, we'll just show the setup
        print("\n📋 Analysis Configuration:")
        print(f"   Model: {analyzer.model}")
        print(f"   Backend: {analyzer.backend}")
        print(f"   API Tier: {analyzer.api_tier}")
        print(f"   Rate Limits: {analyzer.api_config['rpm']} RPM, {analyzer.api_config['tpm']} TPM")
        print(f"   List Size: {analyzer.list_size}")
        print(f"   Middle Segment: {analyzer.middle_start} to {analyzer.middle_end}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None


def demonstrate_data_processing(analyzer, data):
    """Demonstrate data processing capabilities"""
    print("\n🔄 Demonstrating data processing...")
    
    try:
        # Show data structure
        print("📊 Dataset Overview:")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   User IDs: {data['UserID'].nunique()}")
        print(f"   Item IDs: {data['Title'].nunique()}")
        
        # Show sample user data
        sample_user = data['UserID'].iloc[0]
        user_items = data[data['UserID'] == sample_user]
        print(f"\n👤 Sample User '{sample_user}':")
        print(f"   Number of interactions: {len(user_items)}")
        print(f"   Average rating: {user_items['Rating'].mean():.2f}")
        
        # Show rating distribution
        print(f"\n⭐ Rating Distribution:")
        rating_counts = data['Rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"   {rating} stars: {count} ({count/len(data)*100:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error during data processing demo: {e}")


def show_next_steps():
    """Show what developers can do next"""
    print("\n🚀 Next Steps for Developers:")
    print("\n1. 🔑 Set up API keys:")
    print("   - Copy env.example to .env")
    print("   - Add your OpenAI API key")
    print("   - Ensure you have sufficient API credits")
    
    print("\n2. 📊 Prepare your dataset:")
    print("   - Ensure it has UserID, Title columns")
    print("   - Add Rating column if available")
    print("   - Follow the format shown in data/ directory")
    
    print("\n3. 🔍 Run full analysis:")
    print("   - Uncomment the analysis code in this script")
    print("   - Adjust parameters for your use case")
    print("   - Monitor API usage and costs")
    
    print("\n4. 📈 Explore results:")
    print("   - Check the generated checkpoint files")
    print("   - Analyze bias scores and propensity metrics")
    print("   - Compare original vs debiased rankings")
    
    print("\n5. 🧪 Run tests:")
    print("   - Execute: python -m pytest tests/")
    print("   - Check test coverage and fix any issues")
    
    print("\n6. 📚 Study examples:")
    print("   - Review experiment_*.ipynb notebooks")
    print("   - Examine evaluation_*.ipynb for workflows")
    print("   - Check checkpoint_*.ipynb for state management")


def main():
    """Main demonstration function"""
    print("🎯 LLM Position Bias Analysis Framework - Basic Usage Demo")
    print("=" * 60)
    
    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set")
        print("   Set it to run actual LLM analysis")
    
    # Create sample data
    data = create_sample_data()
    
    # Run basic analysis setup
    analyzer = run_basic_analysis(data)
    
    if analyzer:
        # Demonstrate data processing
        demonstrate_data_processing(analyzer, data)
        
        # Show what to do next
        show_next_steps()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("   The framework is ready for your position bias analysis!")
        
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
        print("   Ensure all dependencies are installed and configured correctly.")


if __name__ == "__main__":
    main()
