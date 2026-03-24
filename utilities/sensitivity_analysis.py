import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
import itertools
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the parent directory to the Python path to allow importing LLM_debias
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from LLM_debias import LLMPositionBiasAnalyzer
except ImportError:
    print("Warning: Could not import LLMPositionBiasAnalyzer. Ensure LLM_debias.py is in the parent directory.")

class SensitivityAnalyzer:
    """
    Utility for performing Sensitivity Analysis and Hyperparameter Optimization (HPO)
    on the PROPEL position bias elimination framework.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        data_name: str,
        base_model: str = 'gpt-3.5-turbo',
        base_backend: str = 'openai',
        api_tier: str = 'basic',
        output_dir: str = 'results/sensitivity_analysis'
    ):
        self.data = data
        self.data_name = data_name
        self.base_model = base_model
        self.base_backend = base_backend
        self.api_tier = api_tier
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_parameter_grid(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generates all combinations of parameters from a grid dictionary.
        
        Args:
            param_grid: A dictionary where keys are parameter names and values are lists of parameter values.
            
        Returns:
            A list of dictionaries, where each dictionary is a specific parameter combination.
        """
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = list(itertools.product(*values))
        
        param_list = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            param_list.append(params)
            
        return param_list
        
    def run_sensitivity_analysis(
        self,
        param_grid: Dict[str, List[Any]],
        fixed_params: Optional[Dict[str, Any]] = None,
        experiment_name: str = "sensitivity_study"
    ) -> pd.DataFrame:
        """
        Runs the LLMPositionBiasAnalyzer over a grid of parameters to test sensitivity.
        
        Args:
            param_grid: A dictionary of parameters to test (e.g. {'num_bias_users': [5, 10, 20]})
            fixed_params: Parameters to keep constant across all runs
            experiment_name: Base name for saving results
            
        Returns:
            DataFrame containing the results across all parameter combinations
        """
        if fixed_params is None:
            fixed_params = {
                'num_bias_users': 5,
                'num_eval_users': 20,
                'num_shuffles_bias': 20,
                'num_candidates': 20
            }
            
        param_combinations = self.generate_parameter_grid(param_grid)
        total_runs = len(param_combinations)
        
        print(f"\n🚀 Starting Sensitivity Analysis: {experiment_name}")
        print(f"Total parameter combinations to test: {total_runs}")
        print("-" * 50)
        
        results_list = []
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, config in enumerate(param_combinations):
            print(f"\n▶️ Run {i+1}/{total_runs}")
            print(f"Testing configuration: {config}")
            
            # Merge fixed params and current config
            current_params = fixed_params.copy()
            current_params.update(config)
            
            # Set default values for params that might not be in config or fixed_params
            num_bias_users = current_params.get('num_bias_users', 5)
            num_eval_users = current_params.get('num_eval_users', 20)
            num_shuffles_bias = current_params.get('num_shuffles_bias', 20)
            num_candidates = current_params.get('num_candidates', 20)
            temperature = current_params.get('temperature', 0.0)
            
            try:
                # Initialize Analyzer
                analyzer = LLMPositionBiasAnalyzer(
                    data=self.data,
                    data_name=self.data_name,
                    model=self.base_model,
                    backend=self.base_backend,
                    num_bias_users=num_bias_users,
                    num_eval_users=num_eval_users,
                    num_shuffles_bias=num_shuffles_bias,
                    list_size=num_candidates,      # Note: uses list_size locally
                    api_tier=self.api_tier
                )
                
                # We can inject temperature if modifying the analyzer slightly, 
                # but currently LLMPositionBiasAnalyzer uses default temp in standard setup.
                # If temperature was a required sensitivity param, we would monkeypatch it
                # or pass it if the API allowed.
                
                # Only testing evaluation bias to see the NDCG tradeoff
                print("Running bias analysis...")
                try:
                    results = analyzer.evaluate_our_method(
                        num_bias_users=num_bias_users,
                        num_eval_users=num_eval_users,
                        num_candidates=num_candidates,
                        num_trials=num_shuffles_bias,
                        aggregation_method='mean'
                    )
                    
                    # Extract metrics
                    if 'our_method' in results and 'stella' in results:
                        our_ndcg = results['our_method']['ndcg_10']['mean']
                        our_acc = results['our_method']['accuracy']['mean']
                        
                        raw_ndcg = results['raw_output']['ndcg_10']['mean']
                        raw_acc = results['raw_output']['accuracy']['mean']
                        
                        stella_ndcg = results['stella']['ndcg_10']['mean']
                        stella_acc = results['stella']['accuracy']['mean']
                        
                        bias_stats = results['bias_analysis']
                        primacy = bias_stats.get('avg_primacy_bias', 0.0)
                        recency = bias_stats.get('avg_recency_bias', 0.0)
                        
                        run_result = {
                            **config, # Include the tested parameters
                            'our_ndcg_10': our_ndcg,
                            'our_accuracy': our_acc,
                            'raw_ndcg_10': raw_ndcg,
                            'raw_accuracy': raw_acc,
                            'stella_ndcg_10': stella_ndcg,
                            'stella_accuracy': stella_acc,
                            'primacy_bias': primacy,
                            'recency_bias': recency,
                            'status': 'success'
                        }
                    else:
                        print("Warning: Expected keys not found in results.")
                        run_result = {**config, 'status': 'failed_processing_results'}
                        
                except Exception as e:
                    print(f"Error during evaluation loop: {e}")
                    run_result = {**config, 'status': f'failed: {str(e)}'}
                
            except Exception as e:
                print(f"Failed to initialize or run analyzer for config: {config}")
                print(f"Error: {e}")
                run_result = {
                    **config,
                    'status': f'failed_initialization: {str(e)}'
                }
                
            results_list.append(run_result)
            
            # Save intermediate results
            intermediate_df = pd.DataFrame(results_list)
            intermediate_file = os.path.join(self.output_dir, f"{experiment_name}_intermediate_{timestamp}.csv")
            intermediate_df.to_csv(intermediate_file, index=False)
            
        # Final DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Save final results
        final_file_csv = os.path.join(self.output_dir, f"{experiment_name}_final_{timestamp}.csv")
        final_file_json = os.path.join(self.output_dir, f"{experiment_name}_final_{timestamp}.json")
        
        results_df.to_csv(final_file_csv, index=False)
        with open(final_file_json, 'w') as f:
            json.dump(results_list, f, indent=4)
            
        print(f"\n✅ Sensitivity Analysis Complete!")
        print(f"Results saved to: {final_file_csv}")
        
        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sensitivity Analysis on PROPEL Framework")
    parser.add_argument("--data_name", type=str, default="movie_lens", help="Dataset name to use")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use")
    parser.add_argument("--experiment_name", type=str, default="sensitivity_test", help="Name for the experiment run")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}")
    try:
        data = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        sys.exit(1)
        
    analyzer = SensitivityAnalyzer(
        data=data,
        data_name=args.data_name,
        base_model=args.model
    )
    
    # Example parameter grid:
    # Testing how the number of bias users and number of shuffles affect NDCG
    param_grid = {
        'num_bias_users': [5, 10], 
        'num_shuffles_bias': [10, 20] 
    }
    
    fixed_params = {
        'num_eval_users': 10,  # Keeping small for testing
        'num_candidates': 20
    }
    
    results = analyzer.run_sensitivity_analysis(
        param_grid=param_grid,
        fixed_params=fixed_params,
        experiment_name=args.experiment_name
    )
    
    print("\nSummary of Results:")
    if not results.empty:
        print(results.to_markdown())
    else:
        print("No successful runs to display.")
