import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple

class StatisticalSignificanceAnalyzer:
    """
    Utility to evaluate the statistical significance of PROPEL's performance
    against baseline methods (Raw Output, Bootstrapping, STELLA).
    """
    def __init__(self, confidence_level: float = 0.95):
        """
        Args:
            confidence_level: The confidence level for intervals (default: 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def calculate_confidence_interval(self, data: List[float]) -> Tuple[float, float]:
        """
        Calculates the confidence interval for the mean of a list of floats.
        
        Args:
            data: List of score differences or absolute scores
            
        Returns:
            Tuple containing (lower_bound, upper_bound)
        """
        n = len(data)
        if n < 2:
            return 0.0, 0.0
            
        mean = np.mean(data)
        se = stats.sem(data)
        
        # Use t-distribution
        margin_error = se * stats.t.ppf((1 + self.confidence_level) / 2., n-1)
        
        return mean - margin_error, mean + margin_error
        
    def paired_t_test(self, our_scores: List[float], baseline_scores: List[float]) -> Dict[str, float]:
        """
        Performs a paired t-test between our method's scores and a baseline's scores.
        
        Args:
            our_scores: List of metric scores (e.g. NDCG per user) for our method
            baseline_scores: List of metric scores for the baseline method
            
        Returns:
            Dictionary containing t-statistic, p-value, mean difference, and significance boolean.
        """
        if len(our_scores) != len(baseline_scores):
            raise ValueError("Lengths of our_scores and baseline_scores must match for paired t-test.")
            
        if len(our_scores) < 2:
            raise ValueError("Need at least 2 samples to perform a t-test.")
            
        diffs = [o - b for o, b in zip(our_scores, baseline_scores)]
        mean_diff = np.mean(diffs)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(our_scores, baseline_scores)
        
        # Calculate confidence interval of the difference
        ci_lower, ci_upper = self.calculate_confidence_interval(diffs)
        
        is_significant = p_value < self.alpha
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_difference': float(mean_diff),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'is_significant': bool(is_significant)
        }
        
    def analyze_evaluation_results(self, raw_results: Dict) -> Dict[str, Dict]:
        """
        Parses the nested raw results dictionary output by LLMPositionBiasAnalyzer
        to calculate significance across configured baselines.
        
        Note: This assumes the evaluation framework has been extended to save 
        per-user metric lists (e.g. `raw_results['our_method']['ndcg_10']['per_user']`).
        If per-user lists are not available, it throws a ValueError.
        """
        
        required_keys = ['our_method', 'raw_output']
        for key in required_keys:
            if key not in raw_results:
                raise KeyError(f"Expected key '{key}' not found in results dictionary")
                
        metrics_to_test = ['ndcg_10', 'accuracy']
        baselines = ['raw_output', 'stella']
        
        significance_report = {}
        
        for metric in metrics_to_test:
            metric_report = {}
            
            try:
                our_scores = raw_results['our_method'][metric]['per_user']
            except KeyError:
                raise ValueError(
                    f"Per-user scores for {metric} not found in 'our_method'. "
                    "Ensure LLMPositionBiasAnalyzer saves the 'per_user' list for statistical testing."
                )
                
            for baseline in baselines:
                if baseline in raw_results and metric in raw_results[baseline]:
                    try:
                        base_scores = raw_results[baseline][metric].get('per_user')
                        if base_scores is None:
                            continue # Skip if per_user not saved for this baseline
                            
                        stats_result = self.paired_t_test(our_scores, base_scores)
                        metric_report[f'vs_{baseline}'] = stats_result
                        
                    except Exception as e:
                        print(f"Error calculating stats for {metric} vs {baseline}: {e}")
            
            significance_report[metric] = metric_report
            
        return significance_report
        
    def print_significance_report(self, report: Dict[str, Dict]):
        """
        Pretty prints the statistical significance report.
        """
        print("\n" + "="*50)
        print("📊 Statistical Significance Analysis (p < {:.2f})".format(self.alpha))
        print("="*50)
        
        for metric, baselines in report.items():
            print(f"\nMetric: {metric.upper()}")
            print("-" * 30)
            
            if not baselines:
                print("No baseline comparison data available.")
                continue
                
            for baseline, stats_dict in baselines.items():
                sig_str = "✅ YES" if stats_dict['is_significant'] else "❌ NO"
                
                print(f"Compared to: {baseline.replace('vs_', '').upper()}")
                print(f"  Mean Difference : {stats_dict['mean_difference']:+.4f}")
                print(f"  P-Value         : {stats_dict['p_value']:.4e} ({sig_str})")
                print(f"  {self.confidence_level*100:.0f}% CI        : "
                      f"[{stats_dict['ci_lower']:+.4f}, {stats_dict['ci_upper']:+.4f}]")
                      
        print("="*50 + "\n")


# Dummy execution block for manual testing
if __name__ == "__main__":
    analyzer = StatisticalSignificanceAnalyzer()
    
    # Generate some dummy per-user scores to test the math
    np.random.seed(42)
    dummy_our_scores = np.random.normal(loc=0.45, scale=0.1, size=100).tolist()
    dummy_raw_scores = np.random.normal(loc=0.35, scale=0.1, size=100).tolist()
    dummy_stella_scores = np.random.normal(loc=0.43, scale=0.1, size=100).tolist()
    
    dummy_results = {
        'our_method': {
            'ndcg_10': {'mean': np.mean(dummy_our_scores), 'per_user': dummy_our_scores},
            'accuracy': {'mean': np.mean(dummy_our_scores), 'per_user': dummy_our_scores}
        },
        'raw_output': {
            'ndcg_10': {'mean': np.mean(dummy_raw_scores), 'per_user': dummy_raw_scores},
            'accuracy': {'mean': np.mean(dummy_raw_scores), 'per_user': dummy_raw_scores}
        },
        'stella': {
            'ndcg_10': {'mean': np.mean(dummy_stella_scores), 'per_user': dummy_stella_scores},
            'accuracy': {'mean': np.mean(dummy_stella_scores), 'per_user': dummy_stella_scores}
        }
    }
    
    report = analyzer.analyze_evaluation_results(dummy_results)
    analyzer.print_significance_report(report)
