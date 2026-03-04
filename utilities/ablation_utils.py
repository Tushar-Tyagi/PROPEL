import time
import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from LLM_debias import LLMPositionBiasAnalyzer

class AblationAnalyzer(LLMPositionBiasAnalyzer):
    def _recompute_user_results(
        self,
        user_result: Dict,
        new_propensity_scores: Dict[int, float],
        aggregation_method: str
    ) -> Dict:
        """Recompute a single user's results with new propensity scores."""

        raw_llm_data = user_result.get('raw_llm_data', [])
        target_item = user_result.get('target_item')

        if not raw_llm_data or not target_item:
            return None

        # Reprocess each trial with new propensity scores
        title_scores_across_trials = defaultdict(list)
        title_debiased_scores_across_trials = defaultdict(list)
        title_weights_across_trials = defaultdict(list)

        for trial_data in raw_llm_data:
            llm_reranked_list = trial_data.get('llm_reranked_list', [])

            for item in llm_reranked_list:
                title = item.get('title')
                llm_score = item.get('llm_score', 0.0)
                trial_position = item.get('trial_position', 0)

                # Apply new propensity weight
                weight = new_propensity_scores.get(trial_position, 1.0)
                debiased_score = llm_score * weight

                # Track scores
                title_scores_across_trials[title].append(llm_score)
                title_debiased_scores_across_trials[title].append(debiased_score)
                title_weights_across_trials[title].append(weight)

        # Aggregate scores
        aggregated_scores = {}
        aggregated_debiased_scores = {}

        for title in title_scores_across_trials:
            scores = title_scores_across_trials[title]
            debiased_scores = title_debiased_scores_across_trials[title]

            if aggregation_method == "mean":
                aggregated_scores[title] = np.mean(scores)
                aggregated_debiased_scores[title] = np.mean(debiased_scores)
            elif aggregation_method == "median":
                aggregated_scores[title] = np.median(scores)
                aggregated_debiased_scores[title] = np.median(debiased_scores)
            elif aggregation_method == "max":
                aggregated_scores[title] = np.max(scores)
                aggregated_debiased_scores[title] = np.max(debiased_scores)

        # Create new ranking based on debiased scores
        if aggregated_debiased_scores:
            sorted_titles = sorted(aggregated_debiased_scores.keys(),
                                 key=lambda x: aggregated_debiased_scores[x], reverse=True)

            final_ranking = []
            for i, title in enumerate(sorted_titles):
                final_ranking.append({
                    'title': title,
                    'final_rank': i + 1,
                    'aggregated_score': aggregated_scores.get(title, 0.0),
                    'aggregated_debiased_score': aggregated_debiased_scores[title]
                })
        else:
            final_ranking = []

        # Calculate metrics
        ranked_titles = [item['title'] for item in final_ranking]
        accuracy = 1.0 if ranked_titles and ranked_titles[0] == target_item else 0.0
        ndcg_1 = self._calculate_ndcg(target_item, ranked_titles, 1)
        ndcg_5 = self._calculate_ndcg(target_item, ranked_titles, 5)
        ndcg_10 = self._calculate_ndcg(target_item, ranked_titles, 10)
        ndcg_20 = self._calculate_ndcg(target_item, ranked_titles, 20)

        return {
            'user_id': user_result.get('user_id'),
            'target_item': target_item,
            'accuracy': accuracy,
            'ndcg_1': ndcg_1,
            'ndcg_5': ndcg_5,
            'ndcg_10': ndcg_10,
            'ndcg_20': ndcg_20,
            'final_ranking': final_ranking,
            'aggregated_scores': aggregated_scores,
            'aggregated_debiased_scores': aggregated_debiased_scores
        }

    def create_custom_propensity_scores(
        self,
        N: int,
        formula: str = "inverse",
        primacy_bias: float = None,
        recency_bias: float = None,
        middle_bias: float = None,
        custom_function: callable = None
    ) -> Dict[int, float]:
        """
        Create custom propensity scores using different formulas.
        """
        propensity_scores = {}

        if formula == "inverse" and primacy_bias is not None:
            for pos in range(N):
                if pos < int(0.25 * N):
                    weight = 1.0 / max(primacy_bias, 0.1)
                elif pos >= int(0.75 * N):
                    weight = 1.0 / max(recency_bias, 0.1)
                else:
                    weight = 1.0 / max(middle_bias, 0.1)
                propensity_scores[pos] = weight

        elif formula == "linear":
            for pos in range(N):
                weight = 1.0 - (pos / (N - 1)) * 0.5
                propensity_scores[pos] = weight

        elif formula == "exponential":
            for pos in range(N):
                weight = np.exp(-pos / 10.0)
                propensity_scores[pos] = weight

        elif formula == "uniform":
            for pos in range(N):
                propensity_scores[pos] = 1.0

        elif formula == "custom" and custom_function:
            for pos in range(N):
                weight = custom_function(pos, N)
                propensity_scores[pos] = weight

        else:
            for pos in range(N):
                propensity_scores[pos] = 1.0

        return propensity_scores

    def reapply_debiasing_with_new_bias(
        self,
        checkpoint_file: str,
        new_precalculated_bias: Dict[str, float],
        aggregation_method: str = "mean",
        save_results_to: str = None
    ) -> Dict:
        """
        Reapply debiasing using raw LLM outputs from checkpoint with new bias scores.
        """
        print(f"🔄 REAPPLYING DEBIASING WITH NEW BIAS SCORES")
        print(f"📁 Loading: {checkpoint_file}")
        print(f"🧠 New bias - Primacy: {new_precalculated_bias.get('avg_primacy', 'N/A')}, "
              f"Recency: {new_precalculated_bias.get('avg_recency', 'N/A')}, "
              f"Middle: {new_precalculated_bias.get('avg_middle', 'N/A')}")
        print("=" * 50)

        # Load checkpoint data
        checkpoint_data = self._load_checkpoint(checkpoint_file)
        if not checkpoint_data:
            return {'error': 'No checkpoint found'}

        all_user_results = checkpoint_data.get('all_user_results', [])
        if not all_user_results:
            return {'error': 'No user results found in checkpoint'}

        original_bias_analysis = checkpoint_data.get('bias_analysis', {})
        method_config = checkpoint_data.get('method_config', {})
        num_candidates = method_config.get('num_candidates', 20)

        print(f"👥 Found {len(all_user_results)} users with raw LLM data")
        print(f"🎯 Using {num_candidates} candidates for propensity calculation")

        print(f"\n🧮 Calculating new propensity scores from bias...")
        new_propensity_scores = self.calculate_propensity_scores(num_candidates, new_precalculated_bias)

        print(f"✅ Generated propensity scores for positions 1-{num_candidates}")
        print(f"   Sample weights: Pos 1: {new_propensity_scores.get(0, 'N/A'):.3f}, "
              f"Pos 10: {new_propensity_scores.get(9, 'N/A'):.3f}, "
              f"Pos 20: {new_propensity_scores.get(19, 'N/A'):.3f}")

        print(f"\n🔄 Recomputing user results with new propensity scores...")
        recomputed_results = []
        users_processed = 0

        for user_result in tqdm(all_user_results, desc="Recomputing", ncols=80):
            try:
                user_id = user_result.get('user_id')
                target_item = user_result.get('target_item')
                raw_llm_data = user_result.get('raw_llm_data', [])

                if not raw_llm_data:
                    print(f"⚠️ No raw LLM data for user {user_id}, skipping")
                    continue

                recomputed_user_result = self._recompute_user_results(
                    user_result, new_propensity_scores, aggregation_method
                )

                if recomputed_user_result:
                    recomputed_results.append(recomputed_user_result)
                    users_processed += 1

            except Exception as e:
                print(f"Error recomputing user {user_result.get('user_id', 'Unknown')}: {e}")
                continue

        print(f"✅ Successfully recomputed {users_processed} users")

        if not recomputed_results:
            return {'error': 'No results could be recomputed'}

        accuracies = [r['accuracy'] for r in recomputed_results]
        ndcg_1s = [r['ndcg_1'] for r in recomputed_results]
        ndcg_5s = [r['ndcg_5'] for r in recomputed_results]
        ndcg_10s = [r['ndcg_10'] for r in recomputed_results]
        ndcg_20s = [r['ndcg_20'] for r in recomputed_results]

        recomputed_evaluation = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'num_evaluations': len(accuracies)
            },
            'ndcg_1': {
                'mean': np.mean(ndcg_1s),
                'std': np.std(ndcg_1s),
                'num_evaluations': len(ndcg_1s)
            },
            'ndcg_5': {
                'mean': np.mean(ndcg_5s),
                'std': np.std(ndcg_5s),
                'num_evaluations': len(ndcg_5s)
            },
            'ndcg_10': {
                'mean': np.mean(ndcg_10s),
                'std': np.std(ndcg_10s),
                'num_evaluations': len(ndcg_10s)
            },
            'ndcg_20': {
                'mean': np.mean(ndcg_20s),
                'std': np.std(ndcg_20s),
                'num_evaluations': len(ndcg_20s)
            }
        }

        print(f"\n📊 RECOMPUTED RESULTS WITH NEW BIAS SCORES:")
        print(f"  Accuracy:    {recomputed_evaluation['accuracy']['mean']:.4f} ± {recomputed_evaluation['accuracy']['std']:.4f}")
        print(f"  NDCG@1:      {recomputed_evaluation['ndcg_1']['mean']:.4f} ± {recomputed_evaluation['ndcg_1']['std']:.4f}")
        print(f"  NDCG@5:      {recomputed_evaluation['ndcg_5']['mean']:.4f} ± {recomputed_evaluation['ndcg_5']['std']:.4f}")
        print(f"  NDCG@10:     {recomputed_evaluation['ndcg_10']['mean']:.4f} ± {recomputed_evaluation['ndcg_10']['std']:.4f}")
        print(f"  NDCG@20:     {recomputed_evaluation['ndcg_20']['mean']:.4f} ± {recomputed_evaluation['ndcg_20']['std']:.4f}")
        print(f"  Number of evaluations: {len(accuracies)}")

        try:
            if 'our_method_evaluation' in checkpoint_data:
                original_eval = checkpoint_data['our_method_evaluation']
                orig_acc = original_eval.get('accuracy', {}).get('mean', 0)
                new_acc = recomputed_evaluation['accuracy']['mean']
                diff_acc = new_acc - orig_acc

                print(f"\n📈 COMPARISON WITH ORIGINAL:")
                print(f"  Original accuracy: {orig_acc:.4f}")
                print(f"  New accuracy:      {new_acc:.4f}")
                print(f"  Difference:        {diff_acc:+.4f}")
        except Exception:
            print(f"\n⚠️ Could not compare with original results")

        new_bias_analysis = {
            'bias_scores': new_precalculated_bias,
            'propensity_scores': new_propensity_scores,
            'avg_bias_result': new_precalculated_bias,
            'precalculated_bias_used': True,
            'recomputed_from_checkpoint': True
        }

        final_results = {
            'recomputed_evaluation': recomputed_evaluation,
            'new_bias_analysis': new_bias_analysis,
            'recomputed_user_results': recomputed_results,
            'original_bias_analysis': original_bias_analysis,
            'original_evaluation': checkpoint_data.get('our_method_evaluation', {}),
            'recomputation_info': {
                'source_checkpoint': checkpoint_file,
                'users_processed': users_processed,
                'aggregation_method': aggregation_method,
                'num_candidates': num_candidates,
                'recomputation_timestamp': time.time(),
                'new_bias_scores': new_precalculated_bias
            }
        }

        if save_results_to:
            self._save_checkpoint(final_results, save_results_to)
            print(f"💾 Recomputed results saved to: {save_results_to}")

        print(f"\n✅ Recomputation complete!")
        return final_results

    def verify_bias_users_have_candidates(self, min_candidates: int = 10) -> bool:
        """
        Verify that all selected bias users have sufficient candidates for bias detection.
        """
        print(f"🔍 Verifying bias users have sufficient candidates (min: {min_candidates})...")

        insufficient_users = []

        for user_id in self.bias_users:
            try:
                candidate_list, user_items, last_item, actual_size = self.create_candidate_list(user_id)

                if actual_size < min_candidates:
                    insufficient_users.append((user_id, actual_size))
                    print(f"⚠️  User {user_id}: Only {actual_size} candidates (need {min_candidates})")
                else:
                    print(f"✅ User {user_id}: {actual_size} candidates")

            except Exception as e:
                insufficient_users.append((user_id, 0))
                print(f"❌ User {user_id}: Error creating candidates - {e}")

        if insufficient_users:
            print(f"\n⚠️  {len(insufficient_users)} bias users have insufficient candidates:")
            for user_id, size in insufficient_users:
                print(f"  - {user_id}: {size} candidates")

            self.bias_users = [u for u in self.bias_users if u not in [user_id for user_id, _ in insufficient_users]]
            print(f"\n🔄 Filtered bias users to {len(self.bias_users)} users with sufficient candidates")

            if len(self.bias_users) == 0:
                print("❌ No bias users have sufficient candidates!")
                return False

            print(f"✅ Remaining bias users: {self.bias_users}")

        return True
