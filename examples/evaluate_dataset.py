"""
PROPEL Dataset Evaluation Example.

Demonstrates loading a standardized benchmark cohort (e.g. MovieLens-1M) and evaluating
PROPEL performance metrics (Hit@1, NDCG@5, NDCG@20, MRR).
"""

import os
import sys

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from propel import PROPEL, LLMClient, load_dataset_cohort

def main():
    print("📊 PROPEL Benchmark Dataset Evaluation")
    print("=======================================")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    dataset_name = "ml-1m"

    try:
        profiles = load_dataset_cohort(data_dir=data_dir, dataset_name=dataset_name, cohort="test")
        print(f"✅ Loaded {len(profiles)} test profiles from {dataset_name}.")
    except Exception as e:
        print(f"⚠️ Could not load dataset cohort: {e}")
        return

    # Use first 5 profiles for quick demo
    eval_profiles = profiles[:5]
    print(f"Evaluating {len(eval_profiles)} profiles with demo LLM client...")

    # Mock client if no API key
    if "OPENAI_API_KEY" not in os.environ:
        def mock_llm(prompt: str) -> str:
            # Deterministic mock response
            return '{"ranked_movies": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}'
        client = LLMClient(custom_caller=mock_llm)
        reranker = PROPEL(model="gpt-4o-mini", dataset=dataset_name, llm_client=client)
    else:
        reranker = PROPEL(model="gpt-4o-mini", dataset=dataset_name)

    results = reranker.evaluate_profiles(eval_profiles, num_shuffles=5)

    print("\n🏆 Evaluation Results:")
    print(f"   Hit@1   : {results['hit@1']:.4f}")
    print(f"   NDCG@5  : {results['ndcg@5']:.4f}")
    print(f"   NDCG@20 : {results['ndcg@20']:.4f}")
    print(f"   MRR     : {results['mrr']:.4f}")


if __name__ == "__main__":
    main()
