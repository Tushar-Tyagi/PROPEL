"""
PROPEL Offline Bias Probing Example.

Demonstrates the offline Probing Stage (Section 4.1 & Appendix B) where candidate
lists are randomly shuffled to profile and estimate position bias coefficients
(Primacy B_prim, Recency B_rec, Middle B_mid) on a new LLM model or prompt template.
"""

import os
import sys
import random

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from propel import BiasProfiler, PropensityModel

def main():
    print("🔬 PROPEL Offline Probing Demo")
    print("================================")

    N_candidates = 20
    top_k = 10
    profiler = BiasProfiler(N=N_candidates, top_k=top_k)

    print(f"Simulating probing trials on N={N_candidates} candidates with simulated primacy & recency bias...")

    # Simulate 50 users x 20 shuffles = 1000 trials
    # Ground-truth: simulate an LLM that selects items near top (primacy) 65% of the time,
    # middle 25%, and bottom (recency) 10%
    for _ in range(1000):
        candidates = [f"Item_{i}" for i in range(1, N_candidates + 1)]
        prompt_order = list(candidates)
        random.shuffle(prompt_order)

        # Simulate biased LLM response: strongly favors top prompt positions
        # e.g., picks mostly items from top 5 and a few from rest
        ranked_items = (
            prompt_order[:5]
            + prompt_order[5:15][:3]
            + prompt_order[15:][:2]
            + prompt_order[8:15][3:]
            + prompt_order[17:]
        )
        # Record trial
        profiler.record_trial(prompt_candidate_order=prompt_order, llm_ranked_items=ranked_items)

    # Estimate normalized bias coefficients
    b_prim, b_rec, b_mid = profiler.estimate_bias_coefficients()
    print("\n📈 Estimated Bias Coefficients:")
    print(f"   Primacy Bias (B_prim) : {b_prim:+.3f}")
    print(f"   Recency Bias (B_rec)  : {b_rec:+.3f}")
    print(f"   Middle Bias  (B_mid)  : {b_mid:+.3f}")

    # Generate Propensity Model
    model = profiler.create_propensity_model()
    weights = model.get_inverse_propensity_weights(N_candidates)

    print("\n⚖️ Resulting Inverse Propensity Weights w(p):")
    print(f"   Position 1  (Top):    w(1)  = {weights[1]:.3f} (down-weighted)")
    print(f"   Position 10 (Middle): w(10) = {weights[10]:.3f}")
    print(f"   Position 20 (Bottom): w(20) = {weights[20]:.3f} (up-weighted)")


if __name__ == "__main__":
    main()
