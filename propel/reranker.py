"""
PROPEL Reranker and Pipeline Orchestrator.

Main high-level interface combining offline bias profiling, randomized candidate
generation, parallel LLM reranking, two-phase consensus aggregation, and audit logging.
"""

import os
import random
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union

from propel.defaults import DEFAULT_BIAS_PARAMS
from propel.propensity import PropensityModel
from propel.profiler import BiasProfiler
from propel.aggregation import ConsensusAggregator
from propel.llm import LLMClient, build_ranking_prompt, parse_ranking_response
from propel.explainability import ExplainabilityReport, generate_explainability_report
from propel.metrics import compute_ndcg, compute_hit, compute_mrr
from propel.data import UserProfile


class PropelReranker:
    """
    Main PROPEL Reranking Engine.
    
    Parameters
    ----------
    model : str
        LLM model identifier (e.g., 'gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo').
    dataset : str
        Dataset/domain identifier ('ml-1m', 'books', 'beauty', 'music', 'steam', 'custom').
    bias_params : Dict[str, float], optional
        Explicit bias parameters {"B_prim": float, "B_rec": float, "B_mid": float}.
        If not provided, uses pre-calibrated parameters from the PROPEL paper.
    num_shuffles : int
        Default number of shuffles N_S (default 20).
    backend : str
        LLM backend provider ('openai', 'anthropic', or custom).
    llm_client : LLMClient, optional
        Pre-configured LLM client instance.
    api_key : str, optional
        API key for LLM provider.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        dataset: str = "ml-1m",
        bias_params: Optional[Dict[str, float]] = None,
        num_shuffles: int = 20,
        backend: str = "openai",
        llm_client: Optional[LLMClient] = None,
        api_key: Optional[str] = None,
        clip_val: Optional[float] = None,
    ):
        self.model = model
        self.dataset = dataset.lower()
        self.num_shuffles = num_shuffles
        self.clip_val = clip_val

        # Setup LLM Client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient(model=model, backend=backend, api_key=api_key)

        # Setup Propensity Model
        if bias_params is not None:
            self.propensity_model = PropensityModel.from_bias_dict(bias_params)
        else:
            # Lookup default pre-calibrated parameters
            model_params = DEFAULT_BIAS_PARAMS.get(self.model, DEFAULT_BIAS_PARAMS["gpt-4o-mini"])
            ds_params = model_params.get(self.dataset, {"B_prim": 0.0, "B_rec": 0.0, "B_mid": 0.0})
            self.propensity_model = PropensityModel.from_bias_dict(ds_params)

        self.aggregator = ConsensusAggregator(clip_val=self.clip_val, auto_clip=(self.clip_val is None))

    def rerank(
        self,
        user_history: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        num_shuffles: Optional[int] = None,
        seed: Optional[int] = None,
        recommendation_id: Optional[str] = None,
    ) -> Tuple[List[str], ExplainabilityReport]:
        """
        Execute PROPEL de-biased reranking on candidate items.
        
        Parameters
        ----------
        user_history : List[Dict[str, Any]]
            User interaction history records.
        candidates : List[Dict[str, Any]]
            Candidate items to rerank.
        num_shuffles : int, optional
            Number of candidate list permutations N_S (default self.num_shuffles).
        seed : int, optional
            Random seed for permutation reproducibility.
        recommendation_id : str, optional
            Unique ID for audit tracking.
            
        Returns
        -------
        final_ranking : List[str]
            De-biased consensus ranked list of candidate titles.
        report : ExplainabilityReport
            Auditable explainability report and item adjustment table.
        """
        N_s = num_shuffles or self.num_shuffles
        N_cands = len(candidates)
        if N_cands == 0:
            raise ValueError("Candidate list cannot be empty.")

        rec_id = recommendation_id or f"rec_{uuid.uuid4().hex[:10]}"
        rng = random.Random(seed) if seed is not None else random.Random()

        # 1. Generate N_S random permutations of candidate list
        shuffled_candidate_orders: List[List[Dict[str, Any]]] = []
        original_candidate_titles: List[List[str]] = []
        prompts: List[str] = []

        for _ in range(N_s):
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            shuffled_candidate_orders.append(shuffled)
            original_candidate_titles.append([c.get("title", c.get("Title", str(c))) for c in shuffled])
            prompt = build_ranking_prompt(
                dataset=self.dataset,
                user_history=user_history,
                candidates=shuffled,
            )
            prompts.append(prompt)

        # 2. Parallel LLM execution
        responses = self.llm_client.call_parallel(prompts)

        # 3. Parse LLM rankings
        rankings: List[List[str]] = []
        for resp, shuf_cands in zip(responses, shuffled_candidate_orders):
            ranked = parse_ranking_response(resp, shuf_cands)
            rankings.append(ranked)

        # 4. Compute propensity weights w(p) = 1 / (N * P(p))
        propensity_weights = self.propensity_model.get_inverse_propensity_weights(N_cands)
        propensity_curve = self.propensity_model.get_propensity_curve_data(N_cands)

        # 5. Two-Phase Consensus Aggregation
        final_consensus, agg_metadata = self.aggregator.aggregate(
            rankings=rankings,
            original_candidate_orders=original_candidate_titles,
            propensity_weights=propensity_weights,
        )

        # 6. Generate Explainability Report
        report = generate_explainability_report(
            recommendation_id=rec_id,
            dataset=self.dataset,
            model=self.model,
            bias_coefficients=self.propensity_model.to_dict(),
            propensity_curve=propensity_curve,
            rankings=rankings,
            original_candidate_orders=original_candidate_titles,
            propensity_weights=propensity_weights,
            initial_borda_ranking=agg_metadata.get("initial_borda_ranking", []),
            final_consensus_ranking=final_consensus,
            aggregation_metadata=agg_metadata,
        )

        return final_consensus, report

    def probe(
        self,
        probe_profiles: List[UserProfile],
        num_shuffles: int = 20,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Execute offline bias probing stage (Section 4.1 & Appendix B).
        
        Parameters
        ----------
        probe_profiles : List[UserProfile]
            List of probe user profiles (e.g. N_b = 50).
        num_shuffles : int
            Number of shuffles per user S (default 20).
        top_k : int
            Top-K items considered per trial.
            
        Returns
        -------
        bias_params : Dict[str, float]
            Calibrated bias coefficients {"B_prim": float, "B_rec": float, "B_mid": float}.
        """
        if not probe_profiles:
            raise ValueError("Probe profiles list cannot be empty.")

        N_cands = len(probe_profiles[0].candidates)
        profiler = BiasProfiler(N=N_cands, top_k=top_k)

        for profile in probe_profiles:
            shuffled_cands = []
            prompts = []
            cand_orders = []

            for _ in range(num_shuffles):
                shuf = list(profile.candidates)
                random.shuffle(shuf)
                shuffled_cands.append(shuf)
                cand_orders.append([c.get("title", c.get("Title", str(c))) for c in shuf])
                prompts.append(build_ranking_prompt(self.dataset, profile.history, shuf))

            responses = self.llm_client.call_parallel(prompts)
            for resp, shuf, orig_order in zip(responses, shuffled_cands, cand_orders):
                ranked = parse_ranking_response(resp, shuf)
                profiler.record_trial(prompt_candidate_order=orig_order, llm_ranked_items=ranked)

        self.propensity_model = profiler.create_propensity_model()
        return self.propensity_model.to_dict()

    def evaluate_profiles(
        self,
        profiles: List[UserProfile],
        num_shuffles: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate PROPEL ranking accuracy over a set of user profiles.
        
        Returns average Hit@1, NDCG@5, NDCG@20, and MRR.
        """
        hit1_list = []
        ndcg5_list = []
        ndcg20_list = []
        mrr_list = []

        for profile in profiles:
            gt = profile.ground_truth_title
            if not gt:
                continue

            ranked_titles, _ = self.rerank(
                user_history=profile.history,
                candidates=profile.candidates,
                num_shuffles=num_shuffles,
                recommendation_id=f"eval_{profile.user_id}",
            )

            hit1_list.append(compute_hit(gt, ranked_titles, k=1))
            ndcg5_list.append(compute_ndcg(gt, ranked_titles, k=5))
            ndcg20_list.append(compute_ndcg(gt, ranked_titles, k=20))
            mrr_list.append(compute_mrr(gt, ranked_titles))

        return {
            "hit@1": float(sum(hit1_list) / len(hit1_list)) if hit1_list else 0.0,
            "ndcg@5": float(sum(ndcg5_list) / len(ndcg5_list)) if ndcg5_list else 0.0,
            "ndcg@20": float(sum(ndcg20_list) / len(ndcg20_list)) if ndcg20_list else 0.0,
            "mrr": float(sum(mrr_list) / len(mrr_list)) if mrr_list else 0.0,
            "num_evaluated_users": len(hit1_list),
        }
