"""
PROPEL Propensity Modeling Module.

Implements the parametric exponential Softmax propensity formulation from
Section 4.1 (Equation 2) of the PROPEL paper:
  - Linear primacy basis: (1 - x)
  - Linear recency basis: x
  - Symmetric unimodal quadratic middle-centrality basis: 1 - 4(x - 0.5)^2
  - Softmax selection probabilities P(p)
  - Normalized propensity scores S_hat(p) = N * P(p)
  - Inverse propensity weights w(p) = 1 / (N * P(p))
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class PropensityModel:
    """
    Parametric Softmax Propensity Model for Position Bias Correction.
    
    Parameters
    ----------
    B_prim : float
        Primacy bias coefficient (favors beginning of candidate list).
    B_rec : float
        Recency bias coefficient (favors end of candidate list).
    B_mid : float
        Middle-ignoring bias coefficient (favors/disfavors middle positions).
    """

    def __init__(self, B_prim: float = 0.0, B_rec: float = 0.0, B_mid: float = 0.0):
        self.B_prim = float(B_prim)
        self.B_rec = float(B_rec)
        self.B_mid = float(B_mid)

    @classmethod
    def from_bias_dict(cls, bias_dict: Dict[str, float]) -> "PropensityModel":
        """Initialize from dictionary of bias coefficients."""
        return cls(
            B_prim=bias_dict.get("B_prim", bias_dict.get("primacy", 0.0)),
            B_rec=bias_dict.get("B_rec", bias_dict.get("recency", 0.0)),
            B_mid=bias_dict.get("B_mid", bias_dict.get("middle", 0.0)),
        )

    def compute_logits(self, N: int) -> np.ndarray:
        """
        Compute structural basis function logits for positions 1..N.
        
        Logits: z(x) = B_prim * (1 - x) + B_rec * x + B_mid * [1 - 4(x - 0.5)^2]
        where normalized coordinate x = (p - 1) / (N - 1).
        """
        if N <= 0:
            raise ValueError(f"Candidate list length N must be positive, got {N}")
        if N == 1:
            return np.array([0.0], dtype=float)

        positions = np.arange(1, N + 1, dtype=float)
        x = (positions - 1.0) / (N - 1.0)

        primacy_basis = 1.0 - x
        recency_basis = x
        middle_basis = 1.0 - 4.0 * ((x - 0.5) ** 2)

        logits = (
            self.B_prim * primacy_basis
            + self.B_rec * recency_basis
            + self.B_mid * middle_basis
        )
        return logits

    def compute_probabilities(self, N: int) -> np.ndarray:
        """
        Compute valid probability distribution P(p) over positions 1..N via Softmax (Eq. 2).
        """
        logits = self.compute_logits(N)
        # Stable softmax
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def get_normalized_propensities(self, N: int) -> Dict[int, float]:
        """
        Compute normalized propensity scores S_hat(p) = N * P(p).
        
        Equals 1 under no bias, >1 for over-favored positions, <1 for neglected positions.
        """
        probs = self.compute_probabilities(N)
        return {p: float(N * probs[p - 1]) for p in range(1, N + 1)}

    def get_inverse_propensity_weights(self, N: int) -> Dict[int, float]:
        """
        Compute inverse propensity weights w(p) = 1 / (N * P(p)) for positions 1..N.
        
        Down-weights over-favored positions (w(p) < 1), up-weights neglected positions (w(p) > 1).
        """
        probs = self.compute_probabilities(N)
        weights = {}
        for p in range(1, N + 1):
            prob = probs[p - 1]
            s_hat = N * prob
            weights[p] = float(1.0 / s_hat) if s_hat > 1e-12 else 1e12
        return weights

    def get_propensity_curve_data(self, N: int, K: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Get [position, scaled propensity] pairs for plotting and explainability reports.
        
        Scaled frequency: S_total(p) = K * P(p) (default K = N).
        """
        if K is None:
            K = N
        probs = self.compute_probabilities(N)
        return [(p, float(K * probs[p - 1])) for p in range(1, N + 1)]

    def to_dict(self) -> Dict[str, float]:
        """Return dictionary representation of bias coefficients."""
        return {
            "B_prim": self.B_prim,
            "B_rec": self.B_rec,
            "B_mid": self.B_mid,
        }

    def __repr__(self) -> str:
        return f"PropensityModel(B_prim={self.B_prim:+.3f}, B_rec={self.B_rec:+.3f}, B_mid={self.B_mid:+.3f})"
