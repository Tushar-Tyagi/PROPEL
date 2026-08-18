"""
Pre-calibrated empirical bias parameters from the PROPEL paper (Table 2 & Table 3).

Contains measured Primacy (B_prim), Recency (B_rec), and Middle-Ignoring (B_mid)
bias coefficients obtained during the offline probing stage (N_b = 50 profiles, S = 20 shuffles).
"""

from typing import Dict, Any

# Pre-calibrated bias coefficients for GPT-4o-mini, GPT-4o, and GPT-3.5-turbo
# Across 5 standard benchmark datasets (Beauty, Books, MovieLens-1M, Music, Steam)
DEFAULT_BIAS_PARAMS: Dict[str, Dict[str, Dict[str, float]]] = {
    "gpt-4o-mini": {
        "beauty": {"B_prim": 2.572, "B_rec": -0.920, "B_mid": -0.826},
        "books": {"B_prim": 2.536, "B_rec": -0.912, "B_mid": -0.812},
        "ml-1m": {"B_prim": 1.834, "B_rec": -0.782, "B_mid": -0.526},
        "movielens": {"B_prim": 1.834, "B_rec": -0.782, "B_mid": -0.526},
        "music": {"B_prim": 2.568, "B_rec": -0.942, "B_mid": -0.813},
        "steam": {"B_prim": 2.698, "B_rec": -0.970, "B_mid": -0.864},
    },
    "gpt-4o": {
        "beauty": {"B_prim": 0.244, "B_rec": -0.324, "B_mid": 0.040},
        "books": {"B_prim": 0.289, "B_rec": -0.267, "B_mid": -0.011},
        "ml-1m": {"B_prim": 0.098, "B_rec": -0.076, "B_mid": -0.011},
        "movielens": {"B_prim": 0.098, "B_rec": -0.076, "B_mid": -0.011},
        "music": {"B_prim": 0.333, "B_rec": -0.133, "B_mid": -0.100},
        "steam": {"B_prim": 0.502, "B_rec": -0.258, "B_mid": -0.122},
    },
    "gpt-3.5-turbo": {
        "beauty": {"B_prim": 1.600, "B_rec": -0.884, "B_mid": -0.358},
        "books": {"B_prim": 2.138, "B_rec": -0.947, "B_mid": -0.596},
        "ml-1m": {"B_prim": 1.173, "B_rec": -0.813, "B_mid": -0.180},
        "movielens": {"B_prim": 1.173, "B_rec": -0.813, "B_mid": -0.180},
        "music": {"B_prim": 1.516, "B_rec": -0.942, "B_mid": -0.287},
        "steam": {"B_prim": 1.720, "B_rec": -0.880, "B_mid": -0.420},
    },
}

# Domain prompt system descriptions and JSON key names (Table 1)
DOMAIN_PROMPT_CONFIG: Dict[str, Dict[str, str]] = {
    "ml-1m": {
        "system_role": "movie recommendation system",
        "json_key": "ranked_movies",
        "category_name": "movie",
    },
    "movielens": {
        "system_role": "movie recommendation system",
        "json_key": "ranked_movies",
        "category_name": "movie",
    },
    "books": {
        "system_role": "book recommendation system",
        "json_key": "ranked_books",
        "category_name": "book",
    },
    "music": {
        "system_role": "music recommendation system",
        "json_key": "ranked_songs",
        "category_name": "song",
    },
    "beauty": {
        "system_role": "beauty product recommendation system",
        "json_key": "ranked_beauty",
        "category_name": "beauty product",
    },
    "steam": {
        "system_role": "game recommendation system",
        "json_key": "ranked_steam",
        "category_name": "game",
    },
    "default": {
        "system_role": "recommendation system",
        "json_key": "ranked_items",
        "category_name": "item",
    },
}
