"""
PROPEL Quickstart Example.

Demonstrates how to perform position-bias-corrected LLM reranking using PROPEL
and export structured explainability reports.
"""

import os
import sys

# Ensure repository root is on sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from propel import PROPEL, LLMClient

def main():
    print("🚀 Initializing PROPEL Reranker...")

    # Mock or live LLM backend
    # If OPENAI_API_KEY is not set, we provide a deterministic mock caller for demonstration
    mock_mode = "OPENAI_API_KEY" not in os.environ

    if mock_mode:
        print("ℹ️  OPENAI_API_KEY not set - running with demo mock LLM backend.")
        def mock_llm(prompt: str) -> str:
            # Simple mock that prioritizes movies with 'Star' or 'Godfather' in title
            # and otherwise exhibits slight primacy bias
            import re
            return '{"ranked_movies": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}'

        client = LLMClient(custom_caller=mock_llm)
        reranker = PROPEL(model="gpt-4o-mini", dataset="ml-1m", llm_client=client)
    else:
        reranker = PROPEL(model="gpt-4o-mini", dataset="ml-1m")

    # Sample user history (movies previously liked)
    user_history = [
        {"title": "The Matrix (1999)", "genres": ["Action", "Sci-Fi"], "rating": 5.0},
        {"title": "Inception (2010)", "genres": ["Action", "Sci-Fi", "Thriller"], "rating": 5.0},
        {"title": "Interstellar (2014)", "genres": ["Adventure", "Drama", "Sci-Fi"], "rating": 4.5},
        {"title": "Blade Runner 2049 (2017)", "genres": ["Drama", "Mystery", "Sci-Fi"], "rating": 4.0},
        {"title": "Dark City (1998)", "genres": ["Mystery", "Sci-Fi"], "rating": 4.0},
    ]

    # Sample candidate movies to rerank
    candidates = [
        {"title": "Terminator 2: Judgment Day (1991)", "genres": ["Action", "Sci-Fi"]},
        {"title": "The Godfather (1972)", "genres": ["Crime", "Drama"]},
        {"title": "Star Wars: Episode V (1980)", "genres": ["Action", "Adventure", "Sci-Fi"]},
        {"title": "Pulp Fiction (1994)", "genres": ["Crime", "Drama"]},
        {"title": "Alien (1979)", "genres": ["Horror", "Sci-Fi"]},
        {"title": "The Shawshank Redemption (1994)", "genres": ["Drama"]},
        {"title": "2001: A Space Odyssey (1968)", "genres": ["Adventure", "Sci-Fi"]},
        {"title": "Fight Club (1999)", "genres": ["Drama"]},
        {"title": "The Prestige (2006)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
        {"title": "Memento (2000)", "genres": ["Mystery", "Thriller"]},
        {"title": "Arrival (2016)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
        {"title": "Minority Report (2002)", "genres": ["Action", "Crime", "Sci-Fi"]},
        {"title": "Twelve Monkeys (1995)", "genres": ["Mystery", "Sci-Fi", "Thriller"]},
        {"title": "Children of Men (2006)", "genres": ["Drama", "Sci-Fi", "Thriller"]},
        {"title": "Ex Machina (2014)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
        {"title": "Solaris (1972)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
        {"title": "Gattaca (1997)", "genres": ["Drama", "Sci-Fi", "Thriller"]},
        {"title": "Total Recall (1990)", "genres": ["Action", "Sci-Fi"]},
        {"title": "Contact (1997)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
        {"title": "Moon (2009)", "genres": ["Drama", "Mystery", "Sci-Fi"]},
    ]

    print(f"📊 Reranking {len(candidates)} candidates using PROPEL (N_S = 10 shuffles)...")
    final_ranking, report = reranker.rerank(
        user_history=user_history,
        candidates=candidates,
        num_shuffles=10,
        seed=42,
    )

    print("\n" + report.summary())

    # Print top 5 items
    print("\n🏆 Top 5 Recommendations:")
    for idx, item in enumerate(final_ranking[:5], start=1):
        print(f"  {idx}. {item}")


if __name__ == "__main__":
    main()
