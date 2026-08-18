"""
Unit tests for propel.llm.
"""

from propel.llm import (
    build_ranking_prompt,
    parse_ranking_response,
    format_item_string,
)


def test_format_item_string():
    """Verify item string formatting."""
    item = {"title": "The Matrix", "genres": ["Action", "Sci-Fi"], "price": "14.99", "rating": 4.5}
    s = format_item_string(item, include_rating=True)
    assert "Title: The Matrix" in s
    assert "Genres: Action, Sci-Fi" in s
    assert "Price: $14.99" in s
    assert "Rating: 4.5" in s


def test_build_ranking_prompt():
    """Verify prompt formatting matching Table 1 in paper."""
    history = [{"title": "Film A", "genres": ["Comedy"], "rating": 5.0}]
    candidates = [{"title": "Film B", "genres": ["Drama"]}, {"title": "Film C", "genres": ["Horror"]}]

    prompt = build_ranking_prompt("ml-1m", history, candidates)
    assert "movie recommendation system" in prompt
    assert "ranked_movies" in prompt
    assert "User history:" in prompt
    assert "1) Title: Film A" in prompt
    assert "Candidates to rank:" in prompt
    assert "1) Title: Film B" in prompt
    assert "2) Title: Film C" in prompt


def test_parse_ranking_response_json():
    """Test parsing clean JSON response."""
    candidates = [{"title": "Film A"}, {"title": "Film B"}, {"title": "Film C"}]
    resp = '{"ranked_movies": [2, 1, 3]}'
    ranked = parse_ranking_response(resp, candidates)
    assert ranked == ["Film B", "Film A", "Film C"]


def test_parse_ranking_response_markdown_fence():
    """Test parsing JSON enclosed in markdown code fences."""
    candidates = [{"title": "Film A"}, {"title": "Film B"}]
    resp = "```json\n{\n  \"ranked_movies\": [2, 1]\n}\n```"
    ranked = parse_ranking_response(resp, candidates)
    assert ranked == ["Film B", "Film A"]


def test_parse_ranking_response_incomplete_repaired():
    """Test response that omits an item (should append omitted items at end)."""
    candidates = [{"title": "Film A"}, {"title": "Film B"}, {"title": "Film C"}]
    resp = '{"ranked_movies": [3]}'
    ranked = parse_ranking_response(resp, candidates)
    assert len(ranked) == 3
    assert ranked[0] == "Film C"
    assert "Film A" in ranked
    assert "Film B" in ranked
