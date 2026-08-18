"""
PROPEL LLM Backend and Prompting Interface.

Implements prompt generation (Table 1) and parallelized LLM ranking calls with
strict JSON parsing, retry logic, and rate limiting.
"""

import os
import re
import json
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

from propel.defaults import DOMAIN_PROMPT_CONFIG

logger = logging.getLogger(__name__)


def format_item_string(item: Dict[str, Any], include_rating: bool = False) -> str:
    """Format an individual item dictionary into prompt text."""
    parts = [f"Title: {item.get('title', item.get('Title', 'Unknown'))}"]
    
    genres = item.get("genres", item.get("Genres", []))
    if genres:
        if isinstance(genres, list):
            parts.append(f"Genres: {', '.join(genres)}")
        else:
            parts.append(f"Genres: {genres}")
            
    price = item.get("price", item.get("Price"))
    if price is not None and price != "null":
        try:
            v = float(price)
            if v > 0:
                parts.append(f"Price: ${v:.2f}")
        except (ValueError, TypeError):
            pass
            
    if include_rating:
        rating = item.get("rating", item.get("Rating"))
        if rating is not None:
            parts.append(f"Rating: {rating}")
            
    return " | ".join(parts)


def build_ranking_prompt(
    dataset: str,
    user_history: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> str:
    """
    Construct LLM ranking prompt following Table 1 in the PROPEL paper.
    """
    config = DOMAIN_PROMPT_CONFIG.get(dataset.lower(), DOMAIN_PROMPT_CONFIG["default"])
    system_role = config["system_role"]
    json_key = config["json_key"]

    n_cands = len(candidates)
    if n_cands <= 5:
        ex_list = str(list(range(1, n_cands + 1)))
    else:
        ex_list = str(list(range(1, 6)))[:-1] + ", ...]"

    prompt_lines = [
        f"You are a {system_role}. Rerank all the candidates from most to least recommended.",
        f'Return ONLY a JSON object with candidate numbers in order of preference. Do NOT output any additional text or markdown formatting outside of the raw JSON object. Example: {{"{json_key}": {ex_list}}}',
        "",
        "User history:",
    ]

    # Format user history (up to 5 items)
    for idx, item in enumerate(user_history[-5:], start=1):
        prompt_lines.append(f"{idx}) {format_item_string(item, include_rating=True)}")

    prompt_lines.extend(["", "Candidates to rank:"])
    for idx, item in enumerate(candidates, start=1):
        prompt_lines.append(f"{idx}) {format_item_string(item, include_rating=False)}")

    prompt_lines.extend(["", "Output:"])
    return "\n".join(prompt_lines)


def parse_ranking_response(response_text: str, candidates: List[Dict[str, Any]]) -> List[str]:
    """
    Extract ranked candidate titles from LLM JSON response.
    
    Handles raw JSON, markdown code fences, and fallback regex extraction.
    """
    cand_titles = [c.get("title", c.get("Title", str(c))) for c in candidates]
    num_candidates = len(candidates)
    ranked_titles: List[str] = []
    seen = set()

    # Clean markdown formatting if present
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

    # Attempt JSON parsing
    json_obj = None
    try:
        json_obj = json.loads(cleaned)
    except Exception:
        # Try finding JSON block in text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                json_obj = json.loads(match.group(0))
            except Exception:
                pass

    indices = []
    if isinstance(json_obj, dict):
        for val in json_obj.values():
            if isinstance(val, list):
                indices = val
                break
    elif isinstance(json_obj, list):
        indices = json_obj

    # Fallback to regex digits if JSON failed
    if not indices:
        digits = re.findall(r"\b\d+\b", cleaned)
        indices = [int(d) for d in digits]

    for idx_item in indices:
        try:
            val = int(idx_item) - 1  # 1-indexed to 0-indexed
            if 0 <= val < num_candidates:
                title = cand_titles[val]
                if title not in seen:
                    ranked_titles.append(title)
                    seen.add(title)
        except (ValueError, TypeError):
            continue

    # Ensure all candidates are present (append missing candidates at the end)
    for title in cand_titles:
        if title not in seen:
            ranked_titles.append(title)
            seen.add(title)

    return ranked_titles


class LLMClient:
    """
    Unified LLM Client supporting OpenAI, Anthropic, and custom backends.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        backend: str = "openai",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        temperature: float = 0.0,
        custom_caller: Optional[Callable[[str], str]] = None,
    ):
        self.model = model
        self.backend = backend.lower()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY" if self.backend == "openai" else "ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self.temperature = temperature
        self.custom_caller = custom_caller
        self._openai_client = None
        self._anthropic_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(api_key=self.api_key)
        return self._openai_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

    def call_single(self, prompt: str) -> str:
        """Execute a single LLM API call with retry and exponential backoff."""
        if self.custom_caller is not None:
            return self.custom_caller(prompt)

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                if self.backend == "openai":
                    client = self._get_openai_client()
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content or ""
                elif self.backend == "anthropic":
                    client = self._get_anthropic_client()
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=1000,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text
                else:
                    raise ValueError(f"Unsupported backend: {self.backend}")
            except Exception as e:
                logger.warning(f"LLM API error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay + random.uniform(0.1, 0.5))
                delay *= 2.0
        return ""

    def call_parallel(
        self,
        prompts: List[str],
        max_workers: int = 10,
    ) -> List[str]:
        """Execute multiple LLM calls concurrently."""
        if len(prompts) == 1:
            return [self.call_single(prompts[0])]

        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.call_single, prompt): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [r if r is not None else "" for r in results]
