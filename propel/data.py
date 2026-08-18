"""
PROPEL Dataset Loading and Management Module.

Provides standard dataset loaders for PROPEL benchmark cohorts:
  - Standard Test Set (150 users, 20 candidates: 1 GT + 19 random negatives)
  - Hard Test Set (100 users, 30 candidates: 1 GT + 19 random + 10 category-matched negatives)
  - Probe Set (50 users, 100 random candidates for offline bias profiling)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd


@dataclass
class UserProfile:
    """Represents a single recommendation user profile."""
    user_id: Union[int, str]
    history: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    ground_truth: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def candidate_titles(self) -> List[str]:
        return [c.get("title", c.get("Title", str(c))) for c in self.candidates]

    @property
    def ground_truth_title(self) -> Optional[str]:
        if self.ground_truth:
            return self.ground_truth.get("title", self.ground_truth.get("Title"))
        return None


def load_dataset_cohort(
    data_dir: str,
    dataset_name: str,
    cohort: str = "test",
) -> List[UserProfile]:
    """
    Load a standardized evaluation or probe cohort.
    
    Parameters
    ----------
    data_dir : str
        Path to data root directory.
    dataset_name : str
        Dataset name ('ml-1m', 'books', 'beauty', 'music', 'steam').
    cohort : str
        Cohort type ('test', 'hard_test', 'probe').
    """
    cohort_files = {
        "test": "eval_test_dataset.json",
        "hard_test": "eval_hard_test_dataset.json",
        "probe": "eval_probe_dataset.json",
    }
    filename = cohort_files.get(cohort, f"{cohort}.json")
    fpath = os.path.join(data_dir, dataset_name, filename)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Cohort file not found: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    profiles = []
    for entry in raw_data:
        profile = UserProfile(
            user_id=entry.get("user_id", entry.get("UserID")),
            history=entry.get("history", []),
            candidates=entry.get("candidates", []),
            ground_truth=entry.get("ground_truth"),
            metadata=entry.get("user_metadata", {}),
        )
        profiles.append(profile)

    return profiles


def create_user_profile_from_records(
    user_id: Union[int, str],
    user_history_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    item_col: str = "Title",
    rating_col: Optional[str] = "Rating",
    genre_col: Optional[str] = "Genres",
    ground_truth_title: Optional[str] = None,
) -> UserProfile:
    """Create a UserProfile from pandas DataFrames for custom user workflows."""
    history = []
    for _, row in user_history_df.iterrows():
        item_dict = {"title": str(row[item_col])}
        if rating_col and rating_col in row:
            item_dict["rating"] = row[rating_col]
        if genre_col and genre_col in row:
            item_dict["genres"] = [str(row[genre_col])] if not isinstance(row[genre_col], list) else row[genre_col]
        history.append(item_dict)

    candidates = []
    for _, row in candidate_df.iterrows():
        item_dict = {"title": str(row[item_col])}
        if genre_col and genre_col in row:
            item_dict["genres"] = [str(row[genre_col])] if not isinstance(row[genre_col], list) else row[genre_col]
        candidates.append(item_dict)

    gt_dict = {"title": ground_truth_title} if ground_truth_title else None

    return UserProfile(
        user_id=user_id,
        history=history,
        candidates=candidates,
        ground_truth=gt_dict,
    )
