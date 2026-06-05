"""
create_datasets.py
------------------
Unified master script to generate evaluation datasets for all supported sources:
MovieLens-1M, Amazon Books, Amazon Beauty, Amazon Music, and Steam.

Usage:
  python create_datasets.py --dataset all
  python create_datasets.py --dataset ml-1m
"""

import argparse
import gzip
import json
import random
import os
import ast
import pandas as pd
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────
SEED        = 42
N_TEST      = 150
N_PROBE     = 50
N_HARD_TEST = 100
MIN_HISTORY = 6

N_EVAL_CANDIDATES = 20      # 1 GT + 19 random
N_HARD_CANDIDATES = 30      # 1 GT + 19 random + 10 category-matched
N_PROBE_CANDIDATES = 100    # 100 random (no GT)
NEGATIVE_POOL_SIZE = 10000

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Extractor Classes ─────────────────────────────────────────────────────────

class ML1MExtractor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.has_user_info = True

    def extract(self):
        print("Loading ML-1M files via Pandas...")
        users_df = pd.read_csv(os.path.join(self.data_dir, "users.dat"), sep="::", engine="python", names=["UserID", "Gender", "Age", "Occupation", "Zip"], encoding="latin-1")
        movies_df = pd.read_csv(os.path.join(self.data_dir, "movies.dat"), sep="::", engine="python", names=["MovieID", "Title", "Genres"], encoding="latin-1")
        ratings_df = pd.read_csv(os.path.join(self.data_dir, "ratings.dat"), sep="::", engine="python", names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin-1")

        users_lookup = users_df.set_index("UserID").to_dict(orient="index")
        movie_info = {str(row["MovieID"]): {"title": row["Title"], "genres": [g.strip() for g in row["Genres"].split("|") if g.strip()], "price": None} for _, row in movies_df.iterrows()}

        # Build interactions
        rating_counts = ratings_df.groupby("UserID").size()
        eligible_user_ids = rating_counts[rating_counts >= MIN_HISTORY].index.tolist()
        
        user_interactions = defaultdict(list)
        for uid in eligible_user_ids:
            user_ratings = ratings_df[ratings_df["UserID"] == uid].sort_values("Timestamp")
            for _, row in user_ratings.iterrows():
                user_interactions[str(uid)].append({
                    "asin": str(row["MovieID"]),
                    "rating": int(row["Rating"]),
                    "timestamp": int(row["Timestamp"])
                })

        age_map = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
        occ_map = {0: "other/not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin", 4: "college/grad student", 5: "customer service", 6: "doctor/health care", 7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}

        user_metadata = {}
        for uid, row in users_lookup.items():
            user_metadata[str(uid)] = {
                "gender": row.get("Gender", "Unknown"),
                "age_group": age_map.get(row.get("Age", -1), "Unknown"),
                "occupation": occ_map.get(row.get("Occupation", -1), "Unknown")
            }

        return list(user_interactions.keys()), user_interactions, movie_info, list(movie_info.values()), user_metadata


class AmazonExtractor:
    def __init__(self, data_dir, interactions_file, meta_file, filter_category=None):
        self.interactions_file = os.path.join(data_dir, interactions_file)
        self.meta_file = os.path.join(data_dir, meta_file)
        self.filter_category = filter_category
        self.has_user_info = False

    def extract(self):
        print(f"Streaming {os.path.basename(self.interactions_file)} for users...")
        user_interactions = defaultdict(list)
        eligible_users = set()
        total_needed = N_TEST + N_PROBE + N_HARD_TEST

        with gzip.open(self.interactions_file, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                uid = data.get("user_id")
                asin = data.get("parent_asin") or data.get("asin")
                if not uid or not asin: continue
                    
                user_interactions[uid].append({
                    "asin": asin,
                    "rating": data.get("rating"),
                    "timestamp": data.get("timestamp")
                })
                
                if len(user_interactions[uid]) == MIN_HISTORY:
                    eligible_users.add(uid)
                if len(eligible_users) >= total_needed * 2:
                    break
        
        target_users = list(eligible_users)
        
        target_asins = set()
        for uid in target_users:
            user_interactions[uid].sort(key=lambda x: x["timestamp"] if x["timestamp"] else 0)
            for item in user_interactions[uid]:
                target_asins.add(item["asin"])

        print(f"Streaming {os.path.basename(self.meta_file)} for items...")
        item_metadata = {}
        negative_pool = []

        with gzip.open(self.meta_file, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                asin = data.get("parent_asin") or data.get("asin")
                title = data.get("title")
                if not asin or not title: continue
                    
                categories = data.get("categories", [])
                if self.filter_category:
                    categories = [c for c in categories if c.lower() != self.filter_category.lower()]
                    
                obj = {"asin": asin, "title": title, "genres": categories, "price": data.get("price")}
                
                if asin in target_asins:
                    item_metadata[asin] = obj
                
                if len(negative_pool) < NEGATIVE_POOL_SIZE and random.random() < 0.05:
                    negative_pool.append(obj)
                    
                if len(item_metadata) == len(target_asins) and len(negative_pool) >= NEGATIVE_POOL_SIZE:
                    break

        return target_users, user_interactions, item_metadata, negative_pool, {}


class SteamExtractor:
    def __init__(self, data_dir):
        self.interactions_file = os.path.join(data_dir, "steam_reviews.json.gz")
        self.meta_file = os.path.join(data_dir, "steam_games.json.gz")
        self.has_user_info = False

    def extract(self):
        print(f"Streaming steam_reviews.json.gz for users via AST...")
        user_interactions = defaultdict(list)
        eligible_users = set()
        total_needed = N_TEST + N_PROBE + N_HARD_TEST

        with gzip.open(self.interactions_file, "rt", encoding="utf-8") as f:
            for line in f:
                try: data = ast.literal_eval(line)
                except: continue
                uid = data.get("username")
                asin = str(data.get("product_id"))
                if not uid or not asin: continue
                    
                user_interactions[uid].append({
                    "asin": asin,
                    "rating": 5 if data.get("recommended") else 1,
                    "timestamp": data.get("date")
                })
                
                if len(user_interactions[uid]) == MIN_HISTORY:
                    eligible_users.add(uid)
                if len(eligible_users) >= total_needed * 2:
                    break
        
        target_users = list(eligible_users)
        
        target_asins = set()
        for uid in target_users:
            user_interactions[uid].sort(key=lambda x: str(x["timestamp"]))
            for item in user_interactions[uid]:
                target_asins.add(item["asin"])

        print(f"Streaming steam_games.json.gz for items via AST...")
        item_metadata = {}
        negative_pool = []

        with gzip.open(self.meta_file, "rt", encoding="utf-8") as f:
            for line in f:
                try: data = ast.literal_eval(line)
                except: continue
                asin = str(data.get("id"))
                title = data.get("title") or data.get("app_name")
                if not asin or not title: continue
                    
                obj = {"asin": asin, "title": title, "genres": data.get("genres", []), "price": data.get("price")}
                
                if asin in target_asins:
                    item_metadata[asin] = obj
                
                if len(negative_pool) < NEGATIVE_POOL_SIZE and random.random() < 0.2:
                    negative_pool.append(obj)
                    
                if len(item_metadata) == len(target_asins) and len(negative_pool) >= NEGATIVE_POOL_SIZE:
                    break

        return target_users, user_interactions, item_metadata, negative_pool, {}


# ── Pipeline Engine ───────────────────────────────────────────────────────────

def build_datasets(dataset_name, extractor):
    random.seed(SEED)
    print(f"\n=========================================")
    print(f"Processing Dataset: {dataset_name.upper()}")
    print(f"=========================================")
    
    out_dir = os.path.join(DATA_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    eligible_users, user_interactions, item_metadata, negative_pool, user_meta_dict = extractor.extract()
    
    random.shuffle(eligible_users)
    total_needed = N_TEST + N_PROBE + N_HARD_TEST
    if len(eligible_users) < total_needed:
        print(f"⚠️  Not enough users ({len(eligible_users)} < {total_needed}) for {dataset_name}. Skipping.")
        return

    test_ids = eligible_users[:N_TEST]
    probe_ids = eligible_users[N_TEST:N_TEST+N_PROBE]
    hard_test_ids = eligible_users[N_TEST+N_PROBE:total_needed]

    def build_base_records(user_ids):
        records = []
        for uid in user_ids:
            items = []
            for interaction in user_interactions[uid]:
                meta = item_metadata.get(interaction["asin"])
                if meta:
                    items.append({
                        "title": meta["title"],
                        "genres": meta["genres"],
                        "price": meta.get("price"),
                        "rating": int(interaction["rating"]) if interaction["rating"] else 0
                    })
            if len(items) < MIN_HISTORY:
                continue
            
            record = {
                "user_id": uid,
                "history": items[:-1],
                "ground_truth": items[-1]
            }
            if extractor.has_user_info:
                record["user_info"] = user_meta_dict.get(str(uid), {"gender": "Unknown", "age_group": "Unknown", "occupation": "Unknown"})
            records.append(record)
        return records

    def save(obj, filename):
        with open(os.path.join(out_dir, filename), "w") as f:
            json.dump(obj, f, indent=2)
            
    print("Building Base Records...")
    test_records = build_base_records(test_ids)
    probe_records = build_base_records(probe_ids)
    hard_test_records = build_base_records(hard_test_ids)
    
    save(test_records, "test_dataset.json")
    save([r["user_id"] for r in test_records], "test_user_ids.json")
    save(probe_records, "probe_dataset.json")
    save([r["user_id"] for r in probe_records], "probe_user_ids.json")
    save(hard_test_records, "hard_test_dataset.json")
    save([r["user_id"] for r in hard_test_records], "hard_test_user_ids.json")

    genre_to_items = defaultdict(list)
    for m in negative_pool:
        for g in m.get("genres", []):
            genre_to_items[g].append(m)

    print("Building Evaluation Datasets...")
    
    # 1. EVAL TEST
    eval_test = []
    for r in test_records:
        seen = {i["title"] for i in r["history"]} | {r["ground_truth"]["title"]}
        unseen = [m for m in negative_pool if m["title"] not in seen]
        negatives = random.sample(unseen, min(N_EVAL_CANDIDATES - 1, len(unseen)))
        
        candidates = [{"title": r["ground_truth"]["title"], "genres": r["ground_truth"]["genres"], "price": r["ground_truth"].get("price")}]
        candidates += [{"title": m["title"], "genres": m["genres"], "price": m.get("price")} for m in negatives]
        random.shuffle(candidates)
        
        record = {"user_id": r["user_id"]}
        if extractor.has_user_info: record["user_info"] = r["user_info"]
        record.update({"last_5": r["history"][-5:], "candidates": candidates, "ground_truth": r["ground_truth"]})
        eval_test.append(record)
    save(eval_test, "eval_test_dataset.json")

    # 2. EVAL HARD TEST
    eval_hard = []
    for r in hard_test_records:
        seen = {i["title"] for i in r["history"]} | {r["ground_truth"]["title"]}
        unseen = [m for m in negative_pool if m["title"] not in seen]
        
        random_negatives = random.sample(unseen, min(19, len(unseen)))
        chosen_titles = seen | {m["title"] for m in random_negatives}
        
        gt_categories = r["ground_truth"]["genres"]
        # Exclude root genres for matching
        exclude_roots = {"books", "music", "cds & vinyl"}
        specific_cats = [c for c in gt_categories if c.lower() not in exclude_roots]
        if not specific_cats: specific_cats = gt_categories
            
        selected_genre = random.choice(specific_cats) if specific_cats else None
        
        genre_negatives = []
        if selected_genre:
            genre_pool = [m for m in genre_to_items[selected_genre] if m["title"] not in chosen_titles]
            if len(genre_pool) >= 10:
                genre_negatives = random.sample(genre_pool, 10)
            else:
                genre_negatives = genre_pool
                extra_pool = [m for m in unseen if m["title"] not in (chosen_titles | {x["title"] for x in genre_negatives})]
                genre_negatives += random.sample(extra_pool, min(10 - len(genre_negatives), len(extra_pool)))
        else:
            extra_pool = [m for m in unseen if m["title"] not in chosen_titles]
            genre_negatives = random.sample(extra_pool, min(10, len(extra_pool)))
                
        candidates = [{"title": r["ground_truth"]["title"], "genres": r["ground_truth"]["genres"], "price": r["ground_truth"].get("price")}]
        candidates += [{"title": m["title"], "genres": m["genres"], "price": m.get("price")} for m in random_negatives]
        candidates += [{"title": m["title"], "genres": m["genres"], "price": m.get("price")} for m in genre_negatives]
        random.shuffle(candidates)
        
        record = {"user_id": r["user_id"]}
        if extractor.has_user_info: record["user_info"] = r["user_info"]
        record.update({"last_5": r["history"][-5:], "candidates": candidates, "ground_truth": r["ground_truth"]})
        eval_hard.append(record)
    save(eval_hard, "eval_hard_test_dataset.json")

    # 3. EVAL PROBE
    eval_probe = []
    for r in probe_records:
        seen = {i["title"] for i in r["history"]} | {r["ground_truth"]["title"]}
        unseen = [m for m in negative_pool if m["title"] not in seen]
        
        candidates_raw = random.sample(unseen, min(N_PROBE_CANDIDATES, len(unseen)))
        candidates = [{"title": m["title"], "genres": m["genres"], "price": m.get("price")} for m in candidates_raw]
        random.shuffle(candidates)
        
        record = {"user_id": r["user_id"]}
        if extractor.has_user_info: record["user_info"] = r["user_info"]
        record.update({"last_5": r["history"][-5:], "candidates": candidates, "ground_truth": r["ground_truth"]})
        eval_probe.append(record)
    save(eval_probe, "eval_probe_dataset.json")

    print(f"✅ Finished {dataset_name.upper()} successfully!")


# ── Main Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["all", "ml-1m", "books", "beauty", "music", "steam"])
    args = parser.parse_args()

    datasets = {
        "ml-1m": ML1MExtractor(os.path.join(DATA_ROOT, "ml-1m")),
        "books": AmazonExtractor(os.path.join(DATA_ROOT, "books"), "Books.jsonl.gz", "meta_Books.jsonl.gz"),
        "beauty": AmazonExtractor(os.path.join(DATA_ROOT, "beauty"), "All_Beauty.jsonl.gz", "meta_All_Beauty.jsonl.gz"),
        "music": AmazonExtractor(os.path.join(DATA_ROOT, "music"), "CDs_and_Vinyl.jsonl.gz", "meta_CDs_and_Vinyl.jsonl.gz", filter_category="CDs & Vinyl"),
        "steam": SteamExtractor(os.path.join(DATA_ROOT, "steam"))
    }

    if args.dataset == "all":
        targets = datasets.keys()
    else:
        targets = [args.dataset]

    for t in targets:
        build_datasets(t, datasets[t])
