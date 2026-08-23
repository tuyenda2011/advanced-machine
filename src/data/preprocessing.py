import html
import logging
import re
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def clean_text(text: Any) -> str:
    """Clean and unescape HTML entities and remove raw HTML tags from text."""
    if text is None or not isinstance(text, (str, bytes)):
        return UNKNOWN_PLACEHOLDER
    text_str = str(text)
    # Decode HTML entities like &amp;, &#39;, &quot;
    text_str = html.unescape(text_str)
    # Remove HTML tags
    text_str = re.sub(r"<[^>]+>", " ", text_str)
    # Remove redundant whitespace
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str if text_str else UNKNOWN_PLACEHOLDER


UNKNOWN_PLACEHOLDER = "unknown item"


def parse_categories(categories_raw: Any) -> str:
    """Parse nested category list and format into clear hierarchy string."""
    if isinstance(categories_raw, list) and len(categories_raw) > 0:
        flat_cats = []
        for item in categories_raw:
            if isinstance(item, list):
                flat_cats.extend(
                    [clean_text(c) for c in item if c and clean_text(c) != UNKNOWN_PLACEHOLDER]
                )
            elif isinstance(item, str):
                cleaned = clean_text(item)
                if cleaned != UNKNOWN_PLACEHOLDER:
                    flat_cats.append(cleaned)
        if flat_cats:
            # Deduplicate sequential identical categories while preserving order
            unique_seq = []
            for c in flat_cats:
                if not unique_seq or unique_seq[-1] != c:
                    unique_seq.append(c)
            return " > ".join(unique_seq[-3:])
    return "Electronics"


def preprocess_amazon_electronics(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    positive_threshold: float = 4.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[int, Dict[str, str]], Dict[str, Any]]:
    """Convert ratings to implicit feedback, deduplicate interactions, filter bipartite K-core,
    re-index contiguous IDs, clean metadata, and calculate graph statistics.
    """
    logger.info(
        f"Preprocessing ratings: positive threshold >= {positive_threshold}, "
        f"min_user_interactions >= {min_user_interactions}, min_item_interactions >= {min_item_interactions}"
    )

    # Normalize column names from Amazon to standard
    ratings_df = ratings_df.rename(columns={
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "unixReviewTime": "timestamp"
    })

    # 1. Filter implicit positive feedback
    df = ratings_df[ratings_df["rating"] >= positive_threshold].copy()
    logger.info(f"Retained {len(df)} positive interactions out of {len(ratings_df)} total ratings.")

    # 2. De-duplication: For identical (user_id, item_id), keep the most recent
    # interaction; tie-break by highest rating (spec: [timestamp DESC, rating DESC]).
    initial_len = len(df)
    df = df.sort_values(by=["timestamp", "rating"], ascending=[False, False])
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first").copy()
    num_dups = initial_len - len(df)
    if num_dups > 0:
        logger.info(f"Removed {num_dups} duplicate (user_id, item_id) interactions.")

    # 3. Iterative Bipartite K-core (Users >= min_user_interactions AND Items >= min_item_interactions)
    filter_pbar = tqdm(total=None, desc="Filtering Bipartite K-core (Users & Items)", unit=" passes")
    pass_count = 0
    while True:
        pass_count += 1
        filter_pbar.update(1)
        prev_len = len(df)

        user_counts = df["user_id"].value_counts()
        valid_users = set(user_counts[user_counts >= min_user_interactions].index)

        item_counts = df["item_id"].value_counts()
        valid_items = set(item_counts[item_counts >= min_item_interactions].index)

        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]

        if len(df) == prev_len:
            break
    filter_pbar.close()
    logger.info(f"Bipartite K-core converged in {pass_count} passes. Final interactions: {len(df)}.")

    # 4. Create contiguous 0-indexed mappings
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user2id = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item2id = {item_id: idx for idx, item_id in enumerate(unique_items)}

    df["u_idx"] = df["user_id"].map(user2id)
    df["i_idx"] = df["item_id"].map(item2id)

    # 5. Map and clean item metadata for contiguous item IDs
    items_df = items_df.drop_duplicates(subset=["asin"]).set_index("asin")
    items_dict = items_df.to_dict(orient="index")
    item_metadata = {}

    for item_asin, idx in tqdm(item2id.items(), desc="Cleaning & Mapping Metadata", unit=" items"):
        meta = items_dict.get(
            item_asin,
            {"title": f"Item {item_asin}", "brand": "Unknown", "categories": [["Electronics"]]},
        )

        title_cleaned = clean_text(meta.get("title", f"Item {item_asin}"))
        brand_cleaned = clean_text(meta.get("brand", "Unknown"))
        categories_cleaned = parse_categories(meta.get("categories", [["Electronics"]]))

        item_metadata[idx] = {
            "title": title_cleaned,
            "brand": brand_cleaned,
            "categories": categories_cleaned,
            "original_id": str(item_asin),
        }

    # 6. Compute graph statistics
    num_users = len(unique_users)
    num_items = len(unique_items)
    num_interactions = len(df)
    density = num_interactions / (num_users * num_items)

    user_interaction_counts = df.groupby("u_idx").size()
    item_interaction_counts = df.groupby("i_idx").size()

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "density": float(density),
        "mean_user_interactions": float(user_interaction_counts.mean()),
        "median_user_interactions": float(user_interaction_counts.median()),
        "min_user_interactions": int(user_interaction_counts.min()),
        "max_user_interactions": int(user_interaction_counts.max()),
        "mean_item_interactions": float(item_interaction_counts.mean()),
        "min_item_interactions": int(item_interaction_counts.min()),
    }

    logger.info(
        f"Preprocessing complete: Users={num_users}, Items={num_items}, "
        f"Interactions={num_interactions}, Density={density:.6f}, "
        f"MinUserInteractions={stats['min_user_interactions']}, MinItemInteractions={stats['min_item_interactions']}"
    )

    return df[["u_idx", "i_idx", "timestamp"]], user2id, item2id, item_metadata, stats

