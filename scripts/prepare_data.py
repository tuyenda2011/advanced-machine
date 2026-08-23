import argparse
import os
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
import yaml

from src.data.loader import download_amazon_electronics, load_raw_data
from src.data.negative_collector import extract_explicit_negative_interactions
from src.data.preprocessing import preprocess_amazon_electronics
from src.data.splitter import chronological_per_user_split, verify_no_leakage
from src.data.text_encoder import encode_item_metadata
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Electronics dataset")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to config dir")
    parser.add_argument(
        "--extract_text_embeddings",
        action="store_true",
        default=True,
        help="Extract Dense Semantic Text Embeddings for items using Sentence-Transformers",
    )
    parser.add_argument(
        "--extract_hard_negatives",
        action="store_true",
        default=True,
        help="Extract Explicit Disliked Interactions (1-2 stars) for Hard Negative Mining",
    )
    args = parser.parse_args()

    config = load_config("lightgcn", args.config_dir)
    data_cfg = config["dataset"]

    # 1. Download data
    dataset_dir = download_amazon_electronics(
        data_cfg["raw_dir"], 
        reviews_url=data_cfg.get("reviews_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"),
        meta_url=data_cfg.get("meta_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz")
    )

    from src.data.validation import validate_interactions, validate_metadata

    # 2. Load raw data
    ratings_df, items_df = load_raw_data(dataset_dir)

    # 3. Preprocess
    df, user2id, item2id, item_metadata, stats = preprocess_amazon_electronics(
        ratings_df,
        items_df,
        positive_threshold=data_cfg["positive_rating_threshold"],
        min_user_interactions=data_cfg["min_user_interactions"],
        min_item_interactions=data_cfg.get("min_item_interactions", 5),
    )

    # 4. Validate cleaned metadata & interactions
    meta_df = pd.DataFrame(list(item_metadata.values()))
    validate_metadata(meta_df, raise_on_error=True)

    # 5. Split
    val_ratio = data_cfg["split_ratios"][1]
    test_ratio = data_cfg["split_ratios"][2]
    train_df, val_df, test_df = chronological_per_user_split(
        df, val_ratio=val_ratio, test_ratio=test_ratio, enforce_connectivity=True
    )

    # 6. Leakage check
    verify_no_leakage(train_df, val_df, test_df)

    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # 7. Optional: Extract explicit hard negatives (1-2 stars)
    user_disliked_map = {}
    if args.extract_hard_negatives:
        neg_save_path = os.path.join(processed_dir, "disliked_interactions.parquet")
        _, user_disliked_map = extract_explicit_negative_interactions(
            ratings_df, user2id, item2id, negative_threshold=2.0, save_path=neg_save_path
        )

    # 8. Optional: Extract item text embeddings and cache projection
    if args.extract_text_embeddings:
        text_emb_path = os.path.join(processed_dir, "item_text_embeddings.pt")
        text_tensors = encode_item_metadata(
            item_metadata=item_metadata,
            num_items=len(item2id),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            save_path=text_emb_path,
        )
        # Pre-cache projection embedding for fast training startup
        from scripts.cache_text_proj import cache_projection_pipeline
        cache_proj_path = os.path.join("data", "cache", "item_proj.pt")
        try:
            cache_projection_pipeline(text_emb_path, cache_proj_path, embedding_dim=config["model"].get("embedding_dim", 64))
        except Exception as exc:
            logger.warning(f"Cache projection creation skipped or failed ({exc}).")

    # 9. Save processed artifacts
    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df.to_parquet(os.path.join(processed_dir, "val.parquet"))
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"))

    with open(os.path.join(processed_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(
            {
                "user2id": user2id,
                "item2id": item2id,
                "item_metadata": item_metadata,
                "stats": stats,
                "user_disliked_items": user_disliked_map,
            },
            f,
        )

    logger.info(f"All processed data and mappings saved to {processed_dir}")
    logger.info("Dataset statistics summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")



if __name__ == "__main__":
    main()

