import argparse
import os
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import yaml

from src.data.loader import download_amazon_electronics, load_raw_data
from src.data.preprocessing import preprocess_amazon_electronics
from src.data.splitter import chronological_per_user_split, verify_no_leakage
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Electronics dataset")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to config dir")
    args = parser.parse_args()

    config = load_config("lightgcn", args.config_dir)
    data_cfg = config["dataset"]

    # 1. Download data
    dataset_dir = download_amazon_electronics(
        data_cfg["raw_dir"], 
        reviews_url=data_cfg.get("reviews_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"),
        meta_url=data_cfg.get("meta_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz")
    )

    # 2. Load raw data
    ratings_df, items_df = load_raw_data(dataset_dir)

    # 3. Preprocess
    df, user2id, item2id, item_metadata, stats = preprocess_amazon_electronics(
        ratings_df,
        items_df,
        positive_threshold=data_cfg["positive_rating_threshold"],
        min_user_interactions=data_cfg["min_user_interactions"],
    )

    # 4. Split
    val_ratio = data_cfg["split_ratios"][1]
    test_ratio = data_cfg["split_ratios"][2]
    train_df, val_df, test_df = chronological_per_user_split(
        df, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # 5. Leakage check
    verify_no_leakage(train_df, val_df, test_df)

    # 6. Save processed artifacts
    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

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
            },
            f,
        )

    logger.info(f"All processed data saved to {processed_dir}")
    logger.info("Dataset statistics summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
