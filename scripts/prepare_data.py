import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from importlib.metadata import version

# Ensure project root is in sys.path when script is executed directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import pandas as pd

from src.data.loader import download_amazon_electronics, load_raw_data
from src.data.metadata_overrides import apply_brand_overrides
from src.data.negative_collector import extract_explicit_negative_interactions
from src.data.preprocessing import (
    preprocess_amazon_electronics,
    summarize_metadata_quality,
)
from src.data.provenance import sha256_file
from src.data.splitter import (
    SPLIT_TIE_POLICY,
    chronological_per_user_split,
    summarize_split_timing,
    verify_no_leakage,
)
from src.data.text_encoder import PINNED_REVISION, encode_item_metadata
from src.data.validation import (
    validate_interactions,
    validate_metadata,
    validate_processed_interactions,
)
from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("prepare_data")


def write_dataset_manifest(
    dataset_dir,
    data_cfg,
    stats,
    train_df,
    val_df,
    test_df,
    text_tensors,
    hard_negative_count,
    manifest_path=None,
):
    """Write lightweight provenance and statistics for the course report."""
    source_files = []
    for name in ("reviews_Electronics_5.json.gz", "meta_Electronics.json.gz"):
        path = os.path.join(dataset_dir, name)
        source_files.append(
            {
                "path": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
                "size_bytes": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
        )

    total = len(train_df) + len(val_df) + len(test_df)
    train_items = set(train_df["i_idx"].unique())
    processed_dir = data_cfg["processed_dir"]
    artifact_names = ["train.parquet", "val.parquet", "test.parquet", "mappings.pkl"]
    if text_tensors is not None:
        artifact_names += ["item_text_embeddings.pt", "item_text_embeddings.pt.json"]
    if os.path.exists(os.path.join(processed_dir, "disliked_interactions.parquet")):
        artifact_names.append("disliked_interactions.parquet")
    artifacts = [
        {"path": os.path.relpath(os.path.join(processed_dir, name), REPO_ROOT).replace("\\", "/"),
         "sha256": sha256_file(os.path.join(processed_dir, name))}
        for name in artifact_names
    ]
    text_provenance = {}
    if text_tensors is not None:
        with open(os.path.join(processed_dir, "item_text_embeddings.pt.json"), encoding="utf-8") as stream:
            text_provenance = json.load(stream)
        text_provenance.pop("item_text_mask", None)
    manifest = {
        "dataset": data_cfg["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "artifacts": artifacts,
        "software_versions": {name: version(name) for name in ("pandas", "numpy", "torch")},
        "preprocessing": {
            "positive_rating_threshold": data_cfg["positive_rating_threshold"],
            "min_user_interactions": data_cfg["min_user_interactions"],
            "min_item_interactions": data_cfg.get("min_item_interactions", 5),
            "requested_split_ratios": data_cfg["split_ratios"],
            "split_seed": data_cfg.get("split_seed", 42),
            "split_protocol": "exact_chronological_selected_users",
            "timestamp_tie_policy": SPLIT_TIE_POLICY,
            "chronology_scope": "per_user_non_decreasing; ties do not establish intraday order",
            "positive_dedup_policy": "positive_only_latest_timestamp_then_highest_rating",
            "kcore_scope": "full_positive_graph_before_split",
            "kcore_converged": True,
        },
        "statistics": {
            **stats,
            "train_interactions": len(train_df),
            "validation_interactions": len(val_df),
            "test_interactions": len(test_df),
            "actual_split_ratios": [
                len(train_df) / total,
                len(val_df) / total,
                len(test_df) / total,
            ],
            "validation_cold_start_targets": int(
                (~val_df["i_idx"].isin(train_items)).sum()
            ),
            "test_cold_start_targets": int(
                (~test_df["i_idx"].isin(train_items)).sum()
            ),
            "hard_negative_interactions": hard_negative_count,
            "temporal_audit": summarize_split_timing(train_df, val_df, test_df),
            "evaluation_users": int(val_df["u_idx"].nunique()),
            "train_only_users": int(train_df["u_idx"].nunique() - val_df["u_idx"].nunique()),
        },
        "text_features": {
            **text_provenance,
            "encoder": "sentence-transformers/all-MiniLM-L6-v2",
            "shape": list(text_tensors.shape) if text_tensors is not None else None,
            "normalized": "usable_rows_unit_norm; masked_rows_zero" if text_tensors is not None else None,
        },
    }

    manifest_path = manifest_path or os.path.join(REPO_ROOT, "data", "manifest.json")
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    logger.info(f"Dataset manifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Amazon Electronics dataset")
    parser.add_argument("--config_dir", type=str, default="configs", help="Path to config dir")
    parser.add_argument("--output_dir", help="Optional staging directory for processed artifacts")
    parser.add_argument("--manifest_path", help="Manifest output (required with --output_dir)")
    parser.add_argument(
        "--extract_text_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract Dense Semantic Text Embeddings for items using Sentence-Transformers",
    )
    parser.add_argument(
        "--extract_hard_negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract Explicit Disliked Interactions (1-2 stars) for Hard Negative Mining",
    )
    parser.add_argument(
        "--export_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export human-readable inspection CSV files to data/processed/csv/ (default: True)",
    )
    args = parser.parse_args()
    if args.output_dir and not args.manifest_path:
        parser.error("--output_dir requires --manifest_path to avoid overwriting the active manifest")

    config = load_config("lightgcn", args.config_dir)
    data_cfg = config["dataset"]
    if args.output_dir:
        data_cfg["processed_dir"] = args.output_dir

    # 1. Download data
    dataset_dir = download_amazon_electronics(
        data_cfg["raw_dir"], 
        reviews_url=data_cfg.get("reviews_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"),
        meta_url=data_cfg.get("meta_url", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz")
    )

    # 2. Load raw data
    ratings_df, items_df = load_raw_data(dataset_dir)
    validate_interactions(ratings_df, raise_on_error=True)

    # 3. Preprocess
    df, user2id, item2id, item_metadata, stats = preprocess_amazon_electronics(
        ratings_df,
        items_df,
        positive_threshold=data_cfg["positive_rating_threshold"],
        min_user_interactions=data_cfg["min_user_interactions"],
        min_item_interactions=data_cfg.get("min_item_interactions", 5),
    )
    stats["ingestion_ledger"] = {"reviews": ratings_df.attrs.get("ingestion_ledger", {}),
                                 "metadata": items_df.attrs.get("ingestion_ledger", {})}

    stats["metadata_overrides"] = apply_brand_overrides(item_metadata)
    stats["metadata_quality"] = summarize_metadata_quality(item_metadata)

    # 4. Validate cleaned metadata & interactions
    meta_df = pd.DataFrame(list(item_metadata.values()))
    validate_metadata(meta_df, raise_on_error=True)

    # 5. Split
    val_ratio = data_cfg["split_ratios"][1]
    test_ratio = data_cfg["split_ratios"][2]
    train_df, val_df, test_df = chronological_per_user_split(
        df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        enforce_connectivity=False,
        seed=data_cfg.get("split_seed", 42),
    )

    # 6. Leakage check
    verify_no_leakage(train_df, val_df, test_df)
    for split_df in (train_df, val_df, test_df):
        validate_processed_interactions(split_df, raise_on_error=True)

    total = len(df)
    stats.update(
        {
            "train_interactions": len(train_df),
            "validation_interactions": len(val_df),
            "test_interactions": len(test_df),
            "actual_split_ratios": [
                len(train_df) / total,
                len(val_df) / total,
                len(test_df) / total,
            ],
        }
    )

    processed_dir = data_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    # 7. Optional: Extract explicit hard negatives (1-2 stars)
    user_disliked_map = {}
    neg_df = None
    if args.extract_hard_negatives:
        neg_save_path = os.path.join(processed_dir, "disliked_interactions.parquet")
        train_cutoffs = train_df.groupby("u_idx")["timestamp"].max().to_dict()
        neg_df, user_disliked_map = extract_explicit_negative_interactions(
            ratings_df,
            user2id,
            item2id,
            negative_threshold=2.0,
            save_path=neg_save_path,
            user_train_cutoffs=train_cutoffs,
        )

    # 8. Optional: Extract item text embeddings
    text_tensors = None
    if args.extract_text_embeddings:
        text_emb_path = os.path.join(processed_dir, "item_text_embeddings.pt")
        text_tensors = encode_item_metadata(
            item_metadata=item_metadata,
            num_items=len(item2id),
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            save_path=text_emb_path,
            revision=PINNED_REVISION,
        )

    # 9. Save processed artifacts
    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)

    # 10. Optional: Export human-readable CSV files for inspection
    if args.export_csv:
        csv_dir = os.path.join(processed_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        logger.info(f"Exporting inspection CSV files to {csv_dir}...")

        train_df.to_csv(os.path.join(csv_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(csv_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

        # Export clean metadata for easy product lookup
        meta_df = pd.DataFrame(list(item_metadata.values()))
        meta_df["i_idx"] = list(item_metadata.keys())
        cols = ["i_idx"] + [c for c in meta_df.columns if c != "i_idx"]
        meta_df[cols].to_csv(os.path.join(csv_dir, "metadata.csv"), index=False)

        if neg_df is not None:
            neg_df.to_csv(os.path.join(csv_dir, "disliked_interactions.csv"), index=False)

        logger.info(f"Inspection CSV files successfully exported to {csv_dir}")

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

    write_dataset_manifest(
        dataset_dir=dataset_dir,
        data_cfg=data_cfg,
        stats=stats,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        text_tensors=text_tensors,
        hard_negative_count=len(neg_df) if neg_df is not None else 0,
        manifest_path=args.manifest_path,
    )

    logger.info(f"All processed data and mappings saved to {processed_dir}")
    logger.info("Dataset statistics summary:")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")



if __name__ == "__main__":
    main()
