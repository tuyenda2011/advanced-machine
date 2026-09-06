"""Audit processed recommendation data and fail on invalid benchmark inputs."""

import json
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import summarize_metadata_quality
from src.data.provenance import sha256_file
from src.data.sparsity import create_sparse_train_set
from src.data.splitter import summarize_split_timing, verify_no_leakage
from src.data.text_encoder import get_text_input_fingerprint, load_verified_text_cache, get_item_text_mask
from src.data.validation import validate_processed_interactions


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EXPECTED_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def main(processed_dir=None, manifest_path=None) -> int:
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    errors = []
    warnings = []

    required = {
        "train": processed_dir / "train.parquet",
        "validation": processed_dir / "val.parquet",
        "test": processed_dir / "test.parquet",
        "mappings": processed_dir / "mappings.pkl",
        "text": processed_dir / "item_text_embeddings.pt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        print(f"AUDIT FAILED: missing artifacts: {missing}")
        return 1

    frames = {
        "train": pd.read_parquet(required["train"]),
        "validation": pd.read_parquet(required["validation"]),
        "test": pd.read_parquet(required["test"]),
    }
    with open(required["mappings"], "rb") as file:
        mappings = pickle.load(file)
    with open(manifest_path or REPO_ROOT / "data" / "manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)

    for name, frame in frames.items():
        violations = validate_processed_interactions(frame, raise_on_error=False)
        errors.extend(f"{name}: {violation}" for violation in violations)

    _, leakage = verify_no_leakage(
        frames["train"], frames["validation"], frames["test"], raise_on_error=False
    )
    if any(leakage.values()):
        errors.append(f"split overlap detected: {leakage}")

    total = sum(len(frame) for frame in frames.values())
    actual_ratios = {name: len(frame) / total for name, frame in frames.items()}
    for name, expected in EXPECTED_RATIOS.items():
        if abs(actual_ratios[name] - expected) > 1 / total:
            errors.append(
                f"{name} ratio is {actual_ratios[name]:.6f}, expected {expected:.6f}"
            )

    train = frames["train"]
    train_users = set(train["u_idx"])
    train_items = set(train["i_idx"])
    for name in ("validation", "test"):
        frame = frames[name]
        unseen_users = set(frame["u_idx"]) - train_users
        if unseen_users:
            errors.append(f"{name}: {len(unseen_users)} users absent from train")
        cold_targets = int((~frame["i_idx"].isin(train_items)).sum())
        if cold_targets:
            warnings.append(
                f"{name}: {cold_targets} cold-start targets excluded from warm-start metrics"
            )

    train_max = train.groupby("u_idx")["timestamp"].max()
    for name in ("validation", "test"):
        eval_min = frames[name].groupby("u_idx")["timestamp"].min()
        violations = int((train_max.loc[eval_min.index] > eval_min).sum())
        if violations:
            errors.append(f"{name}: {violations} users violate chronological ordering")
    timing = summarize_split_timing(train, frames["validation"], frames["test"])
    if timing["validation_test_order_violations"]:
        errors.append("validation timestamps exceed test timestamps")
    if not timing["strict_temporal_order"]:
        warnings.append("Timestamp ties cross split boundaries: chronology is non-decreasing, not strict future prediction")

    num_users = len(mappings["user2id"])
    num_items = len(mappings["item2id"])
    all_rows = pd.concat(frames.values(), ignore_index=True)
    if set(all_rows["u_idx"].unique()) != set(range(num_users)):
        errors.append("user indices are not contiguous")
    if set(all_rows["i_idx"].unique()) != set(range(num_items)):
        errors.append("item indices are not contiguous")
    for column, mapping in (("u_idx", "user2id"), ("i_idx", "item2id")):
        if set(mappings[mapping].values()) != set(range(len(mappings[mapping]))):
            errors.append(f"{mapping}: mapping values are not contiguous and unique")
    core_cfg = manifest.get("preprocessing", {})
    for column, setting in (("u_idx", "min_user_interactions"), ("i_idx", "min_item_interactions")):
        if all_rows.groupby(column).size().min() < core_cfg.get(setting, 5):
            errors.append(f"full positive graph violates {setting}")

    metadata_quality = summarize_metadata_quality(mappings["item_metadata"])
    statistics = manifest.get("statistics", {})
    for source, ledger in statistics.get("ingestion_ledger", {}).items():
        if ledger["total_lines"] != ledger["parsed_rows"] + ledger["blank_lines"] + ledger["invalid_lines"]:
            errors.append(f"{source}: ingestion counts do not reconcile")
        if ledger["invalid_lines"]:
            warnings.append(f"{source}: {ledger['invalid_lines']} invalid source lines excluded; inspect ingestion ledger")
    ledger = statistics.get("filter_ledger")
    if ledger:
        if ledger["input_ratings"] != ledger["removed_by_rating"] + ledger["positive_ratings"] or ledger["positive_ratings"] != ledger["removed_duplicates"] + ledger["after_dedup"]:
            errors.append("Filter ledger rating/dedup counts do not reconcile")
        remaining = ledger["after_dedup"]
        for step in ledger["kcore_rounds"]:
            if step["before"] != remaining or step["before"] != step["removed"] + step["after"]:
                errors.append("K-core round counts do not reconcile")
            remaining = step["after"]
        if remaining != total or ledger["output_interactions"] != total:
            errors.append("Filter ledger output does not match split sizes")
    try:
        item_mask = get_item_text_mask(mappings["item_metadata"], num_items)
    except ValueError as exc:
        errors.append(str(exc))
        item_mask = None
    if metadata_quality["missing_or_fallback_titles"]:
        warnings.append(f"Metadata: {metadata_quality['missing_or_fallback_titles']} titles are missing/fallback")
    if metadata_quality["missing_or_unknown_brands"]:
        warnings.append(f"Metadata: {metadata_quality['missing_or_unknown_brands']} brands are missing/unknown")

    text_embeddings = torch.load(required["text"], map_location="cpu", weights_only=True)
    if not isinstance(text_embeddings, torch.Tensor):
        errors.append("text embedding artifact is not a tensor")
        text_shape = None
    else:
        text_shape = list(text_embeddings.shape)
        if text_embeddings.ndim != 2 or text_embeddings.shape[0] != num_items:
            errors.append(
                f"text embedding shape {text_shape} does not match {num_items} items"
            )
        if not torch.isfinite(text_embeddings).all():
            errors.append("text embeddings contain NaN or Inf")
    text_cfg = manifest.get("text_features", {})
    try:
        fingerprint = get_text_input_fingerprint(
            mappings["item_metadata"], num_items,
            text_cfg.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"),
            text_cfg.get("revision"),
        )
    except ValueError as exc:
        errors.append(str(exc))
        fingerprint = "invalid"
    if load_verified_text_cache(required["text"], fingerprint, item_mask=item_mask) is None:
        errors.append("Text cache has missing/stale provenance, invalid norms, or corrupted contents")
    artifacts = manifest.get("artifacts", [])
    expected_names = {path.name for path in required.values()} | {"item_text_embeddings.pt.json"}
    if not expected_names.issubset({Path(entry["path"]).name for entry in artifacts}):
        errors.append("Manifest is missing required processed-artifact hashes")
    for artifact in artifacts:
        path = REPO_ROOT / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            errors.append(f"Artifact digest mismatch: {artifact['path']}")

    negative_path = processed_dir / "disliked_interactions.parquet"
    if negative_path.exists():
        negatives = pd.read_parquet(negative_path)
        cutoffs = negatives["u_idx"].map(train_max)
        future_negatives = int((cutoffs.isna() | (negatives["timestamp"] > cutoffs)).sum())
        if future_negatives:
            errors.append(f"hard negatives contain {future_negatives} future interactions")
        if not set(negatives["u_idx"]).issubset(range(num_users)) or not set(negatives["i_idx"]).issubset(range(num_items)):
            errors.append("Explicit negatives contain unmapped IDs")
        if not negatives["rating"].between(1, 2).all():
            errors.append("Explicit negatives contain ratings outside [1, 2]")
        positive_pairs = pd.MultiIndex.from_frame(all_rows[["u_idx", "i_idx"]])
        negative_pairs = pd.MultiIndex.from_frame(negatives[["u_idx", "i_idx"]])
        if negative_pairs.isin(positive_pairs).any():
            errors.append("Explicit dislikes overlap positive interactions")
        actual_map = negatives.groupby("u_idx")["i_idx"].apply(set).to_dict()
        saved_map = {user: set(items) for user, items in mappings.get("user_disliked_items", {}).items()}
        if actual_map != saved_map:
            errors.append("Explicit dislike table disagrees with mappings.pkl")
        timing["negatives_at_training_cutoff"] = int((negatives["timestamp"] == cutoffs).sum())

    sparsity_report = {}
    for ratio in (1.0, 0.75, 0.5, 0.25):
        sparse = create_sparse_train_set(train, ratio, seed=42)
        coverage_ok = set(sparse["u_idx"]) == train_users and set(sparse["i_idx"]) == train_items
        count_ok = len(sparse) == round(len(train) * ratio)
        sparsity_report[str(ratio)] = {"edges": len(sparse), "coverage_preserved": coverage_ok, "exact_count": count_ok}
        if item_mask is not None:
            valid_users = set(sparse.loc[item_mask.numpy()[sparse["i_idx"].to_numpy()], "u_idx"])
            sparsity_report[str(ratio)]["users_without_usable_text_history"] = num_users - len(valid_users)
        if not coverage_ok or not count_ok:
            errors.append(f"Sparsity {ratio}: failed coverage or exact count")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "counts": {name: len(frame) for name, frame in frames.items()},
        "ratios": actual_ratios,
        "num_users": num_users,
        "num_items": num_items,
        "text_embedding_shape": text_shape,
        "temporal_audit": timing,
        "metadata_quality": metadata_quality,
        "sparsity": sparsity_report,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = processed_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== DATA AUDIT ===")
    print(json.dumps(report, indent=2))
    print(f"Audit report: {report_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir")
    parser.add_argument("--manifest_path")
    args = parser.parse_args()
    sys.exit(main(args.processed_dir, args.manifest_path))
