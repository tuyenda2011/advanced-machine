"""Audit processed recommendation data and fail on invalid benchmark inputs."""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.splitter import verify_no_leakage
from src.data.validation import validate_processed_interactions


PROCESSED_DIR = REPO_ROOT / "data" / "processed"
EXPECTED_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def main() -> int:
    errors = []
    warnings = []

    required = {
        "train": PROCESSED_DIR / "train.parquet",
        "validation": PROCESSED_DIR / "val.parquet",
        "test": PROCESSED_DIR / "test.parquet",
        "mappings": PROCESSED_DIR / "mappings.pkl",
        "text": PROCESSED_DIR / "item_text_embeddings.pt",
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

    num_users = len(mappings["user2id"])
    num_items = len(mappings["item2id"])
    all_rows = pd.concat(frames.values(), ignore_index=True)
    if set(all_rows["u_idx"].unique()) != set(range(num_users)):
        errors.append("user indices are not contiguous")
    if set(all_rows["i_idx"].unique()) != set(range(num_items)):
        errors.append("item indices are not contiguous")

    text_embeddings = torch.load(required["text"], map_location="cpu", weights_only=False)
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

    negative_path = PROCESSED_DIR / "disliked_interactions.parquet"
    if negative_path.exists():
        negatives = pd.read_parquet(negative_path)
        cutoffs = negatives["u_idx"].map(train_max)
        future_negatives = int((cutoffs.isna() | (negatives["timestamp"] > cutoffs)).sum())
        if future_negatives:
            errors.append(f"hard negatives contain {future_negatives} future interactions")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "counts": {name: len(frame) for name, frame in frames.items()},
        "ratios": actual_ratios,
        "num_users": num_users,
        "num_items": num_items,
        "text_embedding_shape": text_shape,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = PROCESSED_DIR / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== DATA AUDIT ===")
    print(json.dumps(report, indent=2))
    print(f"Audit report: {report_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
