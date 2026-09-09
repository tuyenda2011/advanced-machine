"""Create AdaptiveGCL metadata-group and text-eligibility reports.

This is an inspection step. It reads the active processed bundle and writes
derived CSVs; it never changes interaction splits or canonical metadata.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.adaptive_metadata import (
    ADAPTIVE_TEXT_POLICY,
    metadata_view_frame,
    semantic_ssl_mask,
)
from src.data.bundle import BundleError, resolve_bundle
from src.data.quality_report import warm_cohort
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import build_user_history_features, load_training_text


def _degree_stats(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = frame.groupby(column).size()
    if values.empty:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(values.mean()), "median": float(values.median())}


def build_group_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    item_metadata: dict[int, dict],
    *,
    num_users: int,
    num_items: int,
    text_features: torch.Tensor,
    item_mask: torch.Tensor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    view = metadata_view_frame(item_metadata)
    train_degree = train.groupby("i_idx").size()
    val_warm, val_cohort = warm_cohort(train, val)
    ssl_mask = torch.tensor(semantic_ssl_mask(item_metadata, num_items), dtype=torch.bool)
    view["train_degree"] = view["i_idx"].map(train_degree).fillna(0).astype(int)
    view["text_norm"] = text_features.norm(dim=-1).cpu().numpy()
    view["ssl_eligible"] = ssl_mask[view["i_idx"].to_numpy(copy=True)].numpy()

    rows = []
    groups = list(view.groupby("metadata_group", sort=True))
    for group, subset in groups:
        group_items = set(subset["i_idx"].tolist())
        val_group = val_warm[val_warm["i_idx"].isin(group_items)]
        rows.append(
            {
                "metadata_policy": ADAPTIVE_TEXT_POLICY,
                "metadata_group": group,
                "items": len(subset),
                "item_fraction": len(subset) / max(1, num_items),
                "ssl_eligible_items": int(subset["ssl_eligible"].sum()),
                "mean_train_degree": float(subset["train_degree"].mean()),
                "median_train_degree": float(subset["train_degree"].median()),
                "mean_text_norm": float(subset["text_norm"].mean()),
                "val_warm_targets": len(val_group),
                "val_warm_target_fraction": len(val_group) / max(1, len(val_warm)),
                "val_warm_user_count": int(val_group["u_idx"].nunique()),
            }
        )

    overall = {
        "metadata_policy": ADAPTIVE_TEXT_POLICY,
        "metadata_group": "__overall__",
        "items": num_items,
        "item_fraction": 1.0,
        "ssl_eligible_items": int(ssl_mask.sum()),
        "mean_train_degree": _degree_stats(train, "i_idx")["mean"],
        "median_train_degree": _degree_stats(train, "i_idx")["median"],
        "mean_text_norm": float(text_features.norm(dim=-1).mean().item()),
        "val_warm_targets": len(val_warm),
        "val_warm_target_fraction": 1.0,
        "val_warm_user_count": int(val_warm["u_idx"].nunique()),
        "val_total_targets": len(val),
        "val_cohort_json": json.dumps(val_cohort, sort_keys=True),
    }
    rows.append(overall)

    for ratio in (1.0, 0.25):
        sparse = create_sparse_train_set(train, ratio, seed=42)
        _, user_mask = build_user_history_features(
            sparse, text_features, num_users, item_mask
        )
        rows[-1][f"user_text_history_coverage_s{int(ratio * 100)}"] = float(user_mask.float().mean())
        rows[-1][f"user_text_history_users_s{int(ratio * 100)}"] = int(user_mask.sum())
    return pd.DataFrame(rows), view


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Bundle path or data/current.json")
    parser.add_argument(
        "--output_dir",
        help="Output root (default: active data/processed view)",
    )
    args = parser.parse_args(argv)
    try:
        bundle = resolve_bundle(args.bundle)
    except BundleError as exc:
        parser.error(str(exc))
    processed = bundle.train_dir
    output = Path(args.output_dir).resolve() if args.output_dir else (
        processed if bundle.legacy else ROOT / ".tmp" / "adaptivegcl_metadata" / bundle.build_id
    )
    output.mkdir(parents=True, exist_ok=True)
    with (processed / "mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    text, item_mask = load_training_text(processed, mappings)
    groups, view = build_group_report(
        train,
        val,
        mappings["item_metadata"],
        num_users=len(mappings["user2id"]),
        num_items=len(mappings["item2id"]),
        text_features=text,
        item_mask=item_mask,
    )
    reports = output / "reports"
    csv_dir = output / "csv"
    reports.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)
    groups.to_csv(reports / "adaptivegcl_metadata_groups.csv", index=False)
    view.to_csv(csv_dir / "adaptivegcl_metadata_view.csv", index=False)
    print(json.dumps({"output": str(output), "groups": len(groups), "items": len(view)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
