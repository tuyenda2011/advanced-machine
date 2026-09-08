"""Independently verify a saved recommendation checkpoint.

This checker rebuilds the graph and model from the saved run configuration,
scores top-K recommendations with explicit history/candidate masking, and
computes Recall/NDCG without calling the project's metric helper.  It is
intended for a post-training audit, so it reports a sampled audit by default
and never modifies the checkpoint or training artifacts.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph import get_norm_adj_tensor
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import build_user_history_features, load_training_text
from src.models.adaptive_gcl import AdaptiveGCL
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.training.early_stopping import load_checkpoint
from src.utils.checkpoints import get_model_output_dir, get_run_fingerprint
from src.utils.config import load_config
from src.utils.seed import set_seed


def _paths(model: str, density: float, seed: int, output_root: Path) -> tuple[Path, Path]:
    tag = f"s{int(density * 100)}"
    result = Path(get_model_output_dir("raw", model, str(output_root))) / f"{model}_{tag}_seed{seed}.json"
    checkpoint = Path(get_model_output_dir("checkpoints", model, str(output_root))) / f"{model}_{tag}_seed{seed}.pt"
    return result, checkpoint


def _build_model(model_name: str, config: dict, num_users: int, num_items: int, text_data, device):
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]
    if model_name == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif model_name == "xsimgcl":
        cfg = config["xsimgcl"]
        model = XSimGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            contrastive_weight=cfg["contrastive_weight"], temperature=cfg["temperature"],
            epsilon=cfg["epsilon"], contrastive_layer=cfg.get("contrastive_layer", 1),
        )
    elif model_name == "directau":
        cfg = config["directau"]
        model = DirectAU(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            gamma=cfg["gamma"], t=cfg["t"], profile=cfg.get("profile", "project_cosine"),
        )
    elif model_name == "adaptive_gcl":
        cfg = config.get("adaptive_gcl", {})
        text_features, item_mask = text_data
        user_history, user_mask = build_user_history_features(
            text_data[2], text_features, num_users, item_mask
        )
        model = AdaptiveGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            text_dim=text_features.shape[1], text_features=text_features,
            ssl_temp=cfg.get("ssl_temp", 0.2), ssl_reg=cfg.get("ssl_reg", 0.1),
            dirichlet_reg=cfg.get("dirichlet_reg", 0.0), node_dropout=cfg.get("node_dropout", 0.0),
            tau_plus=cfg.get("tau_plus", 0.0), user_history_features=user_history,
            item_text_mask=item_mask, user_text_mask=user_mask,
            use_item_text=cfg.get("use_item_text", True),
            user_semantic_weight=cfg.get("user_semantic_weight", 0.5),
            layer_aggregation=cfg.get("layer_aggregation", "learnable"),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def _dcg(hit_positions: list[int]) -> float:
    return float(sum(1.0 / np.log2(position + 2.0) for position in hit_positions))


def _independent_metrics(predictions: list[list[int]], targets: list[list[int]], k: int) -> dict[str, float]:
    if not predictions:
        return {f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0, f"MRR@{k}": 0.0}
    recalls: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    for pred, target in zip(predictions, targets):
        target_set = set(target)
        hits = [index for index, item in enumerate(pred[:k]) if item in target_set]
        recalls.append(len(hits) / max(1, len(target_set)))
        ideal = _dcg(list(range(min(k, max(1, len(target_set))))))
        ndcgs.append(_dcg(hits) / ideal if ideal > 0 else 0.0)
        reciprocal_ranks.append(1.0 / (hits[0] + 1) if hits else 0.0)
    return {
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"MRR@{k}": float(np.mean(reciprocal_ranks)),
    }


@torch.no_grad()
def _score_audit(model, users: torch.Tensor, items: torch.Tensor, history: dict[int, set[int]], candidates: set[int], targets: dict[int, list[int]], device: torch.device, k: int):
    model.eval()
    all_user, all_item = model._audit_embeddings  # assigned by main after checkpoint load
    excluded = sorted(set(range(items.shape[0])) - candidates)
    predictions: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_violations = 0
    repeat_rows = 0
    for user in users.tolist():
        eligible_count = len(candidates.difference(history.get(user, set())))
        if eligible_count < k:
            raise ValueError(
                f"User {user} has {eligible_count} eligible candidates; top-{k} requires {k}"
            )
        user_tensor = torch.tensor([user], dtype=torch.long, device=device)
        scores = model.get_user_rating_scores(user_tensor, all_user, all_item)[0].clone()
        if excluded:
            scores[torch.tensor(excluded, dtype=torch.long, device=device)] = float("-inf")
        seen = history.get(user, set())
        if seen:
            scores[torch.tensor(sorted(seen), dtype=torch.long, device=device)] = float("-inf")
        top = torch.topk(scores, k=k).indices.cpu().tolist()
        predictions.append(top)
        target_rows.append(targets.get(user, []))
        if len(set(top)) != len(top):
            repeat_rows += 1
        if any(item not in candidates or item in seen for item in top):
            mask_violations += 1
    return predictions, target_rows, mask_violations, repeat_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a saved recommendation checkpoint independently.")
    parser.add_argument("--model", required=True, choices=("lightgcn", "xsimgcl", "directau", "adaptive_gcl"))
    parser.add_argument("--output_root", required=True, help="Root containing raw/ and checkpoints/.")
    parser.add_argument("--density", "--sparsity", dest="density", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max_users", type=int, default=2048, help="Users to audit; use 0 for all users.")
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.density <= 1 or args.seed < 0 or args.max_users < 0:
        raise ValueError("Invalid density, seed or max_users")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = PROJECT_ROOT / config_dir
    result_path, checkpoint_path = _paths(args.model, args.density, args.seed, output_root)
    if not result_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing result or checkpoint: {result_path}, {checkpoint_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("effective_config") or load_config(args.model, str(config_dir))
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    processed_dir = Path(config["dataset"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    mappings = pickle.loads((processed_dir / "mappings.pkl").read_bytes())
    stats = mappings["stats"]
    num_users, num_items = int(stats["num_users"]), int(stats["num_items"])
    sparse_train = create_sparse_train_set(train_df, sparsity_ratio=args.density, seed=args.seed)
    candidates = set(sparse_train["i_idx"].unique().tolist())
    if args.split == "validation":
        history_df, target_df = train_df, val_df
    else:
        history_df, target_df = pd.concat([train_df, val_df], ignore_index=True), test_df
    target_df = target_df[target_df["i_idx"].isin(candidates)]
    history = history_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
    targets = target_df.groupby("u_idx")["i_idx"].apply(list).to_dict()
    users = sorted(targets)
    if args.max_users:
        users = users[:args.max_users]

    text_data = None
    if args.model == "adaptive_gcl":
        text_features, item_mask = load_training_text(str(processed_dir), mappings)
        text_data = (text_features.to(device), item_mask.to(device), sparse_train)
    model = _build_model(args.model, config, num_users, num_items, text_data, device)
    _, _, checkpoint = load_checkpoint(
        str(checkpoint_path), model, device=device, expected_fingerprint=None
    )
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_identity_ok = (
        checkpoint_config.get("model_name", args.model) == args.model
        and checkpoint.get("config", {}).get("experiment_fingerprint")
        == result.get("experiment_fingerprint")
    )
    norm_adj = get_norm_adj_tensor(sparse_train, num_users, num_items, device)
    with torch.no_grad():
        all_user, all_item = model(norm_adj)
    model._audit_embeddings = (all_user, all_item)
    predictions, target_rows, mask_violations, repeat_rows = _score_audit(
        model, torch.tensor(users), torch.arange(num_items), history, candidates, targets, device, k=20
    )
    metrics10 = _independent_metrics(predictions, target_rows, 10)
    metrics20 = _independent_metrics(predictions, target_rows, 20)
    current_fp = get_run_fingerprint(args.model, args.density, args.seed, config=config, config_dir=str(config_dir))
    current_fingerprint_matches = current_fp == result.get("experiment_fingerprint")
    integrity_ok = mask_violations == 0 and repeat_rows == 0 and checkpoint_identity_ok
    if not integrity_ok:
        verification_status = "fail"
    elif current_fingerprint_matches:
        verification_status = "pass"
    else:
        # The checkpoint can still be audited against its saved snapshot, but
        # a code change means this is not a current-code reproduction.
        verification_status = "historical_snapshot"
    report = {
        "model": args.model,
        "split": args.split,
        "density": args.density,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "result": str(result_path),
        "users_audited": len(users),
        "target_users_available": len(targets),
        "candidate_items": len(candidates),
        "mask_violations": mask_violations,
        "duplicate_recommendation_rows": repeat_rows,
        "independent_metrics": {**metrics10, **metrics20},
        "saved_metrics": result.get("val_metrics" if args.split == "validation" else "test_metrics", {}),
        "current_fingerprint": current_fp,
        "saved_fingerprint": result.get("experiment_fingerprint"),
        "checkpoint_fingerprint": checkpoint_config.get("experiment_fingerprint"),
        "checkpoint_identity_matches_result": checkpoint_identity_ok,
        "fingerprint_matches_current_code": current_fingerprint_matches,
        "status": verification_status,
    }
    output_path = output_root / "verification" / f"{args.model}_s{int(args.density * 100)}_seed{args.seed}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved verification report to {output_path}")
    return 0 if report["status"] in {"pass", "historical_snapshot"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
