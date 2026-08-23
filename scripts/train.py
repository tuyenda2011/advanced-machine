import argparse
from datetime import datetime
import json
import os
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 encoding for Windows Command Prompt/PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pickle
import pandas as pd
import torch

from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import Evaluator
from src.models.adaptive_gcl import AdaptiveGCL
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("train_script")


def append_to_model_results_csv(results: dict, model_name: str, sparsity: float, seed: int):
    """Save or append run results to dedicated per-model CSV file (results/aggregated/{model}_results.csv)."""
    agg_dir = os.path.join("results", "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    model_csv = os.path.join(agg_dir, f"{model_name}_results.csv")

    test_m = results.get("test_metrics", {})
    val_m = results.get("val_metrics", {})
    rep_m = results.get("representation_metrics", {})
    svd_m = results.get("svd_metrics", {})
    sub_m = results.get("subgroup_metrics", {})
    tail_m = sub_m.get("Tail (Cold-Start)", {})
    head_m = sub_m.get("Head (Active)", {})

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "sparsity": sparsity,
        "seed": seed,
        "best_epoch": results.get("best_epoch", 0),
        "total_epochs": results.get("total_epochs", 0),
        "train_time_sec": round(results.get("total_train_time", 0.0), 2),
        "inference_latency_ms": round(results.get("inference_latency_ms_per_user", 0.0), 3),
        # Accuracy Metrics
        "Recall@10": round(test_m.get("Recall@10", 0.0), 4),
        "NDCG@10": round(test_m.get("NDCG@10", 0.0), 4),
        "MRR@10": round(test_m.get("MRR@10", 0.0), 4),
        "Recall@20": round(test_m.get("Recall@20", 0.0), 4),
        "NDCG@20": round(test_m.get("NDCG@20", 0.0), 4),
        # Beyond-Accuracy Metrics
        "Diversity@10": round(test_m.get("Diversity@10", 0.0), 4),
        "Novelty@10": round(test_m.get("Novelty@10", 0.0), 4),
        "Coverage@10": round(test_m.get("Coverage@10", 0.0), 4),
        "Gini@10": round(test_m.get("Gini@10", 0.0), 4),
        # Representation Geometry
        "Alignment": round(rep_m.get("alignment", 0.0), 4),
        "Mean_Uniformity": round(rep_m.get("mean_uniformity", 0.0), 4),
        "User_Effective_Rank": round(svd_m.get("user_effective_rank", 0.0), 2),
        "Item_Effective_Rank": round(svd_m.get("item_effective_rank", 0.0), 2),
        # Subgroup
        "Tail_Recall@10": round(tail_m.get("Recall@10", 0.0), 4),
        "Tail_NDCG@10": round(tail_m.get("NDCG@10", 0.0), 4),
        "Head_Recall@10": round(head_m.get("Recall@10", 0.0), 4),
        "Head_NDCG@10": round(head_m.get("NDCG@10", 0.0), 4),
        "Val_NDCG@10": round(val_m.get("NDCG@10", 0.0), 4),
    }

    new_df = pd.DataFrame([row])
    if os.path.exists(model_csv):
        existing_df = pd.read_csv(model_csv)
        # Update row if exact same model, sparsity, seed exists, else append
        mask = (existing_df["sparsity"] == sparsity) & (existing_df["seed"] == seed)
        if mask.any():
            existing_df = existing_df[~mask]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df.to_csv(model_csv, index=False)
    logger.info(f"Updated dedicated model results file: {model_csv}")


def main():
    parser = argparse.ArgumentParser(description="Train Graph Recommendation Models (LightGCN, XSimGCL, DirectAU, AdaptiveGCL)")
    parser.add_argument("--model", type=str, required=True, choices=["lightgcn", "xsimgcl", "directau", "adaptive_gcl"], help="Model name (4 SOTA models)")
    parser.add_argument("--sparsity", type=float, default=1.0, help="Sparsity ratio for training edges (0.25 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest saved checkpoint")
    parser.add_argument("--config_dir", type=str, default="configs", help="Config directory")
    args = parser.parse_args()

    # 1. Set seed
    set_seed(args.seed)

    # 2. Get device
    device = get_device()

    # 3. Load config
    config = load_config(args.model, args.config_dir)
    config["training"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    # 4. Load dataset processed files
    processed_dir = config["dataset"]["processed_dir"]
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    mappings_path = os.path.join(processed_dir, "mappings.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed dataset not found at {processed_dir}. Run prepare_data.py first.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    stats = mappings["stats"]
    num_users = stats["num_users"]
    num_items = stats["num_items"]

    # 5. Apply sparsity sampling to training set (Validation & Test stay 100% fixed)
    train_df_sparse = create_sparse_train_set(train_df, sparsity_ratio=args.sparsity, seed=args.seed)

    # 6. Initialize evaluators
    top_k_list = config["evaluation"]["top_k"]
    val_evaluator = Evaluator(train_df_sparse, val_df, num_users, num_items, k_list=top_k_list)
    test_evaluator = Evaluator(train_df_sparse, test_df, num_users, num_items, k_list=top_k_list)

    # 7. Instantiate model
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]

    if args.model == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif args.model == "xsimgcl":
        xsim_cfg = config["xsimgcl"]
        model = XSimGCL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            contrastive_weight=xsim_cfg["contrastive_weight"],
            temperature=xsim_cfg["temperature"],
            epsilon=xsim_cfg["epsilon"],
        )
    elif args.model == "directau":
        dau_cfg = config["directau"]
        model = DirectAU(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            gamma=dau_cfg["gamma"],
            t=dau_cfg["t"],
        )
    elif args.model == "adaptive_gcl":
        ada_cfg = config.get("adaptive_gcl", {})
        text_emb_path = os.path.join(processed_dir, "item_text_embeddings.pt")
        text_features = None
        if os.path.exists(text_emb_path):
            logger.info(f"Loading item text features from {text_emb_path} for AdaptiveGCL...")
            text_features = torch.load(text_emb_path, map_location="cpu", weights_only=False)
            text_dim = text_features.shape[1]
        else:
            text_dim = ada_cfg.get("text_dim", 384)
            logger.warning(f"Item text features not found at {text_emb_path}. Using fallback zero tensor.")

        model = AdaptiveGCL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            text_dim=text_dim,
            text_features=text_features,
            ssl_temp=ada_cfg.get("ssl_temp", 0.2),
            ssl_reg=ada_cfg.get("ssl_reg", 0.1),
            dirichlet_reg=ada_cfg.get("dirichlet_reg", 0.01),
            node_dropout=ada_cfg.get("node_dropout", 0.0),
        )


    # 8. Train model
    sparsity_tag = f"s{int(args.sparsity * 100)}"
    checkpoint_dir = os.path.join("results", "checkpoints", args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.pt")

    trainer = Trainer(model, train_df_sparse, val_evaluator, test_evaluator, config, device)
    results = trainer.train(checkpoint_path, resume=args.resume)

    # 9. Save run results to JSON
    results["sparsity_level"] = args.sparsity
    results["seed"] = args.seed

    results_dir = os.path.join("results", "raw", args.model)
    os.makedirs(results_dir, exist_ok=True)
    run_file = os.path.join(results_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.json")

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved run results to {run_file}")

    # 10. Save / append to dedicated per-model CSV file (results/aggregated/{model}_results.csv)
    append_to_model_results_csv(results, args.model, args.sparsity, args.seed)

    # 11. Update Global Best Model if this run achieved the highest test NDCG@10
    global_best_meta_path = os.path.join(checkpoint_dir, f"{args.model}_best_meta.json")
    global_best_pt_path = os.path.join(checkpoint_dir, f"{args.model}_best.pt")
    
    current_test_ndcg = results.get("test_metrics", {}).get("NDCG@10", 0.0)
    is_new_global_best = True

    if os.path.exists(global_best_meta_path):
        try:
            with open(global_best_meta_path, "r", encoding="utf-8") as f:
                prev_best = json.load(f)
            if prev_best.get("NDCG@10", 0.0) >= current_test_ndcg:
                is_new_global_best = False
        except Exception:
            is_new_global_best = True

    if is_new_global_best and os.path.exists(checkpoint_path):
        import shutil
        shutil.copyfile(checkpoint_path, global_best_pt_path)
        with open(global_best_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "sparsity": args.sparsity,
                "seed": args.seed,
                "best_epoch": results.get("best_epoch", 0),
                "NDCG@10": current_test_ndcg,
                "Recall@10": results.get("test_metrics", {}).get("Recall@10", 0.0),
                "Diversity@10": results.get("test_metrics", {}).get("Diversity@10", 0.0),
                "Novelty@10": results.get("test_metrics", {}).get("Novelty@10", 0.0),
                "source_checkpoint": checkpoint_path,
            }, f, indent=2)
        logger.info(f"🏆 Updated GLOBAL BEST MODEL for {args.model.upper()} -> {global_best_pt_path} (Test NDCG@10: {current_test_ndcg:.4f})")


if __name__ == "__main__":
    main()
