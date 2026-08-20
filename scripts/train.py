import argparse
import json
import os
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
import torch

from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import Evaluator
from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("train_script")


def main():
    parser = argparse.ArgumentParser(description="Train Graph Recommendation Models (LightGCN, SGL, SimGCL)")
    parser.add_argument("--model", type=str, required=True, choices=["lightgcn", "sgl", "simgcl"], help="Model name")
    parser.add_argument("--sparsity", type=float, default=1.0, help="Sparsity ratio for training edges (0.25 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
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
    elif args.model == "sgl":
        sgl_cfg = config["sgl"]
        model = SGL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            ssl_weight=sgl_cfg["ssl_weight"],
            temperature=sgl_cfg["temperature"],
            drop_ratio=sgl_cfg["drop_ratio"],
        )
    elif args.model == "simgcl":
        sim_cfg = config["simgcl"]
        model = SimGCL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            contrastive_weight=sim_cfg["contrastive_weight"],
            temperature=sim_cfg["temperature"],
            epsilon=sim_cfg["epsilon"],
        )

    # 8. Train model
    sparsity_tag = f"s{int(args.sparsity * 100)}"
    checkpoint_dir = os.path.join("artifacts", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.pt")

    trainer = Trainer(model, train_df_sparse, val_evaluator, test_evaluator, config, device)
    results = trainer.train(checkpoint_path)

    # 9. Save run results to JSON
    results["sparsity_level"] = args.sparsity
    results["seed"] = args.seed

    results_dir = os.path.join("results", "raw")
    os.makedirs(results_dir, exist_ok=True)
    run_file = os.path.join(results_dir, f"{args.model}_{sparsity_tag}_seed{args.seed}.json")

    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved run results to {run_file}")


if __name__ == "__main__":
    main()
