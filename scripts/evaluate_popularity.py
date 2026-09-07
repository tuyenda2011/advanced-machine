"""Evaluate MostPopular without training; validation is the default split."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.sparsity import create_sparse_train_set
from src.evaluation.evaluator import EVALUATION_PROTOCOL, Evaluator
from src.evaluation.metrics import compute_topk_metrics
from src.evaluation.popularity import popularity_predictions
from src.utils.checkpoints import get_run_fingerprint
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.sparsity <= 1 or args.seed < 0:
        parser.error("Invalid sparsity or seed")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = load_config("lightgcn", args.config_dir)
    processed = Path(config["dataset"]["processed_dir"])
    train = pd.read_parquet(processed / "train.parquet")
    targets = pd.read_parquet(processed / f"{args.split}.parquet")
    with (processed / "mappings.pkl").open("rb") as handle:
        stats = pickle.load(handle)["stats"]
    sparse = create_sparse_train_set(train, args.sparsity, args.seed)
    candidates = set(sparse.i_idx.unique())
    history = train
    if args.split == "test":
        history = pd.concat([train, pd.read_parquet(processed / "val.parquet")], ignore_index=True)
    evaluator = Evaluator(history, targets[targets.i_idx.isin(candidates)],
                          stats["num_users"], stats["num_items"],
                          k_list=config["evaluation"]["top_k"],
                          candidate_items=candidates, popularity_df=train)
    predictions = popularity_predictions(sparse, evaluator)
    result = {"model_name": "mostpopular", "split": args.split, "seed": args.seed,
              "sparsity": args.sparsity, "evaluation_protocol": EVALUATION_PROTOCOL,
              "candidate_items": len(candidates), "users": len(evaluator.eval_users),
              "reference_fingerprint": get_run_fingerprint("lightgcn", args.sparsity, args.seed, config, args.config_dir),
              "baseline_code_sha256": hashlib.sha256(Path(__file__).read_bytes() + Path("src/evaluation/popularity.py").read_bytes()).hexdigest(),
              "policy": "sparse_train_unique_users; item_id_ascending_ties",
              "metrics": compute_topk_metrics(evaluator.ground_truth, predictions, evaluator.k_list)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
