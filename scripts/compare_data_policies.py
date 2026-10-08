"""Paired, validation-only 2x2 pilot of brand text policy and SSL eligibility."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.ablate_adaptive import build_model
from src.data.bundle import resolve_bundle
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import get_ssl_item_mask, load_text_view
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer
from src.utils.checkpoints import get_run_fingerprint
from src.utils.config import load_config
from src.utils.seed import set_seed

VARIANTS = {
    "shared_usable": ("shared", "usable"),
    "shared_specific": ("shared", "item_specific"),
    "quality_usable": ("adaptivegcl_quality", "usable"),
    "quality_specific": ("adaptivegcl_quality", "item_specific"),
}


def paired_summary(rows):
    comparisons = {}
    pairs = {
        "brand_with_usable_ssl": ("shared_usable", "quality_usable"),
        "brand_with_specific_ssl": ("shared_specific", "quality_specific"),
        "mask_with_shared_text": ("shared_usable", "shared_specific"),
        "mask_with_quality_text": ("quality_usable", "quality_specific"),
    }
    for density in sorted({row["density"] for row in rows}):
        selected = [row for row in rows if row["density"] == density]
        for name, (reference, candidate) in pairs.items():
            left = {row["seed"]: row for row in selected if row["variant"] == reference}
            right = {row["seed"]: row for row in selected if row["variant"] == candidate}
            seeds = sorted(left.keys() & right.keys())
            if not seeds:
                continue
            metrics = {}
            for metric in ("NDCG@20", "Recall@20"):
                delta = [right[seed]["val_metrics"][metric] - left[seed]["val_metrics"][metric] for seed in seeds]
                metrics[metric] = {
                    "paired_deltas": delta, "mean_delta": float(np.mean(delta)),
                    "sd_delta": float(np.std(delta, ddof=1)) if len(delta) > 1 else None,
                    "positive_seeds": sum(value > 0 for value in delta),
                }
            comparisons[f"density_{density}/{name}"] = {"seeds": seeds, "reference": reference, "candidate": candidate, "metrics": metrics}
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--output", default="results/experiments/data_quality_20261008")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.25])
    args = parser.parse_args(argv)
    if args.epochs < 1 or any(seed < 0 for seed in args.seeds) or any(not 0 < d <= 1 for d in args.densities):
        parser.error("Invalid epoch budget, seed or density")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(4)
    bundle = resolve_bundle(args.bundle)
    with bundle.artifact("mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    # Only positive train and validation targets enter the experiment.
    train = pd.read_parquet(bundle.split_path("train"))
    val = pd.read_parquet(bundle.split_path("val"))
    users, items = len(mappings["user2id"]), len(mappings["item2id"])
    views = {name: load_text_view(bundle.train_dir, mappings, name) for name in ("shared", "adaptivegcl_quality")}
    base_config = load_config("adaptive_gcl")
    base_config["training"].update(epochs=args.epochs, early_stopping_patience=args.epochs + 1)
    base_config["validation_only"] = True
    base_config["evaluation"]["model_diagnostics"] = False
    protocol = {
        "kind": "exploratory_validation_pilot", "validation_only": True,
        "bundle_id": bundle.build_id, "seeds": args.seeds, "densities": args.densities,
        "epochs": args.epochs, "variants": VARIANTS, "monitor": "NDCG@20",
        "decision_rule": "Keep shared/usable for the main benchmark until evidence beyond this short pilot supports a policy change",
        "scope": "paired training-seed variation on this fixed split; not confirmatory evidence or final test performance",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows = []
    for density in args.densities:
        for seed in args.seeds:
            sparse = create_sparse_train_set(train, density, seed=seed)
            candidates = set(sparse["i_idx"].unique())
            warm_val = val[val["i_idx"].isin(candidates)]
            for variant, (view, eligibility) in VARIANTS.items():
                set_seed(seed)
                run = output / variant / f"s{int(density * 100)}_seed{seed}"
                run.mkdir(parents=True)
                config = deepcopy(base_config)
                config["training"]["seed"] = seed
                config["adaptive_gcl"].update(feature_view=view, ssl_eligibility=eligibility)
                config["history_dir"] = str(run / "history")
                config["experiment_fingerprint"] = get_run_fingerprint("adaptive_gcl", density, seed, config, manifest_path=bundle.manifest_path)
                features, mask, _, _ = views[view]
                ssl = get_ssl_item_mask(mappings["item_metadata"], mask, feature_view=view, eligibility=eligibility)
                model = build_model(config, mappings, sparse, features, mask, ssl)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                evaluator = Evaluator(train, warm_val, users, items, k_list=[10, 20], candidate_items=candidates, popularity_df=train)
                trainer = Trainer(model, sparse, evaluator, None, config, device, user_disliked_items=mappings.get("user_disliked_items", {}), subgroup_reference_df=train)
                print(f"START {variant} density={density} seed={seed}", flush=True)
                (run / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
                result = trainer.train(str(run / "best.pt"))
                result.update(variant=variant, density=density, seed=seed, ssl_eligible_items=int(ssl.sum()), bundle_id=bundle.build_id)
                (run / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                rows.append(result)
                (output / "summary.json").write_text(json.dumps({"protocol": protocol, "completed_runs": len(rows), "planned_runs": len(VARIANTS) * len(args.seeds) * len(args.densities), "comparisons": paired_summary(rows), "results": rows}, indent=2) + "\n", encoding="utf-8")
                del trainer, evaluator, model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print(f"COMPLETE {len(rows)} runs; test targets were not used", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
