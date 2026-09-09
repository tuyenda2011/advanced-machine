"""Validation-only AdaptiveGCL ablations. Dry-run by default; pass --run to train."""

import argparse
import gc
import hashlib
import json
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.data.bundle import BundleError, resolve_bundle
from src.data.provenance import sha256_file
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import (
    build_user_history_features,
    load_text_view,
    load_training_text,
)
from src.evaluation.evaluator import Evaluator
from src.models.adaptive_gcl import AdaptiveGCL
from src.training.trainer import Trainer
from src.utils.checkpoints import get_experiment_fingerprint
from src.utils.config import load_config
from src.utils.config_schemas import validate_model_config
from src.utils.paths import resolve_output_root, write_run_manifest
from src.utils.seed import set_seed

# One factor at a time, relative to the loaded full configuration.
VARIANTS = {
    "previous_bounded": {"layer_aggregation": "learnable", "user_semantic_gate": False, "ssl_target": "projected"},
    "without_layer_anchor": {"layer_aggregation": "learnable"},
    "without_user_gate": {"user_semantic_gate": False},
    "projected_ssl_target": {"ssl_target": "projected"},
    "bounded_residual": {"fusion_mode": "bounded_residual", "residual_alpha_init": 0.1},
    "bounded_residual_ssl_001": {
        "fusion_mode": "bounded_residual", "residual_alpha_init": 0.1, "ssl_reg": 0.01,
    },
    "full": {},
    "no_item_text": {"use_item_text": False},
    "no_user_text": {"user_semantic_weight": 0.0},
    "no_ssl": {"ssl_reg": 0.0},
    "ssl_001": {"ssl_reg": 0.01},
    "ssl_003": {"ssl_reg": 0.03},
    "residual_alpha_01": {
        "fusion_mode": "residual",
        "residual_alpha_init": 0.1,
    },
    "residual_alpha_01_ssl_001": {
        "fusion_mode": "residual",
        "residual_alpha_init": 0.1,
        "ssl_reg": 0.01,
    },
    "mean_layers": {"layer_aggregation": "mean"},
    "user_weight_025": {"user_semantic_weight": 0.25},
    "mlp_decay_1e4": {"mlp_weight_decay": 1e-4},
    "no_dislikes": {"hard_neg_alpha": 0.0},
    "no_all_text": {
        "use_item_text": False,
        "user_semantic_weight": 0.0,
        "ssl_reg": 0.0,
    },
    "no_dropout": {"node_dropout": 0.0},
    "interaction_only": {"use_item_text": False, "user_semantic_weight": 0.0,
                         "ssl_reg": 0.0, "layer_aggregation": "mean",
                         "node_dropout": 0.0, "hard_neg_alpha": 0.0,
                         "dirichlet_reg": 0.0, "tau_plus": 0.0},
}


def variant_config(base: dict, variant: str) -> dict:
    config = deepcopy(base)
    config["adaptive_gcl"].update(VARIANTS[variant])
    return validate_model_config(config, "adaptive_gcl")


def build_model(config, mappings, sparse, features, mask, ssl_mask=None):
    users, items = len(mappings["user2id"]), len(mappings["item2id"])
    profiles, user_mask = build_user_history_features(sparse, features, users, mask)
    ada = config["adaptive_gcl"]
    return AdaptiveGCL(
        users,
        items,
        embedding_dim=config["model"]["embedding_dim"],
        num_layers=config["model"]["num_layers"],
        text_dim=features.shape[1],
        text_features=features,
        ssl_item_mask=ssl_mask if ssl_mask is not None else mask,
        user_history_features=profiles,
        item_text_mask=mask,
        user_text_mask=user_mask,
        ssl_temp=ada["ssl_temp"],
        ssl_reg=ada["ssl_reg"],
        dirichlet_reg=ada["dirichlet_reg"],
        node_dropout=ada["node_dropout"],
        tau_plus=ada["tau_plus"],
        use_item_text=ada["use_item_text"],
        user_semantic_weight=ada["user_semantic_weight"],
        layer_aggregation=ada["layer_aggregation"],
        user_semantic_gate=ada.get("user_semantic_gate", False),
        ssl_target=ada.get("ssl_target", "projected"),
        fusion_mode=ada.get("fusion_mode", "convex"),
        residual_alpha_init=ada.get("residual_alpha_init", 0.1),
        residual_alpha_max=ada.get("residual_alpha_max", 1.0),
    )


def run(args):
    base = load_config("adaptive_gcl", config_dir=args.config_dir)
    planned = {name: variant_config(base, name) for name in args.variants}
    for name, config in planned.items():
        if name != "full" and config["adaptive_gcl"] == planned["full"]["adaptive_gcl"]:
            raise ValueError(
                f"{name} equals full: choose a meaningful reference configuration"
            )
    print(
        json.dumps(
            {
                "runs": len(planned) * len(args.sparsities) * len(args.seeds),
                "epochs_per_run": args.epochs,
                "seeds": args.seeds,
                "sparsities": args.sparsities,
                "validation_only": True,
                "variants": {
                    name: cfg["adaptive_gcl"] for name, cfg in planned.items()
                },
            },
            indent=2,
        )
    )
    if not args.run:
        print("Dry-run only. Pass --run to train; no data or artifacts were written.")
        return []

    if args.output_dir:
        output = resolve_output_root(args.output_dir, kind="ablation")
    else:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "results" / "experiments" / "adaptivegcl_early_decline" / stamp
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ablation output: {output}")
    configured_processed = Path(base["dataset"].get("processed_dir", "data/processed"))
    if not configured_processed.is_absolute():
        configured_processed = (ROOT / configured_processed).resolve()
    use_active_bundle = configured_processed == (ROOT / "data" / "processed").resolve()
    try:
        bundle = resolve_bundle(args.bundle) if args.bundle or (use_active_bundle and (ROOT / "data" / "current.json").exists()) else None
    except BundleError as exc:
        raise RuntimeError(f"Dataset bundle is invalid: {exc}") from exc
    processed = bundle.train_dir if bundle is not None else Path(base["dataset"]["processed_dir"])
    # Only load trusted, locally prepared mappings. Test split is never loaded.
    with (processed / "mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    feature_view = base.get("adaptive_gcl", {}).get("feature_view", "shared")
    if feature_view == "shared":
        features, mask = load_training_text(processed, mappings)
        ssl_mask = mask.clone()
        feature_hashes = {}
    else:
        features, mask, ssl_mask, _view_metadata = load_text_view(
            processed, mappings, feature_view
        )
        feature_path = processed / "adaptivegcl_text_embeddings.pt"
        feature_hashes = {
            "adaptivegcl_text_embeddings.pt": sha256_file(feature_path),
            "adaptivegcl_text_embeddings.pt.json": sha256_file(
                Path(str(feature_path) + ".json")
            ),
        }
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    data_hashes = {
        name: sha256_file(processed / name)
        for name in (
            "train.parquet",
            "val.parquet",
            "mappings.pkl",
            "item_text_embeddings.pt",
            "item_text_embeddings.pt.json",
        )
    }
    data_hashes.update(feature_hashes)
    code_hash = get_experiment_fingerprint(
        "adaptive_gcl",
        config_dir=args.config_dir,
        manifest_path=bundle.manifest_path if bundle is not None else None,
    )
    runner_hash = sha256_file(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = write_run_manifest(
        output,
        kind="ablation",
        metadata={
            "variants": list(planned),
            "densities": list(args.sparsities),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "validation_only": True,
        },
    )
    root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root_manifest.setdefault("parameters", {}).update(
        {
            "feature_view": feature_view,
            "bundle_id": bundle.build_id if bundle is not None else None,
            "manifest_sha256": (
                sha256_file(bundle.manifest_path) if bundle is not None else None
            ),
            "code_sha256": code_hash,
            "runner_sha256": runner_hash,
            "data_sha256": data_hashes,
        }
    )
    manifest_path.write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in args.seeds:
        for ratio in args.sparsities:
            sparse = create_sparse_train_set(train, ratio, seed=seed)
            candidates = set(sparse["i_idx"].unique())
            evaluator = Evaluator(
                train,
                val[val["i_idx"].isin(candidates)],
                len(mappings["user2id"]),
                len(mappings["item2id"]),
                k_list=[10, 20],
                candidate_items=candidates,
                popularity_df=train,
                batch_size=base["evaluation"]["eval_batch_size"],
            )
            for name, template in planned.items():
                set_seed(seed)
                config = deepcopy(template)
                config["training"].update(epochs=args.epochs, seed=seed)
                config.setdefault("evaluation", {})["model_diagnostics"] = True
                config["adaptive_gcl"]["feature_view"] = feature_view
                identity = {
                    "config": deepcopy(config),
                    "variant": name,
                    "sparsity": ratio,
                    "code_sha256": code_hash,
                    "runner_sha256": runner_hash,
                    "data_sha256": data_hashes,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest()
                run_dir = output / name / f"s{int(ratio * 100)}_seed{seed}"
                run_dir.mkdir(parents=True)
                config.update(
                    validation_only=True,
                    experiment_fingerprint=fingerprint,
                    history_dir=str(run_dir / "history"),
                    ablation_variant=name,
                )
                (run_dir / "identity.json").write_text(
                    json.dumps(identity, indent=2), encoding="utf-8"
                )
                (run_dir / "config.json").write_text(
                    json.dumps(config, indent=2), encoding="utf-8"
                )
                model = build_model(config, mappings, sparse, features, mask, ssl_mask)
                trainer = Trainer(
                    model,
                    sparse,
                    evaluator,
                    None,
                    config,
                    device,
                    user_disliked_items=mappings.get("user_disliked_items", {}),
                    subgroup_reference_df=train,
                )
                print(
                    f"ABLATION {name} sparsity={ratio} seed={seed} -> {run_dir}",
                    flush=True,
                )
                result = trainer.train(str(run_dir / "best.pt"))
                result.update(
                    variant=name,
                    sparsity=ratio,
                    seed=seed,
                    fingerprint=fingerprint,
                    fusion_mode=config["adaptive_gcl"].get("fusion_mode", "convex"),
                    residual_alpha=(
                        float(model.residual_alpha.item())
                        if model.residual_alpha is not None
                        else None
                    ),
                    validation_users=len(evaluator.eval_users),
                    requested_epochs=args.epochs,
                )
                (run_dir / "result.json").write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )
                results.append(result)
                (output / "summary.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
                _write_experiment_tables(output, results)
                del trainer, model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return results


def _write_experiment_tables(output: Path, results: list[dict]) -> None:
    """Write review-friendly comparisons without touching the data bundle."""
    rows = []
    diagnostic_rows = []
    for result in results:
        val = result.get("val_metrics", {})
        history = result.get("history", [])
        last = history[-1] if history else {}
        rows.append(
            {
                "variant": result.get("variant"),
                "fusion_mode": result.get("fusion_mode"),
                "residual_alpha": result.get("residual_alpha"),
                "seed": result.get("seed"),
                "sparsity": result.get("sparsity"),
                "best_epoch": result.get("best_epoch"),
                "best_val_ndcg20": val.get("NDCG@20"),
                "recall10": val.get("Recall@10"),
                "ndcg10": val.get("NDCG@10"),
                "last_val_ndcg20": last.get("val_ndcg_20"),
                "total_train_time": result.get("total_train_time"),
                "fingerprint": result.get("fingerprint"),
            }
        )
        for record in history:
            diagnostic_rows.append(
                {
                    "variant": result.get("variant"),
                    "seed": result.get("seed"),
                    "sparsity": result.get("sparsity"),
                    "epoch": record.get("epoch"),
                    **{
                        key.removeprefix("diagnostic_"): value
                        for key, value in record.items()
                        if key.startswith("diagnostic_")
                    },
                }
            )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output / "diagnostics.csv", index=False)
    report = output / "cause_analysis.md"
    lines = [
        "# AdaptiveGCL early-decline evidence report",
        "",
        "This report is generated from validation-only runs. It does not claim a causal explanation by itself.",
        "",
        "## Evidence to inspect",
        "",
        "- Compare `best_val_ndcg20` and `best_epoch` in `comparison.csv` before comparing loss values.",
        "- Use `diagnostics.csv` to inspect gate contribution, effective rank, semantic profile coverage, and SSL eligibility.",
        "- Compare one variant at a time with the `full` row at the same seed and sparsity.",
        "- A lower loss with lower NDCG is evidence of objective/ranking misalignment, not proof of metadata causality.",
        "",
        "## Decision status",
        "",
        "The runner records evidence only. Keep the full configuration unless a paired validation comparison supports a change; reserve test metrics for the locked candidate.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--output_dir")
    parser.add_argument("--bundle", help="Bundle path or data/current.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--sparsities", nargs="+", type=float, choices=[1.0, 0.25], default=[1.0, 0.25]
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=["full", "no_ssl", "no_user_text", "no_item_text", "no_dislikes"],
    )
    args = parser.parse_args(argv)
    if args.epochs < 1 or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("epochs must be positive and seeds must be in [0, 2**32)")
    if "full" not in args.variants:
        parser.error("Include full as the paired reference")
    for key in ("seeds", "sparsities", "variants"):
        values = getattr(args, key)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate --{key} would overwrite a run")
    return args


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    run(parse_args())
