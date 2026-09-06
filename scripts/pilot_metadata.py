"""Isolated, validation-only metadata pilot; never runs the 48-run benchmark."""

import argparse
import gc
import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.data.preprocessing import METADATA_FLAGS
from src.data.provenance import sha256_file
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import (
    PINNED_REVISION,
    build_user_history_features,
    load_training_text,
)
from src.evaluation.evaluator import Evaluator
from src.models.adaptive_gcl import AdaptiveGCL
from src.training.trainer import Trainer
from src.utils.checkpoints import get_experiment_fingerprint
from src.utils.config import load_config
from src.utils.seed import set_seed


def import_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load reference code: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_reference(snapshot, processed, cache):
    """Prove source/split/text identity before regenerating the legacy tensor."""
    original = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for entry in original["source_files"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Raw source differs from the reference snapshot")
    for entry in original["artifacts"]:
        name = Path(entry["path"]).name
        if (
            name.endswith(".parquet")
            and sha256_file(processed / name) != entry["sha256"]
        ):
            raise ValueError(f"Split/dislike artifact differs from reference: {name}")
    with (processed / "mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    metadata = {
        idx: {key: value for key, value in meta.items() if key not in METADATA_FLAGS}
        for idx, meta in mappings["item_metadata"].items()
    }
    reference = import_reference(
        snapshot / "src/data/text_encoder.py", "pilot_legacy_text"
    )
    cfg = original["text_features"]
    fingerprint = reference.get_text_input_fingerprint(
        metadata, len(metadata), cfg["model_name"], cfg["revision"]
    )
    if fingerprint != cfg["input_fingerprint"]:
        raise ValueError(
            "Cannot reconstruct original text/ID ordering; rebuild from the original source"
        )
    if cfg["resolved_revision"] != PINNED_REVISION:
        raise ValueError("Reference and masked encoders must use the same commit")
    cache.mkdir(parents=True, exist_ok=True)
    tensor_path = cache / "item_text_embeddings.pt"
    features = reference.encode_item_metadata(
        metadata,
        len(metadata),
        model_name=cfg["model_name"],
        revision=PINNED_REVISION,
        save_path=str(tensor_path),
    )
    identity = {
        "original_text_fingerprint": fingerprint,
        "encoder_revision": PINNED_REVISION,
        "legacy_encoder_code_sha256": sha256_file(
            snapshot / "src/data/text_encoder.py"
        ),
        "legacy_model_code_sha256": sha256_file(
            snapshot / "src/models/adaptive_gcl.py"
        ),
        "rebuilt_tensor_sha256": sha256_file(tensor_path),
        "original_tensor_sha256": cfg["sha256"],
        "note": "Original text fingerprint verified; tensor regenerated at the recorded encoder commit",
    }
    (cache / "reference.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    return mappings, features, identity


def make_model(variant, snapshot, config, mappings, sparse, features, item_mask):
    users, items = len(mappings["user2id"]), len(mappings["item2id"])
    ada = config["adaptive_gcl"]
    common = {
        "num_users": users,
        "num_items": items,
        "embedding_dim": config["model"]["embedding_dim"],
        "num_layers": config["model"]["num_layers"],
        "text_dim": features.shape[1],
        "text_features": features,
        "ssl_temp": ada["ssl_temp"],
        "ssl_reg": ada["ssl_reg"],
        "dirichlet_reg": ada["dirichlet_reg"],
        "node_dropout": ada["node_dropout"],
        "tau_plus": ada["tau_plus"],
    }
    if variant == "legacy_text":
        defaults = {"use_item_text": True, "user_semantic_weight": 0.5,
                    "layer_aggregation": "learnable", "mlp_weight_decay": 0.0}
        if any(ada.get(key, value) != value for key, value in defaults.items()):
            raise ValueError("Legacy metadata pilot requires default architecture; use ablate_adaptive.py for ablations")
        reference = import_reference(
            snapshot / "src/data/text_encoder.py", "pilot_legacy_pool"
        )
        profiles = reference.build_user_history_features(sparse, features, users)
        cls = import_reference(
            snapshot / "src/models/adaptive_gcl.py", "pilot_legacy_model"
        ).AdaptiveGCL
        model = cls(**common, user_history_features=profiles)
        # Instrumentation only: reference forward/SSL never reads these buffers.
        model.register_buffer(
            "item_text_mask", torch.ones(items, dtype=torch.bool), persistent=False
        )
        present = torch.zeros(users, dtype=torch.bool)
        present[torch.tensor(sparse["u_idx"].unique())] = True
        model.register_buffer("user_text_mask", present, persistent=False)
    else:
        profiles, user_mask = build_user_history_features(
            sparse, features, users, item_mask
        )
        model = AdaptiveGCL(
            **common,
            user_history_features=profiles,
            item_text_mask=item_mask,
            user_text_mask=user_mask,
            use_item_text=ada.get("use_item_text", True),
            user_semantic_weight=ada.get("user_semantic_weight", 0.5),
            layer_aggregation=ada.get("layer_aggregation", "learnable"),
        )
    return model


def run(args):
    snapshot = Path(args.legacy_snapshot).resolve()
    processed = ROOT / "data/processed"
    mappings, legacy, identity = prepare_reference(
        snapshot, processed, ROOT / "data/cache/metadata-legacy-v1"
    )
    masked, mask = load_training_text(processed, mappings)
    print(
        json.dumps(
            {
                "reference": identity,
                "usable_items": int(mask.sum()),
                "total_items": len(mask),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    output = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT
        / "results/pilots"
        / datetime.now(timezone.utc).strftime("metadata-%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "reference.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for ratio in args.sparsities:
        sparse = create_sparse_train_set(train, ratio, seed=args.seed)
        candidates = set(sparse["i_idx"].unique())
        warm_val = val[val["i_idx"].isin(candidates)]
        evaluator = Evaluator(
            train,
            warm_val,
            len(mappings["user2id"]),
            len(mappings["item2id"]),
            k_list=[10, 20],
            candidate_items=candidates,
            popularity_df=sparse,
        )
        for variant in args.variants:
            set_seed(args.seed)
            config = load_config("adaptive_gcl")
            config["training"].update(epochs=args.epochs, seed=args.seed)
            run_dir = output / variant / f"s{int(ratio * 100)}_seed{args.seed}"
            run_dir.mkdir(parents=True)
            config.update(
                validation_only=True,
                history_dir=str(run_dir / "history"),
                pilot_variant=variant,
            )
            code_hash = hashlib.sha256(
                (
                    get_experiment_fingerprint("adaptive_gcl")
                    + sha256_file(Path(__file__))
                    + json.dumps(identity, sort_keys=True)
                ).encode()
            ).hexdigest()
            config["experiment_fingerprint"] = hashlib.sha256(
                (
                    code_hash + variant + str(ratio) + str(args.seed) + str(args.epochs)
                ).encode()
            ).hexdigest()
            features = legacy if variant == "legacy_text" else masked
            model = make_model(
                variant, snapshot, config, mappings, sparse, features, mask
            )
            config["semantic_profile_users"] = int(model.user_text_mask.sum())
            config["usable_items"] = int(model.item_text_mask.sum())
            (run_dir / "config.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
            trainer = Trainer(
                model,
                sparse,
                evaluator,
                None,
                config,
                device,
                user_disliked_items=mappings.get("user_disliked_items", {}),
            )
            print(
                f"PILOT {variant} sparsity={ratio} seed={args.seed} epochs={args.epochs} -> {run_dir}",
                flush=True,
            )
            result = trainer.train(str(run_dir / "best.pt"))
            result.update(
                variant=variant,
                sparsity=ratio,
                seed=args.seed,
                fingerprint=config["experiment_fingerprint"],
                usable_items=config["usable_items"],
                semantic_profile_users=config["semantic_profile_users"],
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
            del trainer, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"Pilot completed: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy_snapshot", required=True, help="Trusted local code/manifest snapshot"
    )
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--output_dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sparsities", nargs="+", type=float, choices=[1.0, 0.25], default=[1.0, 0.25]
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["legacy_text", "masked_text"],
        default=["legacy_text", "masked_text"],
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    run(args)
