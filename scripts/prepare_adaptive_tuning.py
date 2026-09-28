"""Freeze tuning configs from a trusted local control checkpoint; never train."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablate_adaptive import variant_config
from src.data.provenance import sha256_file
from src.utils.config import load_config

CASES = {
    "C0": "full", "C1": "no_ssl", "C2": "ssl_0003",
    "C3": "mlp_decay_1e4", "C4": "lr_0003", "C5": "no_dislikes",
}


def prepare(checkpoint: Path, destination: Path):
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    # Keep only the input sections; discard run-specific fingerprint/paths.
    base = {key: deepcopy(config[key]) for key in
            ("dataset", "model", "training", "evaluation", "sparsity", "adaptive_gcl")}
    base["model_name"] = "adaptive_gcl"
    base["training"].update(epochs=30, seed=42)
    base["evaluation"]["model_diagnostics"] = True
    variants = {case: variant_config(base, name) for case, name in CASES.items()}
    destination.mkdir(parents=True)
    for case, effective in variants.items():
        folder = destination / case
        folder.mkdir()
        common = {k: v for k, v in effective.items() if k not in {"adaptive_gcl", "model_name"}}
        model = {"model_name": "adaptive_gcl", "adaptive_gcl": effective["adaptive_gcl"]}
        (folder / "common.yaml").write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
        (folder / "adaptive_gcl.yaml").write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
        assert load_config("adaptive_gcl", str(folder)) == effective
    source = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": saved["epoch"], "best_score": saved["best_score"],
        "cases": CASES, "validation_only": True,
    }
    (destination / "control.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.checkpoint, args.output)
