"""Compare aggregation implementations on the actual graph (not a train benchmark)."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.data.bundle import resolve_bundle, sha256_file
from src.data.graph import get_norm_adj_tensor
from src.data.sparsity import create_sparse_train_set
from src.models.adaptive_gcl import AdaptiveGCL
from src.models.lightgcn import LightGCN


def stacked_reference(model, adj):
    if isinstance(model, AdaptiveGCL):
        adj = model._get_propagation_adj(adj)
        current = torch.cat([model.get_user_initial_embeddings(), model.get_gated_item_embeddings()[0]])
    else:
        current = torch.cat([model.user_embedding.weight, model.item_embedding.weight])
    layers = [current]
    for _ in range(model.num_layers):
        current = torch.sparse.mm(adj, current)
        layers.append(current)
    stacked = torch.stack(layers, dim=1)
    if isinstance(model, AdaptiveGCL):
        final = (stacked * model.get_layer_weights()[None, :, None]).sum(1)
    else:
        final = stacked.mean(1)
    return final.split([model.num_users, model.num_items])


def measure(model, adj, forward, repeats):
    def step():
        model.zero_grad(set_to_none=True)
        u, i = forward(model, adj)
        # Same differentiable scalar for both paths; includes backward through all layers.
        (u[:2048].square().mean() + i[:2048].square().mean()).backward()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(times)), "min_ms": min(times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 2**20,
            "samples_ms": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle")
    parser.add_argument("--density", type=float, default=.25)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This measurement requires CUDA")
    torch.set_num_threads(4)
    bundle = resolve_bundle(args.bundle)
    with bundle.artifact("mappings.pkl").open("rb") as stream:
        stats = pickle.load(stream)["stats"]
    train = create_sparse_train_set(pd.read_parquet(bundle.artifact("train.parquet")), args.density, 42)
    adj = get_norm_adj_tensor(train, stats["num_users"], stats["num_items"], torch.device("cuda"))
    report = {"scope": "FP32 graph propagation + scalar-loss backward; ID-only AdaptiveGCL isolates aggregation; excludes data loading, semantic MLPs, optimizer and ranking",
              "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "density": args.density, "edges": len(train), "nodes": adj.shape[0], "dim": 64, "layers": 3,
              "seed": 42, "warmup": 3, "repeats": args.repeats, "bundle_id": bundle.build_id,
              "manifest_sha256": sha256_file(bundle.manifest_path), "models": {}}
    for cls in (LightGCN, AdaptiveGCL):
        torch.manual_seed(42)
        kwargs = {"use_item_text": False, "user_semantic_weight": 0, "layer_aggregation": "anchored"} if cls is AdaptiveGCL else {}
        model = cls(stats["num_users"], stats["num_items"], **kwargs).cuda()
        measurements = {}
        # Alternate order to reduce systematic thermal/order effects.
        for name, fn in (("stacked", stacked_reference), ("current", lambda m, a: m(a)), ("current_repeat", lambda m, a: m(a)), ("stacked_repeat", stacked_reference)):
            measurements[name] = measure(model, adj, fn, args.repeats)
        report["models"][cls.__name__] = measurements
        del model
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
