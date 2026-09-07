"""Run each recommender once at one training density.

The default is a five-epoch validation-only smoke test on all four models at
100% of the training graph.  Use ``--with_test`` only after you want the test
split and beyond-accuracy evaluation to run as well.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lightgcn", "xsimgcl", "directau", "adaptive_gcl")

# Make project imports work when this file is called from any working directory.
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.checkpoints import get_model_output_dir, write_run_status


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run each model once at one density for a quick pipeline check."
    )
    parser.add_argument(
        "--density",
        "--sparsity",
        dest="density",
        type=float,
        default=1.0,
        help="Fraction of training interactions to keep (default: 1.0 = 100%%).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epoch budget for each model (default: 5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Models to run once each (default: all four models).",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output directory. Defaults to a new results/smoke_YYYYMMDD_HHMMSS directory.",
    )
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument(
        "--with_test",
        action="store_true",
        help="Also read test labels and run final test/beyond-accuracy evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each model from its matching latest checkpoint.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned commands without creating output or training.",
    )
    args = parser.parse_args()
    if not 0 < args.density <= 1:
        parser.error("--density must be greater than 0 and at most 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if len(set(args.models)) != len(args.models):
        parser.error("Each model may be listed only once")
    return args


def output_root_for(args):
    if args.output_root:
        return Path(args.output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"smoke_{stamp}"


def build_command(args, model, output_root):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--model",
        model,
        "--sparsity",
        str(args.density),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--config_dir",
        str(args.config_dir),
        "--output_root",
        str(output_root),
    ]
    if not args.with_test:
        command.append("--validation_only")
    if args.resume:
        command.append("--resume")
    return command


def find_result(model, density, seed, output_root):
    tag = f"s{int(density * 100)}"
    path = Path(get_model_output_dir("raw", model, str(output_root))) / f"{model}_{tag}_seed{seed}.json"
    return path


def read_result(model, args, output_root):
    path = find_result(model, args.density, args.seed, output_root)
    if not path.exists():
        raise FileNotFoundError(f"Training finished without result JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("model_name") != model or result.get("seed") != args.seed:
        raise ValueError(f"Result identity mismatch in {path}")
    return path, result


def summary_row(model, result, status, returncode=None):
    monitor = result.get("monitor", "NDCG@20")
    val_metrics = result.get("val_metrics", {})
    test_metrics = result.get("test_metrics", {})
    return {
        "model": model,
        "status": status,
        "returncode": returncode,
        "best_epoch": result.get("best_epoch"),
        "total_epochs": result.get("total_epochs"),
        "monitor": monitor,
        "val_monitor": val_metrics.get(monitor),
        "test_ndcg20": test_metrics.get("NDCG@20"),
        "scoring_metric": result.get("scoring_metric"),
        "profile": result.get("profile"),
        "train_time_sec": result.get("total_train_time"),
        "result_file": result.get("_result_file"),
    }


def save_summary(output_root, rows, args):
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "density": args.density,
        "seed": args.seed,
        "epochs": args.epochs,
        "validation_only": not args.with_test,
        "models": list(args.models),
        "runs": rows,
    }
    json_path = output_root / "smoke_summary.json"
    temp_path = json_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, json_path)

    csv_path = output_root / "smoke_summary.csv"
    fields = list(rows[0]) if rows else ["model", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_root = output_root_for(args)
    commands = [build_command(args, model, output_root) for model in args.models]
    if args.dry_run:
        print(json.dumps({
            "output_root": str(output_root),
            "density": args.density,
            "seed": args.seed,
            "epochs": args.epochs,
            "validation_only": not args.with_test,
            "runs": [" ".join(command) for command in commands],
        }, indent=2))
        return 0

    if (output_root / "runner_status.json").exists() and not args.resume:
        raise FileExistsError(
            f"Output already contains runner_status.json: {output_root}. "
            "Choose a new --output_root or pass --resume."
        )

    total = len(args.models)
    attempted = 0
    succeeded = 0
    rows = []
    write_run_status(str(output_root), total, succeeded, attempted)
    for model, command in zip(args.models, commands):
        attempted += 1
        print(f"\n[{attempted}/{total}] Running {model.upper()} at density {args.density:.2f}", flush=True)
        result = None
        returncode = None
        try:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
            returncode = completed.returncode
            if returncode != 0:
                raise RuntimeError(f"train.py exited with code {returncode}")
            result_file, result = read_result(model, args, output_root)
            result["_result_file"] = str(result_file)
            succeeded += 1
            rows.append(summary_row(model, result, "succeeded", returncode))
            print(
                f"✅ {model.upper()}: {result.get('monitor', 'NDCG@20')}="
                f"{result.get('val_metrics', {}).get(result.get('monitor', 'NDCG@20'), float('nan')):.4f}",
                flush=True,
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            rows.append(summary_row(model, result or {}, "failed", returncode))
            print(f"❌ {model.upper()}: {error}", flush=True)
        finally:
            write_run_status(str(output_root), total, succeeded, attempted)
            save_summary(output_root, rows, args)

    if succeeded != total:
        raise RuntimeError(f"Smoke test incomplete: {succeeded}/{total} models succeeded")
    print(f"\nSaved smoke summary to {output_root / 'smoke_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
