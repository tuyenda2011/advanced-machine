"""Run each recommender once at one training density.

The default is a five-epoch validation-only smoke check. Supplying a larger
budget or ``--with_test`` makes this a real evaluation run and names the
output ``evaluation_<timestamp>``.
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lightgcn", "xsimgcl", "directau", "adaptive_gcl")

# Make project imports work when this file is called from any working directory.
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_all import validate_run_result
from src.data.bundle import BundleError, resolve_bundle
from src.utils.checkpoints import (
    get_model_output_dir,
    get_run_fingerprint,
    write_run_status,
)
from src.utils.paths import resolve_output_root, write_run_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run each model once at one density; use 5 validation-only epochs for a smoke check or a larger budget for evaluation."
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
        help="Output directory. Defaults to results/runs/smoke_<timestamp> for the 5-epoch check, otherwise evaluation_<timestamp>.",
    )
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional bundle/archive path; default uses active data/processed view",
    )
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
    if args.resume and not args.output_root:
        parser.error("--resume requires an explicit --output_root")
    return args


def run_kind_for(args):
    """Name the artifact by what was actually executed."""

    return "smoke" if args.epochs <= 5 and not args.with_test else "evaluation"


def output_root_for(args):
    return resolve_output_root(args.output_root, kind=run_kind_for(args))


def resolve_project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
        str(resolve_project_path(args.config_dir)),
        "--output_root",
        str(output_root),
    ]
    if not args.with_test:
        command.append("--validation_only")
    if args.resume:
        command.append("--resume")
    if args.bundle:
        command.extend(["--bundle", str(resolve_project_path(args.bundle))])
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
    config_dir = resolve_project_path(args.config_dir)
    bundle = None
    if args.bundle:
        try:
            bundle = resolve_bundle(resolve_project_path(args.bundle))
        except BundleError as exc:
            raise ValueError(f"Dataset bundle is invalid: {exc}") from exc
    expected_fingerprint = get_run_fingerprint(
        model,
        args.density,
        args.seed,
        config_dir=str(config_dir),
        manifest_path=bundle.manifest_path if bundle is not None else None,
    )
    errors = validate_run_result(
        result,
        model,
        args.density,
        args.seed,
        args.epochs,
        expected_fingerprint,
        validation_only=not args.with_test,
    )
    if errors:
        raise ValueError(f"Invalid result {path}: " + "; ".join(errors))
    return path, result


def summary_row(model, result, status, returncode=None):
    monitor = result.get("monitor", "NDCG@20")
    val_metrics = result.get("val_metrics", {})
    test_metrics = result.get("test_metrics", {})
    return {
        "model": model,
        "status": status,
        "returncode": returncode,
        "error_type": result.get("_error_type"),
        "error_message": result.get("_error_message"),
        "best_epoch": result.get("best_epoch"),
        "total_epochs": result.get("total_epochs"),
        "monitor": monitor,
        "val_monitor": val_metrics.get(monitor),
        "test_ndcg20": test_metrics.get("NDCG@20"),
        "scoring_metric": result.get("scoring_metric"),
        "profile": result.get("profile"),
        "train_time_sec": result.get("total_train_time"),
        "result_file": result.get("_result_file"),
        "validation_only": result.get("validation_only"),
    }


def save_summary(output_root, rows, args, run_kind):
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "density": args.density,
        "seed": args.seed,
        "epochs": args.epochs,
        "validation_only": not args.with_test,
        "run_kind": run_kind,
        "models": list(args.models),
        "bundle": args.bundle,
        "runs": rows,
    }
    json_path = output_root / f"{run_kind}_summary.json"
    temp_path = json_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp_path, json_path)

    csv_path = output_root / f"{run_kind}_summary.csv"
    fields = [
        "model", "status", "returncode", "error_type", "error_message",
        "best_epoch", "total_epochs", "monitor", "val_monitor", "test_ndcg20",
        "scoring_metric", "profile", "train_time_sec", "result_file", "validation_only",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run_kind = run_kind_for(args)
    output_root = output_root_for(args)
    commands = [build_command(args, model, output_root) for model in args.models]
    if args.dry_run:
        print(json.dumps({
            "output_root": str(output_root),
            "density": args.density,
            "seed": args.seed,
            "epochs": args.epochs,
            "validation_only": not args.with_test,
            "bundle": args.bundle,
            "runs": [" ".join(command) for command in commands],
        }, indent=2))
        return 0

    if (output_root / "runner_status.json").exists() and not args.resume:
        raise FileExistsError(
            f"Output already contains runner_status.json: {output_root}. "
            "Choose a new --output_root or pass --resume."
        )

    write_run_manifest(
        output_root,
        kind=run_kind,
        metadata={
            "models": list(args.models),
            "density": args.density,
            "seed": args.seed,
            "epochs": args.epochs,
            "validation_only": not args.with_test,
            "bundle": args.bundle,
        },
    )

    total = len(args.models)
    attempted = 0
    succeeded = 0
    rows = []
    write_run_status(str(output_root), total, succeeded, attempted, status="running")
    interrupted = False
    for model, command in zip(args.models, commands):
        attempted += 1
        print(f"\n[{attempted}/{total}] Running {model.upper()} at density {args.density:.2f}", flush=True)
        result = None
        returncode = None
        process = None
        log_path = output_root / "logs" / f"{model}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_run_status(str(output_root), total, succeeded, attempted,
                             status="running", current_model=model)
            # A valid artifact with the requested identity/budget is reusable on
            # resume.  Invalid or stale artifacts fall through to train.py.
            if args.resume:
                try:
                    result_file, cached = read_result(model, args, output_root)
                    cached["_result_file"] = str(result_file)
                    succeeded += 1
                    rows.append(summary_row(model, cached, "cached", 0))
                    print(f"[CACHED] {model.upper()}: using validated existing result", flush=True)
                    continue
                except (FileNotFoundError, ValueError, json.JSONDecodeError):
                    pass
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(PROJECT_ROOT)
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_handle.write(line)
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"train.py exited with code {returncode}")
            result_file, result = read_result(model, args, output_root)
            result["_result_file"] = str(result_file)
            succeeded += 1
            rows.append(summary_row(model, result, "succeeded", returncode))
            print(
                f"[OK] {model.upper()}: {result.get('monitor', 'NDCG@20')}="
                f"{result.get('val_metrics', {}).get(result.get('monitor', 'NDCG@20'), float('nan')):.4f}",
                flush=True,
            )
        except KeyboardInterrupt as error:
            interrupted = True
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait()
            payload = {"_error_type": type(error).__name__, "_error_message": "interrupted"}
            rows.append(summary_row(model, payload, "interrupted", returncode))
            print(f"[INTERRUPTED] {model.upper()}: interrupted", flush=True)
            break
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
            payload = result or {}
            payload["_error_type"] = type(error).__name__
            payload["_error_message"] = str(error)
            rows.append(summary_row(model, payload, "failed", returncode))
            print(f"[ERROR] {model.upper()}: {error}", flush=True)
        finally:
            write_run_status(
                str(output_root), total, succeeded, attempted,
                status="interrupted" if interrupted else "running",
            )
            save_summary(output_root, rows, args, run_kind)

    if interrupted:
        raise KeyboardInterrupt("Smoke test interrupted; runner status and partial summary were saved")
    if succeeded != total:
        write_run_status(str(output_root), total, succeeded, attempted, status="failed")
        raise RuntimeError(f"Smoke test incomplete: {succeeded}/{total} models succeeded")
    write_run_status(str(output_root), total, succeeded, attempted, status="complete")
    print(f"\nSaved {run_kind} summary to {output_root / f'{run_kind}_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
