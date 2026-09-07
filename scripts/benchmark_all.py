import argparse
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

import subprocess
from numbers import Real

import numpy as np
import pandas as pd

from src.evaluation.evaluator import EVALUATION_PROTOCOL
from src.evaluation.significance import (
    compute_statistical_significance,
    generate_latex_table,
    summarize_metric,
)
from src.utils.checkpoints import (
    get_model_output_dir,
    get_run_fingerprint,
    write_run_status,
)
from src.utils.logging import setup_logger

logger = setup_logger("benchmark_all")


REQUIRED_METRIC_PATHS = [
    ("val_metrics", "NDCG@10"),
    ("test_metrics", "Recall@10"),
    ("test_metrics", "NDCG@10"),
    ("test_metrics", "MRR@10"),
    ("test_metrics", "Recall@20"),
    ("test_metrics", "NDCG@20"),
    ("test_metrics", "Diversity@10"),
    ("test_metrics", "Novelty@10"),
    ("test_metrics", "Coverage@10"),
    ("test_metrics", "Gini@10"),
    ("representation_metrics", "alignment"),
    ("representation_metrics", "mean_uniformity"),
    ("svd_metrics", "user_effective_rank"),
    ("svd_metrics", "item_effective_rank"),
    ("subgroup_metrics", "Tail (Low-Activity)", "Recall@10"),
    ("subgroup_metrics", "Tail (Low-Activity)", "NDCG@10"),
    ("subgroup_metrics", "Head (Active)", "Recall@10"),
    ("subgroup_metrics", "Head (Active)", "NDCG@10"),
]


def _nested_value(data, path):
    value = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def validate_run_result(data, model, sparsity, seed, epochs, fingerprint):
    """Return all reasons a cached or newly produced run is unusable."""
    errors = []
    if not isinstance(data, dict):
        return ["result must be a JSON object"]
    expected = {
        "model_name": model,
        "sparsity_level": sparsity,
        "seed": seed,
        "max_epochs": epochs,
        "experiment_fingerprint": fingerprint,
        "evaluation_protocol": EVALUATION_PROTOCOL,
    }
    for key, value in expected.items():
        if isinstance(data.get(key), bool) or data.get(key) != value:
            errors.append(f"{key}={data.get(key)!r}, expected {value!r}")
    if not isinstance(data.get("experiment_family"), str) or not data["experiment_family"]:
        errors.append("missing experiment_family")
    if data.get("scoring_metric") not in ("dot_product", "cosine"):
        errors.append("invalid scoring_metric")
    if "profile" not in data or "monitor" not in data:
        errors.append("missing profile or monitor")
    if not isinstance(data.get("evaluation_metadata"), dict):
        errors.append("missing evaluation_metadata")
    else:
        metadata = data["evaluation_metadata"]
        required_metadata = {
            "history_mask_policy": "full_train_for_val_full_train_plus_val_for_test",
            "sparsity_scope": "model_training_graph_only",
            "popularity_reference": "full_train_unique_users",
            "subgroup_degree_reference": "full_train_fixed_across_sparsity",
        }
        for key, value in required_metadata.items():
            if metadata.get(key) != value:
                errors.append(f"evaluation_metadata.{key} is invalid")
    for key in (
        "best_epoch",
        "total_epochs",
        "total_train_time",
        "avg_epoch_time",
        "inference_latency_ms_per_user",
        "throughput_users_per_sec",
    ):
        try:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                errors.append(f"{key} must be a finite nonnegative number")
        except (KeyError, TypeError, ValueError):
            errors.append(f"missing or invalid {key}")
    for path in REQUIRED_METRIC_PATHS:
        try:
            value = _nested_value(data, path)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                errors.append(f"{'.'.join(path)} must be a finite number")
                continue
            metric = path[-1]
            if metric.startswith(("Recall@", "NDCG@", "MRR@", "Coverage@", "Gini@")) and not 0 <= value <= 1:
                errors.append(f"{'.'.join(path)} must be in [0, 1]")
            if metric.startswith("Diversity@") and not 0 <= value <= 2:
                errors.append(f"{metric} must be in [0, 2]")
            if (metric.startswith("Novelty@") or metric in {"alignment", "user_effective_rank", "item_effective_rank"}) and value < 0:
                errors.append(f"{metric} must be nonnegative")
        except (KeyError, TypeError, ValueError):
            errors.append(f"missing or invalid {'.'.join(path)}")
    for key in ("best_epoch", "total_epochs"):
        if type(data.get(key)) is not int:
            errors.append(f"{key} must be an integer")
    if all(type(data.get(key)) is int for key in ("best_epoch", "total_epochs")):
        if not 1 <= data["best_epoch"] <= data["total_epochs"] <= epochs:
            errors.append("best_epoch must be within completed epochs and budget")
    monitor = data.get("monitor", "NDCG@10")
    if not isinstance(monitor, str) or monitor not in {"NDCG@10", "NDCG@20"}:
        errors.append("invalid monitor")
    elif not isinstance(data.get("val_metrics"), dict) or monitor not in data["val_metrics"]:
        errors.append("missing validation monitor metric")
    else:
        value = data["val_metrics"][monitor]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
            errors.append("invalid validation monitor value")
    effective = data.get("effective_config")
    if isinstance(effective, dict):
        evaluation = effective.get("evaluation", {})
        if not isinstance(evaluation, dict) or evaluation.get("monitor", "NDCG@10") != monitor:
            errors.append("monitor does not match effective config")
    return errors


def holm_adjust(p_values):
    """Holm family-wise error correction while preserving NaN entries."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if valid_idx.size == 0:
        return adjusted
    ordered = valid_idx[np.argsort(values[valid_idx])]
    running_max = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        running_max = max(running_max, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for LightGCN, XSimGCL, DirectAU, and the proposed AdaptiveGCL model under sparsity"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgcn", "xsimgcl", "directau", "adaptive_gcl"],
        help="Models to benchmark (default: LightGCN, XSimGCL, DirectAU, AdaptiveGCL)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 1 seed, 5 epochs, 100%% data only"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for all benchmark runs"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching checkpoints (disabled by default to avoid reusing stale data splits)",
    )
    parser.add_argument("--sparsities", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry_run", action="store_true", help="List planned runs without training")
    parser.add_argument("--output_root", default="results")
    parser.add_argument("--config_dir", default="configs")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate models are not allowed")

    models = args.models
    if args.quick:
        sparsity_levels = [1.0]
        seeds = [42]
        epochs = args.epochs if args.epochs is not None else 5
    else:
        sparsity_levels = [1.0, 0.75, 0.50, 0.25]
        seeds = [42, 2024, 2025]
        epochs = args.epochs if args.epochs is not None else 100

    if args.sparsities is not None:
        sparsity_levels = args.sparsities
    if args.seeds is not None:
        seeds = args.seeds
    if epochs < 1 or any(not 0 < ratio <= 1 for ratio in sparsity_levels) or any(seed < 0 for seed in seeds):
        parser.error("Invalid epoch budget, sparsity or seed")
    if len(set(seeds)) != len(seeds) or len(set(sparsity_levels)) != len(sparsity_levels):
        parser.error("Duplicate seeds or sparsities are not allowed")
    logger.info(f"Planned {len(models)*len(sparsity_levels)*len(seeds)} runs: "
                f"{len(models)} models x {len(sparsity_levels)} sparsities x {len(seeds)} seeds, {epochs} epochs")
    if args.dry_run:
        print(json.dumps({"models": models, "sparsities": sparsity_levels, "seeds": seeds, "epochs": epochs, "runs": len(models)*len(sparsity_levels)*len(seeds)}, indent=2))
        return

    results_dir = os.path.join(args.output_root, "raw")
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        (model, sparsity, seed)
        for model in models
        for sparsity in sparsity_levels
        for seed in seeds
    ]

    all_runs = []
    total_experiments = len(experiments)
    write_run_status(args.output_root, total_experiments, 0, 0)

    for idx, (model, sparsity, seed) in enumerate(experiments, start=1):
        sparsity_pct = int(sparsity * 100)
        sparsity_tag = f"s{sparsity_pct}"

        preferred_run_file = os.path.join(
            get_model_output_dir("raw", model, args.output_root), f"{model}_{sparsity_tag}_seed{seed}.json"
        )
        legacy_run_file = os.path.join(
            results_dir, f"{model}_{sparsity_tag}_seed{seed}.json"
        )
        run_file = (
            preferred_run_file
            if os.path.exists(preferred_run_file)
            else legacy_run_file
        )

        # Check if run file exists and contains new metrics
        if os.path.exists(run_file):
            current_fingerprint = get_run_fingerprint(model, sparsity, seed, config_dir=args.config_dir)
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                run_errors = validate_run_result(
                    data, model, sparsity, seed, epochs, current_fingerprint
                )
            except (OSError, json.JSONDecodeError) as ex:
                run_errors = [f"cannot read result: {ex}"]
            if not run_errors:
                test_ndcg = data["test_metrics"]["NDCG@10"]
                test_recall = data["test_metrics"]["Recall@10"]
                print(
                    f"[{idx:02d}/{total_experiments:02d}] ⏩ [ĐÃ CÓ KẾT QUẢ] Bỏ qua {model.upper():<12} | Sparsity: {sparsity_pct:>3}% | Seed: {seed:>4} | Test NDCG@10: {test_ndcg:.4f} | Recall@10: {test_recall:.4f}",
                    flush=True,
                )
                all_runs.append(data)
                write_run_status(args.output_root, total_experiments, len(all_runs), idx)
                continue
            logger.warning("Ignoring unusable cached run %s: %s", run_file, "; ".join(run_errors))

        print("\n" + "=" * 85, flush=True)
        print(
            f"[{idx:02d}/{total_experiments:02d}] 🚀 BẮT ĐẦU HUẤN LUYỆN: {model.upper()} | SPARSITY: {sparsity_pct}% | SEED: {seed} | MAX EPOCHS: {epochs}",
            flush=True,
        )
        print("=" * 85, flush=True)

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--model",
            model,
            "--sparsity",
            str(sparsity),
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
        ]
        cmd.extend(["--output_root", args.output_root, "--config_dir", args.config_dir])
        if args.resume:
            cmd.append("--resume")

        # Run train script as subprocess with live streaming to terminal
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        res = subprocess.run(cmd, env=env, check=False)

        write_run_status(args.output_root, total_experiments, len(all_runs), idx)
        if res.returncode != 0:
            print(
                f"[{idx:02d}/{total_experiments:02d}] ❌ [LỖI LƯỢT CHẠY] {model.upper():<12} | Sparsity: {sparsity_pct:>3}% | Seed: {seed:>4}",
                flush=True,
            )
            logger.error(
                f"Error executing run {model} {sparsity_tag} seed {seed} (exit code: {res.returncode})"
            )
            continue

        run_file = preferred_run_file
        if os.path.exists(run_file):
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            current_fingerprint = get_run_fingerprint(model, sparsity, seed, config_dir=args.config_dir)
            run_errors = validate_run_result(
                data, model, sparsity, seed, epochs, current_fingerprint
            )
            if run_errors:
                raise RuntimeError(
                    f"Training produced an invalid result at {run_file}: "
                    + "; ".join(run_errors)
                )
            all_runs.append(data)
            write_run_status(args.output_root, total_experiments, len(all_runs), idx)
            test_ndcg = data["test_metrics"]["NDCG@10"]
            test_recall = data["test_metrics"]["Recall@10"]
            best_epoch = data["best_epoch"]
            print(
                f"[{idx:02d}/{total_experiments:02d}] ✅ [HOÀN THÀNH TRAIN] {model.upper():<12} | Sparsity: {sparsity_pct:>3}% | Seed: {seed:>4} | Best Epoch: {best_epoch:>2} | NDCG@10: {test_ndcg:.4f} | Recall@10: {test_recall:.4f}\n",
                flush=True,
            )
        else:
            raise FileNotFoundError(f"Training completed without result file: {run_file}")

    if len(all_runs) != total_experiments:
        raise RuntimeError(f"Benchmark incomplete: {len(all_runs)}/{total_experiments} succeeded; {total_experiments-len(all_runs)} failed")

    # Process and aggregate results into DataFrame
    rows = []
    for r in all_runs:
        test_m = r["test_metrics"]
        val_m = r["val_metrics"]
        rep_m = r["representation_metrics"]
        svd_m = r["svd_metrics"]
        sub_m = r["subgroup_metrics"]

        tail_res = sub_m["Tail (Low-Activity)"]
        head_res = sub_m["Head (Active)"]

        rows.append(
            {
                "model": r["model_name"],
                "evaluation_protocol": r["evaluation_protocol"],
                "monitor": r["monitor"],
                "monitor_value": val_m[r["monitor"]],
                "profile": r.get("profile"),
                "scoring_metric": r["scoring_metric"],
                "experiment_fingerprint": r["experiment_fingerprint"],
                "experiment_family": r["experiment_family"],
                "sparsity": r["sparsity_level"],
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "total_epochs": r["total_epochs"],
                "total_train_time": r["total_train_time"],
                "avg_epoch_time": r["avg_epoch_time"],
                "inference_latency_ms": r["inference_latency_ms_per_user"],
                "throughput_users_per_sec": r["throughput_users_per_sec"],
                # Accuracy Metrics
                "Recall@10": test_m["Recall@10"],
                "NDCG@10": test_m["NDCG@10"],
                "MRR@10": test_m["MRR@10"],
                "Recall@20": test_m["Recall@20"],
                "NDCG@20": test_m["NDCG@20"],
                # Beyond-Accuracy Metrics
                "Diversity@10": test_m["Diversity@10"],
                "Novelty@10": test_m["Novelty@10"],
                "Coverage@10": test_m["Coverage@10"],
                "Gini@10": test_m["Gini@10"],
                # Representation Geometry Metrics
                "Alignment": rep_m["alignment"],
                "Mean_Uniformity": rep_m["mean_uniformity"],
                "User_Effective_Rank": svd_m["user_effective_rank"],
                "Item_Effective_Rank": svd_m["item_effective_rank"],
                # Subgroup Metrics
                "Tail_Recall@10": tail_res["Recall@10"],
                "Tail_NDCG@10": tail_res["NDCG@10"],
                "Head_Recall@10": head_res["Recall@10"],
                "Head_NDCG@10": head_res["NDCG@10"],
                "Val_NDCG@10": val_m["NDCG@10"],
            }
        )

    df = pd.DataFrame(rows)

    # Save raw benchmark DataFrame
    agg_dir = os.path.join(args.output_root, "aggregated")
    os.makedirs(agg_dir, exist_ok=True)
    raw_csv = os.path.join(agg_dir, "raw_benchmark_runs.csv")
    df.to_csv(raw_csv, index=False)
    logger.info(f"Saved raw benchmark runs table to {raw_csv}")

    # Compute mean +/- std aggregated table grouped by model and sparsity
    grouped = df.groupby(["sparsity", "model"])
    agg_rows = []

    metrics_list = [
        "Recall@10",
        "NDCG@10",
        "MRR@10",
        "Recall@20",
        "NDCG@20",
        "Diversity@10",
        "Novelty@10",
        "Coverage@10",
        "Gini@10",
        "Alignment",
        "Mean_Uniformity",
        "User_Effective_Rank",
        "Item_Effective_Rank",
        "Tail_Recall@10",
        "Head_Recall@10",
        "inference_latency_ms",
        "total_train_time",
    ]

    for (sparsity, model), group in grouped:
        if group["evaluation_protocol"].nunique() != 1:
            raise ValueError(f"Mixed evaluation protocols for {model} at sparsity {sparsity}")
        if group["experiment_family"].nunique() != 1:
            raise ValueError(f"Mixed experiment settings for {model} at sparsity {sparsity}")
        row_dict = {
            "sparsity": sparsity,
            "model": model,
            "runs": len(group),
            "evaluation_protocol": group["evaluation_protocol"].iloc[0],
            "experiment_family": group["experiment_family"].iloc[0],
            "source_run_fingerprints": json.dumps(sorted(group["experiment_fingerprint"].unique().tolist())),
            "monitor": group["monitor"].iloc[0],
        }

        for m in metrics_list:
            if m in group:
                for suffix, value in summarize_metric(group[m]).items():
                    row_dict[f"{m}_{suffix}"] = value

        agg_rows.append(row_dict)

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(agg_dir, "benchmark_summary.csv")
    agg_df.to_csv(agg_csv, index=False, na_rep="N/A")
    logger.info(f"Saved aggregated benchmark summary to {agg_csv}")

    # Save per-model summary files
    for m_name in models:
        m_df = agg_df[agg_df["model"] == m_name]
        if not m_df.empty:
            m_csv = os.path.join(agg_dir, f"{m_name}_summary.csv")
            m_df.to_csv(m_csv, index=False, na_rep="N/A")
            logger.info(f"Saved dedicated summary for {m_name.upper()} to {m_csv}")

    # Statistical Significance Testing (Any model vs LightGCN)
    sig_results = []
    for sp in sorted(df["sparsity"].unique(), reverse=True):
        sp_df = df[df["sparsity"] == sp]
        for m_name in ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "Diversity@10", "Novelty@10"]:
            lgcn_scores = sp_df[sp_df["model"] == "lightgcn"][["seed", m_name]]
            if lgcn_scores.empty:
                continue
            for other_model in [m for m in models if m != "lightgcn"]:
                other_scores = sp_df[sp_df["model"] == other_model][["seed", m_name]]
                paired = other_scores.merge(
                    lgcn_scores,
                    on="seed",
                    suffixes=("_model", "_lightgcn"),
                ).sort_values("seed").dropna()
                if not paired.empty:
                    sig_res = compute_statistical_significance(
                        paired[f"{m_name}_model"].values,
                        paired[f"{m_name}_lightgcn"].values,
                    )
                    sig_results.append({
                        "sparsity": sp,
                        "metric": m_name,
                        "comparison": f"{other_model.upper()} vs LightGCN",
                        **sig_res,
                    })

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        sig_df["holm_p_value"] = holm_adjust(sig_df["t_p_value"].to_numpy())
        sig_df["holm_significance"] = np.select(
            [sig_df["holm_p_value"].isna(), sig_df["holm_p_value"] < 0.05],
            ["N/A", "significant"],
            default="ns",
        )
        sig_df["inference_note"] = np.where(
            sig_df["paired_samples"] < 5,
            "exploratory_low_power_fewer_than_5_seeds",
            "confirmatory",
        )
        sig_csv = os.path.join(agg_dir, "statistical_significance.csv")
        sig_df.to_csv(sig_csv, index=False, na_rep="N/A")
        logger.info(f"Saved statistical significance analysis to {sig_csv}")

    # Generate Publication-ready LaTeX Table
    display_df = agg_df.copy()
    for m in ["Recall@10", "NDCG@10", "Recall@20", "NDCG@20", "MRR@10", "Diversity@10", "Novelty@10", "Coverage@10"]:
        if f"{m}_mean" in display_df.columns:
            display_df[m] = display_df[f"{m}_mean"]

    latex_code = generate_latex_table(
        display_df,
        caption="Empirical evaluation of LightGCN, XSimGCL, DirectAU, and AdaptiveGCL on Amazon Electronics across data sparsity levels.",
        label="tab:main_benchmark",
        ranking_k=20,
    )
    latex_path = os.path.join(agg_dir, "benchmark_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    logger.info(f"Generated academic LaTeX table at {latex_path}")

    # Performance Drop@25%
    drop_rows = []
    for model in models:
        m100 = agg_df[(agg_df["model"] == model) & (agg_df["sparsity"] == 1.0)]
        m25 = agg_df[(agg_df["model"] == model) & (agg_df["sparsity"] == 0.25)]

        if not m100.empty and not m25.empty:
            rec100 = m100["Recall@10_mean"].values[0]
            rec25 = m25["Recall@10_mean"].values[0]
            ndcg100 = m100["NDCG@10_mean"].values[0]
            ndcg25 = m25["NDCG@10_mean"].values[0]

            drop_rec = ((rec100 - rec25) / rec100) * 100.0 if rec100 > 0 else 0.0
            drop_ndcg = ((ndcg100 - ndcg25) / ndcg100) * 100.0 if ndcg100 > 0 else 0.0

            drop_rows.append({
                "model": model,
                "Recall@10_100": rec100,
                "Recall@10_25": rec25,
                "Drop_Recall@10_pct": drop_rec,
                "NDCG@10_100": ndcg100,
                "NDCG@10_25": ndcg25,
                "Drop_NDCG@10_pct": drop_ndcg,
            })

    if drop_rows:
        drop_df = pd.DataFrame(drop_rows)
        drop_csv = os.path.join(agg_dir, "sparsity_drop25_summary.csv")
        drop_df.to_csv(drop_csv, index=False)
        logger.info(f"Saved sparsity performance drop table to {drop_csv}")

    # Automatically generate all publication figures
    try:
        from scripts.generate_plots import main as generate_all_figures
        logger.info("Automatically generating research publication figures...")
        if args.output_root == "results":
            generate_all_figures()
    except Exception as e:
        logger.warning(f"Could not automatically generate figures: {e}")

    logger.info("Comprehensive benchmark suite completed successfully!")


if __name__ == "__main__":
    main()
