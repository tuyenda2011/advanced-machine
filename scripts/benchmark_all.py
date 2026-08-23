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
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.significance import (
    compute_statistical_significance,
    generate_latex_table,
)
from src.utils.logging import setup_logger

logger = setup_logger("benchmark_all")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Academic Benchmark Suite for LightGCN, SGL, SimGCL, XSimGCL, DirectAU under Sparsity"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgcn", "xsimgcl", "directau", "adaptive_gcl"],
        help="List of models to benchmark (default: 4 SOTA models: LightGCN, XSimGCL, DirectAU, AdaptiveGCL)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 1 seed, 5 epochs, 100% data only"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for all benchmark runs"
    )
    args = parser.parse_args()

    models = args.models
    if args.quick:
        sparsity_levels = [1.0]
        seeds = [42]
        epochs = args.epochs if args.epochs is not None else 5
        logger.info("Running BENCHMARK SUITE in QUICK MODE (1 seed, 5 epochs, 100% data)...")
    else:
        sparsity_levels = [1.0, 0.75, 0.50, 0.25]
        seeds = [42, 2025, 3407]
        epochs = args.epochs if args.epochs is not None else 50
        logger.info(
            f"Running FULL BENCHMARK SUITE ({len(models)} models x {len(sparsity_levels)} sparsity levels x {len(seeds)} seeds = {len(models)*len(sparsity_levels)*len(seeds)} runs)..."
        )

    results_dir = os.path.join("results", "raw")
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        (model, sparsity, seed)
        for model in models
        for sparsity in sparsity_levels
        for seed in seeds
    ]

    all_runs = []
    pbar = tqdm(experiments, desc="Benchmark Suite Progress", unit="exp", dynamic_ncols=True)

    for model, sparsity, seed in pbar:
        sparsity_pct = int(sparsity * 100)
        sparsity_tag = f"s{sparsity_pct}"
        pbar.set_postfix({"Model": model.upper(), "Sparsity": f"{sparsity_pct}%", "Seed": seed})

        run_file = os.path.join(
            results_dir, model, f"{model}_{sparsity_tag}_seed{seed}.json"
        )
        if not os.path.exists(run_file):
            run_file = os.path.join(
                results_dir, f"{model}_{sparsity_tag}_seed{seed}.json"
            )

        # Check if run file exists and contains new metrics
        if os.path.exists(run_file):
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "representation_metrics" in data and "subgroup_metrics" in data:
                logger.info(f"Skipping existing completed run: {run_file}")
                all_runs.append(data)
                continue


        logger.info(
            f"Executing run: model={model}, sparsity={sparsity}, seed={seed}, epochs={epochs}..."
        )
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

        # Run train script as subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if res.returncode != 0:
            logger.error(
                f"Error executing run {model} {sparsity_tag} seed {seed}:"
            )
            logger.error(res.stderr)
            continue

        if os.path.exists(run_file):
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_runs.append(data)

    pbar.close()

    if not all_runs:
        logger.warning("No benchmark run results found.")
        return

    # Process and aggregate results into DataFrame
    rows = []
    for r in all_runs:
        test_m = r.get("test_metrics", {})
        val_m = r.get("val_metrics", {})
        rep_m = r.get("representation_metrics", {})
        svd_m = r.get("svd_metrics", {})
        sub_m = r.get("subgroup_metrics", {})

        tail_res = sub_m.get("Tail (Cold-Start)", {})
        head_res = sub_m.get("Head (Active)", {})

        rows.append(
            {
                "model": r["model_name"],
                "sparsity": r["sparsity_level"],
                "seed": r["seed"],
                "best_epoch": r.get("best_epoch", 0),
                "total_epochs": r.get("total_epochs", 0),
                "total_train_time": r.get("total_train_time", 0.0),
                "avg_epoch_time": r.get("avg_epoch_time", 0.0),
                "inference_latency_ms": r.get("inference_latency_ms_per_user", 0.0),
                "throughput_users_per_sec": r.get("throughput_users_per_sec", 0.0),
                # Accuracy Metrics
                "Recall@10": test_m.get("Recall@10", 0.0),
                "NDCG@10": test_m.get("NDCG@10", 0.0),
                "MRR@10": test_m.get("MRR@10", 0.0),
                "Recall@20": test_m.get("Recall@20", 0.0),
                "NDCG@20": test_m.get("NDCG@20", 0.0),
                # Beyond-Accuracy Metrics
                "Diversity@10": test_m.get("Diversity@10", 0.0),
                "Novelty@10": test_m.get("Novelty@10", 0.0),
                "Coverage@10": test_m.get("Coverage@10", 0.0),
                "Gini@10": test_m.get("Gini@10", 0.0),
                # Representation Geometry Metrics
                "Alignment": rep_m.get("alignment", 0.0),
                "Mean_Uniformity": rep_m.get("mean_uniformity", 0.0),
                "User_Effective_Rank": svd_m.get("user_effective_rank", 0.0),
                "Item_Effective_Rank": svd_m.get("item_effective_rank", 0.0),
                # Subgroup Metrics
                "Tail_Recall@10": tail_res.get("Recall@10", 0.0),
                "Tail_NDCG@10": tail_res.get("NDCG@10", 0.0),
                "Head_Recall@10": head_res.get("Recall@10", 0.0),
                "Head_NDCG@10": head_res.get("NDCG@10", 0.0),
                "Val_NDCG@10": val_m.get("NDCG@10", 0.0),
            }
        )

    df = pd.DataFrame(rows)

    # Save raw benchmark DataFrame
    agg_dir = os.path.join("results", "aggregated")
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
        "Tail_Recall@10",
        "Head_Recall@10",
        "inference_latency_ms",
        "total_train_time",
    ]

    for (sparsity, model), group in grouped:
        row_dict = {
            "sparsity": sparsity,
            "model": model,
            "runs": len(group),
        }

        for m in metrics_list:
            if m in group:
                m_mean = float(group[m].mean())
                m_std = float(group[m].std()) if len(group) > 1 else 0.0
                row_dict[f"{m}_mean"] = m_mean
                row_dict[f"{m}_std"] = m_std if not np.isnan(m_std) else 0.0
                row_dict[f"{m}_str"] = f"{m_mean:.4f} ± {0.0 if np.isnan(m_std) else m_std:.4f}"

        agg_rows.append(row_dict)

    agg_df = pd.DataFrame(agg_rows)
    agg_csv = os.path.join(agg_dir, "benchmark_summary.csv")
    agg_df.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregated benchmark summary to {agg_csv}")

    # Save per-model summary files
    for m_name in models:
        m_df = agg_df[agg_df["model"] == m_name]
        if not m_df.empty:
            m_csv = os.path.join(agg_dir, f"{m_name}_summary.csv")
            m_df.to_csv(m_csv, index=False)
            logger.info(f"Saved dedicated summary for {m_name.upper()} to {m_csv}")

    # Statistical Significance Testing (Any model vs LightGCN)
    sig_results = []
    for sp in sorted(df["sparsity"].unique(), reverse=True):
        sp_df = df[df["sparsity"] == sp]
        for m_name in ["Recall@10", "NDCG@10", "Diversity@10", "Novelty@10"]:
            lgcn_scores = sp_df[sp_df["model"] == "lightgcn"][m_name].values
            if len(lgcn_scores) == 0:
                continue
            for other_model in [m for m in models if m != "lightgcn"]:
                other_scores = sp_df[sp_df["model"] == other_model][m_name].values
                if len(other_scores) > 0:
                    sig_res = compute_statistical_significance(other_scores, lgcn_scores)
                    sig_results.append({
                        "sparsity": sp,
                        "metric": m_name,
                        "comparison": f"{other_model.upper()} vs LightGCN",
                        **sig_res,
                    })

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        sig_csv = os.path.join(agg_dir, "statistical_significance.csv")
        sig_df.to_csv(sig_csv, index=False)
        logger.info(f"Saved statistical significance analysis to {sig_csv}")

    # Generate Publication-ready LaTeX Table
    display_df = agg_df.copy()
    for m in ["Recall@10", "NDCG@10", "MRR@10", "Diversity@10", "Novelty@10", "Coverage@10"]:
        if f"{m}_mean" in display_df.columns:
            display_df[m] = display_df[f"{m}_mean"]

    latex_code = generate_latex_table(
        display_df,
        caption="Empirical evaluation of LightGCN, SGL, and SimGCL on Amazon Electronics across data sparsity levels.",
        label="tab:main_benchmark",
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
        generate_all_figures()
    except Exception as e:
        logger.warning(f"Could not automatically generate figures: {e}")

    logger.info("Comprehensive benchmark suite completed successfully!")


if __name__ == "__main__":
    main()
