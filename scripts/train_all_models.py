import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 encoding for Windows Command Prompt/PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from tabulate import tabulate

from src.utils.logging import setup_logger

logger = setup_logger("train_all_models")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate LightGCN, XSimGCL, DirectAU, and AdaptiveGCL under one or all sparsity levels."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgcn", "xsimgcl", "directau", "adaptive_gcl"],
        choices=["lightgcn", "xsimgcl", "directau", "adaptive_gcl"],
        help="Models to train sequentially (default: LightGCN, XSimGCL, DirectAU, AdaptiveGCL)",
    )
    parser.add_argument(
        "--sparsity",
        nargs="+",
        type=float,
        default=[1.0],
        help="One or more training edge sparsity ratios, e.g. --sparsity 1.0 0.75 0.5 0.25 (default: [1.0])",
    )
    parser.add_argument(
        "--all_sparsity",
        action="store_true",
        help="Shortcut to run across all 4 benchmark sparsity levels: [1.0, 0.75, 0.50, 0.25]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset partitioning and model initialization",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per model (default: 100 epochs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint if interrupted",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip automatic research figure generation after completion",
    )
    args = parser.parse_args()

    sparsity_list = [1.0, 0.75, 0.50, 0.25] if args.all_sparsity else args.sparsity
    start_total_time = time.perf_counter()

    print("=" * 85)
    print("🚀 GRAPH RECSYS SUITE: SEQUENTIAL 4-MODEL MULTI-SPARSITY RUNNER")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Models: {', '.join([m.upper() for m in args.models])}")
    print(f"📊 Sparsity Levels: {', '.join([f'{int(s*100)}%' for s in sparsity_list])} | Seed: {args.seed}")
    print(f"⏱️ Epochs per Model: {args.epochs}")
    print("=" * 85, flush=True)

    results_dir = os.path.join("results", "raw")
    os.makedirs(results_dir, exist_ok=True)

    completed_runs = []
    experiments = [(sp, m) for sp in sparsity_list for m in args.models]
    total_runs = len(experiments)

    for idx, (sp, model_name) in enumerate(experiments, 1):
        sparsity_pct = int(sp * 100)
        sparsity_tag = f"s{sparsity_pct}"

        print(f"\n[{idx}/{total_runs}] " + "-" * 70, flush=True)
        print(f"▶️  STARTING: {model_name.upper()} | SPARSITY: {sparsity_pct}% DATA | SEED: {args.seed} | EPOCHS: {args.epochs}", flush=True)
        print("-" * 75, flush=True)

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--model",
            model_name,
            "--sparsity",
            str(sp),
            "--seed",
            str(args.seed),
            "--epochs",
            str(args.epochs),
        ]
        if args.resume:
            cmd.append("--resume")

        env = os.environ.copy()
        env["PYTHONPATH"] = "."

        run_start = time.perf_counter()
        res = subprocess.run(cmd, env=env)
        run_duration = time.perf_counter() - run_start

        if res.returncode != 0:
            logger.error(f"❌ Training failed for {model_name.upper()} ({sparsity_pct}%) with exit code {res.returncode}")
            continue

        # Load run result JSON
        run_file = os.path.join(results_dir, model_name, f"{model_name}_{sparsity_tag}_seed{args.seed}.json")
        if not os.path.exists(run_file):
            run_file = os.path.join(results_dir, f"{model_name}_{sparsity_tag}_seed{args.seed}.json")

        if os.path.exists(run_file):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    run_data = json.load(f)
                run_data["run_duration_sec"] = run_duration
                completed_runs.append(run_data)
                print(f"✨ Finished {model_name.upper()} ({sparsity_pct}%) in {run_duration:.1f}s", flush=True)
            except Exception as e:
                logger.error(f"Could not parse run JSON for {model_name}: {e}")

    total_elapsed = time.perf_counter() - start_total_time

    # Display comparison table
    if completed_runs:
        print("\n" + "=" * 105)
        print("🏆 4-MODEL BENCHMARK RESULTS COMPARISON")
        print("=" * 105)

        summary_rows = []
        for r in completed_runs:
            m_name = r.get("model_name", "").upper()
            sp_val = r.get("sparsity_level", 1.0)
            test_m = r.get("test_metrics", {})
            rep_m = r.get("representation_metrics", {})
            svd_m = r.get("svd_metrics", {})
            sub_m = r.get("subgroup_metrics", {})
            tail_m = sub_m.get("Tail (Cold-Start)", {})

            summary_rows.append({
                "Model": m_name,
                "Sparsity": f"{int(sp_val*100)}%",
                "Recall@10": f"{test_m.get('Recall@10', 0):.4f}",
                "NDCG@10": f"{test_m.get('NDCG@10', 0):.4f}",
                "MRR@10": f"{test_m.get('MRR@10', 0):.4f}",
                "Diversity@10": f"{test_m.get('Diversity@10', 0):.4f}",
                "Novelty@10": f"{test_m.get('Novelty@10', 0):.4f}",
                "Alignment": f"{rep_m.get('alignment', 0):.4f}",
                "Uniformity": f"{rep_m.get('mean_uniformity', 0):.4f}",
                "Tail Rec@10": f"{tail_m.get('Recall@10', 0):.4f}",
                "Eff. Rank": f"{svd_m.get('user_effective_rank', 0):.2f}",
                "Latency (ms)": f"{r.get('inference_latency_ms_per_user', 0):.2f}",
            })

        summary_df = pd.DataFrame(summary_rows)
        print(tabulate(summary_df, headers="keys", tablefmt="fancy_grid", showindex=False))

        # Save comparative CSV
        agg_dir = os.path.join("results", "aggregated")
        os.makedirs(agg_dir, exist_ok=True)
        comparison_filename = (
            f"models_all_sparsity_seed{args.seed}.csv"
            if len(sparsity_list) > 1
            else f"four_models_comparison_s{int(sparsity_list[0]*100)}_seed{args.seed}.csv"
        )
        all_models_csv = os.path.join(agg_dir, comparison_filename)
        summary_df.to_csv(all_models_csv, index=False)
        print(f"\n📁 Saved comparative table to: {all_models_csv}")

    # Generate research figures unless disabled
    if not args.no_plots:
        print("\n📊 Generating research comparison figures & learning curves...", flush=True)
        try:
            from scripts.generate_plots import main as generate_figures
            generate_figures()
            print("✅ Figures saved to results/figures/", flush=True)
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    print("\n" + "=" * 85)
    print(f"🎉 ALL {len(completed_runs)} RUNS FINISHED IN {total_elapsed/60:.2f} MINUTES!")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
