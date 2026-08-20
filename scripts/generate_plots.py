import os
import sys

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import setup_logger

logger = setup_logger("generate_plots")

# Set publication style
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


def plot_radar_chart(df100: pd.DataFrame, fig_dir: str):
    """Generate multi-dimensional Radar Chart comparing 6 scientific dimensions."""
    categories = [
        "Recall@10",
        "NDCG@10",
        "Diversity@10",
        "Novelty@10",
        "Coverage@10",
        "User Eff Rank",
    ]
    num_vars = len(categories)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # complete loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    colors = {"lightgcn": "#1f77b4", "sgl": "#ff7f0e", "simgcl": "#2ca02c"}
    labels = {"lightgcn": "LightGCN", "sgl": "SGL", "simgcl": "SimGCL"}

    # Min-max normalization for radar balance
    raw_vals = {}
    for model in ["lightgcn", "sgl", "simgcl"]:
        m_df = df100[df100["model"] == model]
        if not m_df.empty:
            raw_vals[model] = [
                m_df["Recall@10_mean"].values[0] if "Recall@10_mean" in m_df else 0.0,
                m_df["NDCG@10_mean"].values[0] if "NDCG@10_mean" in m_df else 0.0,
                m_df["Diversity@10_mean"].values[0] if "Diversity@10_mean" in m_df else 0.0,
                m_df["Novelty@10_mean"].values[0] if "Novelty@10_mean" in m_df else 0.0,
                m_df["Coverage@10_mean"].values[0] if "Coverage@10_mean" in m_df else 0.0,
                m_df["User_Effective_Rank_mean"].values[0] if "User_Effective_Rank_mean" in m_df else 0.0,
            ]

    # Normalize each column across models to [0.2, 1.0] for fair visual radar
    all_matrix = np.array(list(raw_vals.values()))
    if len(all_matrix) > 0:
        min_c = all_matrix.min(axis=0)
        max_c = all_matrix.max(axis=0)
        denom = max_c - min_c
        denom[denom == 0] = 1.0

        for model, values in raw_vals.items():
            norm_values = 0.3 + 0.7 * (np.array(values) - min_c) / denom
            plot_vals = norm_values.tolist()
            plot_vals += plot_vals[:1]

            ax.plot(
                angles,
                plot_vals,
                color=colors.get(model, "#333333"),
                linewidth=2.2,
                label=labels.get(model, model),
            )
            ax.fill(angles, plot_vals, color=colors.get(model, "#333333"), alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Multi-Dimensional Beyond-Accuracy & Representation Profile", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()

    filename = "beyond_accuracy_radar.png"
    fig.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {filename}")


def main():
    fig_dir = os.path.join("results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    colors = {"lightgcn": "#1f77b4", "sgl": "#ff7f0e", "simgcl": "#2ca02c"}
    labels = {"lightgcn": "LightGCN", "sgl": "SGL", "simgcl": "SimGCL"}

    # 1. Always generate Training Learning Curves if history records exist
    history_dir = os.path.join("results", "history")
    if os.path.exists(history_dir):
        history_files = [f for f in os.listdir(history_dir) if f.endswith("_history.csv")]
        if history_files:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            for h_file in history_files:
                h_path = os.path.join(history_dir, h_file)
                h_df = pd.read_csv(h_path)
                if h_df.empty:
                    continue

                m_name = h_file.split("_")[0]
                lbl = labels.get(m_name, m_name.upper()) + f" ({h_file.replace('_history.csv', '')})"
                c = colors.get(m_name, "#333333")

                ax1.plot(h_df["epoch"], h_df["train_loss"], label=lbl, color=c, linewidth=2.0)
                ax2.plot(h_df["epoch"], h_df["val_ndcg_10"], label=lbl, color=c, linewidth=2.0)

            ax1.set_title("Training Loss Convergence Curve", fontsize=12, fontweight="bold", pad=10)
            ax1.set_xlabel("Epoch", fontsize=11)
            ax1.set_ylabel("Total Loss", fontsize=11)
            ax1.legend(frameon=True)
            ax1.grid(True, linestyle="--", alpha=0.6)

            ax2.set_title("Validation NDCG@10 Progression", fontsize=12, fontweight="bold", pad=10)
            ax2.set_xlabel("Epoch", fontsize=11)
            ax2.set_ylabel("Validation NDCG@10", fontsize=11)
            ax2.legend(frameon=True)
            ax2.grid(True, linestyle="--", alpha=0.6)

            plt.tight_layout()
            filename = "training_learning_curves.png"
            fig.savefig(os.path.join(fig_dir, filename), dpi=300)
            plt.close(fig)
            logger.info(f"Saved figure: {filename}")

    # 2. Benchmark Summary Figures (if benchmark summary exists)
    agg_csv = os.path.join("results", "aggregated", "benchmark_summary.csv")
    if not os.path.exists(agg_csv):
        logger.info(f"Full benchmark summary {agg_csv} not found yet (will be generated when running benchmark_all.py).")
        return

    df = pd.read_csv(agg_csv)

    # 3. Bar Plot: Accuracy & Beyond-Accuracy at 100% data
    df100 = df[df["sparsity"] == 1.0].copy()
    if not df100.empty:

        plot_metrics = ["Recall@10", "NDCG@10", "MRR@10", "Diversity@10", "Novelty@10", "Coverage@10"]
        for metric in plot_metrics:
            if f"{metric}_mean" not in df100.columns:
                continue

            fig, ax = plt.subplots(figsize=(7, 5))
            models = df100["model"].values
            means = df100[f"{metric}_mean"].values
            stds = df100[f"{metric}_std"].values if f"{metric}_std" in df100.columns else np.zeros_like(means)

            bar_colors = [colors.get(m, "#333333") for m in models]
            bar_labels = [labels.get(m, m) for m in models]

            bars = ax.bar(bar_labels, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85, width=0.5)

            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(
                    f"{mean:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_title(f"Comparison on Amazon Electronics: {metric} (100% Data)", fontsize=12, fontweight="bold", pad=12)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_ylim(0, max(means) * 1.25)
            plt.tight_layout()

            filename = metric.lower().replace("@", "_") + "_by_model.png"
            fig.savefig(os.path.join(fig_dir, filename), dpi=300)
            plt.close(fig)
            logger.info(f"Saved figure: {filename}")

        # Radar Chart
        plot_radar_chart(df100, fig_dir)

    # 2. Alignment vs Uniformity Scatter Plot (Pareto Representation Space)
    if "Alignment_mean" in df.columns and "Mean_Uniformity_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for model in ["lightgcn", "sgl", "simgcl"]:
            m_df = df[(df["model"] == model) & (df["sparsity"] == 1.0)]
            if m_df.empty:
                continue

            align = m_df["Alignment_mean"].values[0]
            unif = m_df["Mean_Uniformity_mean"].values[0]

            ax.scatter(
                unif,
                align,
                color=colors.get(model, "#333333"),
                s=250,
                label=labels.get(model, model),
                edgecolor="black",
                zorder=5,
            )
            ax.annotate(
                f"{labels.get(model, model)}\n(Align: {align:.3f}, Unif: {unif:.3f})",
                xy=(unif, align),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title("Representation Geometry: Alignment vs Uniformity (Wang & Isola, ICML 2020)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Uniformity Loss (Lower = More Uniform)", fontsize=11)
        ax.set_ylabel("Alignment Loss (Lower = Better Aligned)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        filename = "alignment_vs_uniformity.png"
        fig.savefig(os.path.join(fig_dir, filename), dpi=300)
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")

    # 3. Sparsity Curves (Recall@10, NDCG@10, Diversity@10)
    for metric in ["Recall@10", "NDCG@10", "Diversity@10"]:
        if f"{metric}_mean" not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for model in ["lightgcn", "sgl", "simgcl"]:
            m_df = df[df["model"] == model].sort_values(by="sparsity")
            if m_df.empty:
                continue

            x = (m_df["sparsity"] * 100).values
            y = m_df[f"{metric}_mean"].values
            y_err = m_df[f"{metric}_std"].values if f"{metric}_std" in m_df.columns else np.zeros_like(y)

            ax.errorbar(
                x,
                y,
                yerr=y_err,
                label=labels.get(model, model),
                color=colors.get(model, "#333333"),
                marker="o",
                linewidth=2.5,
                markersize=7,
                capsize=4,
            )

        ax.set_title(f"Data Sparsity Robustness: Training Data % vs {metric}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Training Interactions (%)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_xticks([25, 50, 75, 100])
        ax.legend(title="Model", frameon=True, facecolor="white", framealpha=0.9)
        plt.tight_layout()

        filename = f"sparsity_{metric.lower().replace('@', '_')}_curve.png"
        fig.savefig(os.path.join(fig_dir, filename), dpi=300)
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")

    # 4. Subgroup Analysis Plot (Cold-Start Tail vs Active Head Users)
    if "Tail_Recall@10_mean" in df.columns and "Head_Recall@10_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        df100 = df[df["sparsity"] == 1.0]

        models = ["lightgcn", "sgl", "simgcl"]
        tail_means = [df100[df100["model"] == m]["Tail_Recall@10_mean"].values[0] if not df100[df100["model"] == m].empty else 0.0 for m in models]
        head_means = [df100[df100["model"] == m]["Head_Recall@10_mean"].values[0] if not df100[df100["model"] == m].empty else 0.0 for m in models]

        x = np.arange(len(models))
        width = 0.35

        ax.bar(x - width/2, tail_means, width, label="Tail (Cold-Start Users)", color="#d62728", alpha=0.85)
        ax.bar(x + width/2, head_means, width, label="Head (Active Users)", color="#1f77b4", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in models], fontweight="bold")
        ax.set_ylabel("Recall@10", fontsize=11)
        ax.set_title("Performance Stratified by User Degree (Tail vs Head)", fontsize=12, fontweight="bold", pad=12)
        ax.legend()
        plt.tight_layout()

        filename = "subgroup_tail_vs_head.png"
        fig.savefig(os.path.join(fig_dir, filename), dpi=300)
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")

    logger.info("All enhanced research figures generated successfully!")


if __name__ == "__main__":
    main()

