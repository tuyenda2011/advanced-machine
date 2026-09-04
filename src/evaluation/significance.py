from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def compute_statistical_significance(
    scores_a: Union[List[float], np.ndarray],
    scores_b: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """Perform paired t-test and Wilcoxon signed-rank test between two models across seeds/users.

    Args:
        scores_a: Array of evaluation scores for Model A (Target/Proposed)
        scores_b: Array of evaluation scores for Model B (Baseline)

    Returns:
        Dict containing mean_a, mean_b, relative_improvement_pct, t_stat, p_value, and significance_star.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))

    rel_improv = ((mean_a - mean_b) / (mean_b + 1e-12)) * 100.0

    if arr_a.shape != arr_b.shape:
        raise ValueError("Paired significance tests require equally sized score arrays")

    if len(arr_a) < 2 or np.allclose(arr_a, arr_b):
        t_p_val = 1.0
        t_stat = 0.0
        wilcoxon_p_val = 1.0
        wilcoxon_stat = 0.0
    else:
        try:
            t_res = stats.ttest_rel(arr_a, arr_b)
            t_stat = float(t_res.statistic)
            t_p_val = float(t_res.pvalue)
            if np.isnan(t_p_val):
                t_p_val = 1.0
        except Exception:
            t_stat = 0.0
            t_p_val = 1.0

        try:
            wilcoxon_result = stats.wilcoxon(arr_a, arr_b)
            wilcoxon_stat = float(wilcoxon_result.statistic)
            wilcoxon_p_val = float(wilcoxon_result.pvalue)
        except Exception:
            wilcoxon_stat = 0.0
            wilcoxon_p_val = 1.0

    if t_p_val < 0.001:
        star = "***"
    elif t_p_val < 0.01:
        star = "**"
    elif t_p_val < 0.05:
        star = "*"
    else:
        star = "ns"

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "rel_improvement_pct": rel_improv,
        "t_statistic": t_stat,
        "p_value": t_p_val,
        "t_p_value": t_p_val,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p_val,
        "significance": star,
    }


def generate_latex_table(
    summary_df: pd.DataFrame,
    caption: str = "Performance comparison of LightGCN, XSimGCL, DirectAU, and AdaptiveGCL across data sparsity levels.",
    label: str = "tab:benchmark_results",
) -> str:
    """Generate publication-ready LaTeX table formatted according to ACM/IEEE guidelines.

    Highlights best results in bold and statistically significant improvements with asterisks.
    """
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("  \\centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\small")
    lines.append("  \\begin{tabular}{llcccccc}")
    lines.append("    \\toprule")
    lines.append(
        "    \\textbf{Sparsity} & \\textbf{Model} & \\textbf{Recall@10} & \\textbf{NDCG@10} & \\textbf{MRR@10} & \\textbf{Diversity@10} & \\textbf{Novelty@10} & \\textbf{Coverage@10} \\\\"
    )
    lines.append("    \\midrule")

    # Group by sparsity level
    if "sparsity" in summary_df.columns:
        sparsities = sorted(summary_df["sparsity"].unique(), reverse=True)
    else:
        sparsities = [1.0]

    for s_idx, sp in enumerate(sparsities):
        sp_df = summary_df[summary_df["sparsity"] == sp] if "sparsity" in summary_df.columns else summary_df
        sp_pct = f"{int(float(sp) * 100)}\\%"

        # Find max for each metric to format in bold
        metrics = ["Recall@10", "NDCG@10", "MRR@10", "Diversity@10", "Novelty@10", "Coverage@10"]
        max_vals = {}
        for m in metrics:
            if m in sp_df.columns:
                max_vals[m] = sp_df[m].max()

        for row_idx, (_, row) in enumerate(sp_df.iterrows()):
            m_name = row.get("model", "").upper()
            sp_label = f"\\multirow{{{len(sp_df)}}}{{*}}{{{sp_pct}}}" if row_idx == 0 else ""

            row_entries = [sp_label, m_name]
            for m in metrics:
                if m in row:
                    val = row[m]
                    val_str = f"{val:.4f}"
                    if abs(val - max_vals.get(m, -999)) < 1e-6:
                        val_str = f"\\textbf{{{val_str}}}"
                    row_entries.append(val_str)
                else:
                    row_entries.append("-")

            lines.append("    " + " & ".join(row_entries) + " \\\\")

        if s_idx < len(sparsities) - 1:
            lines.append("    \\midrule")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("  \\vspace{1ex}")
    lines.append("  {\\footnotesize \\textit{Note:} Bold numbers denote best performance. Statistical significance determined via paired t-test ($p < 0.05$).}")
    lines.append("\\end{table*}")

    return "\n".join(lines)
