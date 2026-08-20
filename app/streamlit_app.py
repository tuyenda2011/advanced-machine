import json
import os
import pickle
import sys
import time

# Ensure project root is in sys.path when script is executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.utils.config import load_config

# Page Configuration
st.set_page_config(
    page_title="Graph Contrastive RecSys Research Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_processed_data():
    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, "train.parquet")
    val_path = os.path.join(processed_dir, "val.parquet")
    test_path = os.path.join(processed_dir, "test.parquet")
    mappings_path = os.path.join(processed_dir, "mappings.pkl")

    if not os.path.exists(train_path):
        return None, None, None, None

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    with open(mappings_path, "rb") as f:
        mappings = pickle.load(f)

    return train_df, val_df, test_df, mappings


@st.cache_resource
def load_trained_model(
    model_name: str,
    num_users: int,
    num_items: int,
    sparsity: float = 1.0,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(model_name, "configs")
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]

    if model_name == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif model_name == "sgl":
        sgl_cfg = config["sgl"]
        model = SGL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            ssl_weight=sgl_cfg["ssl_weight"],
        )
    elif model_name == "simgcl":
        sim_cfg = config["simgcl"]
        model = SimGCL(
            num_users,
            num_items,
            embedding_dim=emb_dim,
            num_layers=num_layers,
            contrastive_weight=sim_cfg["contrastive_weight"],
        )

    sparsity_tag = f"s{int(sparsity * 100)}"
    best_checkpoint_path = os.path.join("results", "checkpoints", f"{model_name}_best.pt")
    specific_checkpoint_path = os.path.join(
        "results", "checkpoints", f"{model_name}_{sparsity_tag}_seed{seed}.pt"
    )

    checkpoint_path = (
        best_checkpoint_path
        if os.path.exists(best_checkpoint_path)
        else specific_checkpoint_path
    )

    if not os.path.exists(checkpoint_path):
        # Fallback to artifacts/checkpoints if exists
        fallback_path = os.path.join("artifacts", "checkpoints", f"{model_name}_{sparsity_tag}_seed{seed}.pt")
        if os.path.exists(fallback_path):
            checkpoint_path = fallback_path

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.to(device)
    model.eval()

    from src.data.graph import get_norm_adj_tensor

    train_df, _, _, _ = load_processed_data()
    norm_adj = get_norm_adj_tensor(train_df, num_users, num_items, device)

    with torch.no_grad():
        u_embeds, i_embeds = model(norm_adj)

    return model, u_embeds, i_embeds, device


def main() -> None:
    st.title("⚡ Graph Contrastive Learning for Recommendation: Advanced Research Suite")
    st.caption(
        "A Rigorous Comparative Study & Representation Geometry Analysis of **LightGCN** vs. **SGL** vs. **SimGCL** under Data Sparsity"
    )

    train_df, val_df, test_df, mappings = load_processed_data()

    if train_df is None:
        st.error(
            "⚠️ Processed Amazon Electronics dataset not found. Please run `python scripts/prepare_data.py` first."
        )
        return

    num_users = mappings["stats"]["num_users"]
    num_items = mappings["stats"]["num_items"]
    item_metadata = mappings["item_metadata"]
    user2id = mappings["user2id"]

    # Precalculate popularity for novelty score
    item_pop = train_df["i_idx"].value_counts().to_dict()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎯 Interactive Recommendation",
            "📊 Benchmark & Significance",
            "🌐 Representation Geometry & SVD",
            "📉 Sparsity & Cold-Start Subgroups",
            "📘 Theoretical Foundations & Math",
        ]
    )

    # ==============================================================================
    # TAB 1: INTERACTIVE RECOMMENDATION & LIVE BEYOND-ACCURACY METRICS
    # ==============================================================================
    with tab1:
        st.subheader("Interactive Top-10 Recommendation & Live Beyond-Accuracy Metrics")
        col_select, col_info = st.columns([1, 2])

        with col_select:
            id2user = {v: k for k, v in user2id.items()}
            u_idx = st.number_input(
                "Select User Index (0 to 135,995):",
                min_value=0,
                max_value=max(0, num_users - 1),
                value=min(42, num_users - 1),
                step=1,
            )
            reviewer_id = id2user.get(u_idx, f"User_{u_idx}")
            st.info(f"👤 **Reviewer ID**: `{reviewer_id}` (Index: `{u_idx}`)")

        with col_info:
            user_history = train_df[train_df["u_idx"] == u_idx]
            st.markdown(
                f"**User History:** `{len(user_history)} rated items (≥ 4.0)` in Training Graph"
            )
            with st.expander("View User Recent History", expanded=False):
                history_items = user_history.sort_values(by="timestamp", ascending=False).head(10)
                h_data = []
                for _, row in history_items.iterrows():
                    info = item_metadata.get(row["i_idx"], {})
                    h_data.append({
                        "Product Name": info.get("title", "Unknown"),
                        "Brand": info.get("brand", "Unknown"),
                        "Category": info.get("categories", "Unknown"),
                    })
                st.dataframe(pd.DataFrame(h_data), use_container_width=True, hide_index=True)

        st.markdown("---")

        if st.button("🚀 Generate Top-10 Recommendations & Real-Time Metrics", type="primary"):
            seen_items = set(user_history["i_idx"])
            cols = st.columns(3)
            model_names = ["lightgcn", "sgl", "simgcl"]
            display_titles = [
                "LightGCN (Pure CF)",
                "SGL (Edge Dropout CL)",
                "SimGCL (Noise Perturbation CL)",
            ]

            for col, m_name, d_title in zip(cols, model_names, display_titles):
                with col:
                    st.markdown(f"### {d_title}")
                    try:
                        with st.spinner(f"Loading {d_title}..."):
                            model, u_embeds, i_embeds, device = load_trained_model(
                                m_name, num_users, num_items
                            )

                        start_t = time.perf_counter()
                        u_vec = u_embeds[u_idx : u_idx + 1]
                        scores = torch.matmul(u_vec, i_embeds.T).squeeze(0)

                        if seen_items:
                            seen_tensor = torch.tensor(list(seen_items), dtype=torch.long, device=device)
                            scores[seen_tensor] = -1e9

                        topk_scores, topk_indices = torch.topk(scores, k=10)
                        latency_ms = (time.perf_counter() - start_t) * 1000.0

                        topk_ids = topk_indices.cpu().numpy()
                        rec_list = []
                        for rank, (idx, score) in enumerate(zip(topk_ids, topk_scores.cpu().numpy()), start=1):
                            info = item_metadata.get(idx, {})
                            rec_list.append({
                                "Rank": rank,
                                "Product Name": info.get("title", "Unknown")[:45] + "...",
                                "Brand": info.get("brand", "Unknown"),
                                "Score": f"{score:.3f}",
                            })

                        # Compute user-level Diversity (ILD) and Novelty
                        rec_item_embeds = F.normalize(i_embeds[topk_indices], dim=-1)
                        sim_mat = torch.matmul(rec_item_embeds, rec_item_embeds.T)
                        triu_idx = torch.triu_indices(10, 10, offset=1)
                        ild_score = (1.0 - sim_mat[triu_idx[0], triu_idx[1]]).mean().item()

                        novelty_bits = np.mean([
                            -np.log2((item_pop.get(int(item_id), 0) + 1) / float(num_users))
                            for item_id in topk_ids
                        ])

                        st.dataframe(pd.DataFrame(rec_list), use_container_width=True, hide_index=True)

                        # Live metric badges
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Latency", f"{latency_ms:.2f} ms")
                        m2.metric("Diversity (ILD)", f"{ild_score:.3f}")
                        m3.metric("Novelty", f"{novelty_bits:.2f} bits")

                    except Exception as e:
                        st.error(f"Error loading {m_name}: Checkpoint missing.")
                        st.caption(str(e))

    # ==============================================================================
    # TAB 2: COMPREHENSIVE BENCHMARK RESULTS & STATISTICAL SIGNIFICANCE
    # ==============================================================================
    with tab2:
        st.subheader("Comprehensive Benchmark Suite (Accuracy + Beyond-Accuracy)")
        agg_csv = os.path.join("results", "aggregated", "benchmark_summary.csv")

        if os.path.exists(agg_csv):
            df_res = pd.read_csv(agg_csv)
            st.dataframe(df_res, use_container_width=True)

            st.markdown("### 🔬 Statistical Significance Analysis (Paired t-test & Wilcoxon)")
            sig_csv = os.path.join("results", "aggregated", "statistical_significance.csv")
            if os.path.exists(sig_csv):
                df_sig = pd.read_csv(sig_csv)
                st.dataframe(df_sig, use_container_width=True)
                st.caption(
                    "Significance levels: *** (p < 0.001), ** (p < 0.01), * (p < 0.05), ns (not significant)."
                )

            st.markdown("### 📄 LaTeX Academic Table Generator")
            tex_file = os.path.join("results", "aggregated", "benchmark_table.tex")
            if os.path.exists(tex_file):
                with open(tex_file, "r", encoding="utf-8") as f:
                    tex_code = f.read()
                with st.expander("Show LaTeX Code for Academic Paper / Thesis", expanded=False):
                    st.code(tex_code, language="latex")

            st.markdown("### 📊 Multi-Metric Comparison Charts")
            c1, c2, c3 = st.columns(3)
            with c1:
                if os.path.exists("results/figures/recall_10_by_model.png"):
                    st.image("results/figures/recall_10_by_model.png", caption="Recall@10 Comparison")
            with c2:
                if os.path.exists("results/figures/diversity_10_by_model.png"):
                    st.image("results/figures/diversity_10_by_model.png", caption="Diversity@10 (ILD)")
            with c3:
                if os.path.exists("results/figures/novelty_10_by_model.png"):
                    st.image("results/figures/novelty_10_by_model.png", caption="Novelty@10 (Self-Info)")
        else:
            st.info(
                "No aggregated benchmark results found yet. Run `python scripts/benchmark_all.py` to populate."
            )

    # ==============================================================================
    # TAB 3: REPRESENTATION GEOMETRY & SVD PHỔ
    # ==============================================================================
    with tab3:
        st.subheader("Representation Geometry: Alignment vs Uniformity & SVD Spectrum")
        st.markdown(
            """
        According to **Wang & Isola (ICML 2020)** and **SimGCL (SIGIR 2022)**, Contrastive Learning operates directly on the representation hypersphere:
        - **Alignment Loss $\\mathcal{L}_{align}$**: Encourages connected user-item pairs to map to nearby points on the unit sphere.
        - **Uniformity Loss $\\mathcal{L}_{uniform}$**: Enforces representations to distribute uniformly over the sphere, maximizing information entropy and preventing dimensional collapse.
        """
        )

        col_geom1, col_geom2 = st.columns(2)
        with col_geom1:
            if os.path.exists("results/figures/alignment_vs_uniformity.png"):
                st.image(
                    "results/figures/alignment_vs_uniformity.png",
                    caption="Alignment vs Uniformity Pareto Frontier",
                )
            else:
                st.info("Run `scripts/generate_plots.py` to generate the Alignment-Uniformity figure.")

        with col_geom2:
            if os.path.exists("results/figures/beyond_accuracy_radar.png"):
                st.image(
                    "results/figures/beyond_accuracy_radar.png",
                    caption="6-Dimensional Radar Profile (Accuracy, Diversity, Novelty, SVD Rank)",
                )
            else:
                st.info("Run `scripts/generate_plots.py` to generate the Radar Profile figure.")

    # ==============================================================================
    # TAB 4: SPARSITY & LONG-TAIL COLD-START SUBGROUP ANALYSIS
    # ==============================================================================
    with tab4:
        st.subheader("Sparsity Robustness & Long-Tail Degree Subgroup Breakdown")
        st.markdown(
            "Graph Contrastive Learning is designed to specifically rescue **Tail (Cold-Start)** users who have minimal interaction history."
        )

        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            if os.path.exists("results/figures/sparsity_recall_10_curve.png"):
                st.image(
                    "results/figures/sparsity_recall_10_curve.png",
                    caption="Sparsity Curve: Training Edge % vs Recall@10",
                )
        with col_sub2:
            if os.path.exists("results/figures/subgroup_tail_vs_head.png"):
                st.image(
                    "results/figures/subgroup_tail_vs_head.png",
                    caption="Performance Stratified by Interaction Degree (Tail vs Head)",
                )

        drop_csv = os.path.join("results", "aggregated", "sparsity_drop25_summary.csv")
        if os.path.exists(drop_csv):
            st.markdown("### Relative Performance Degradation (Drop@25%)")
            st.dataframe(pd.read_csv(drop_csv), use_container_width=True)

    # ==============================================================================
    # TAB 5: THEORETICAL FOUNDATIONS, PROOFS & COMPLEXITY
    # ==============================================================================
    with tab5:
        st.subheader("Theoretical Formulations, Loss Objectives & Complexity Proofs")

        st.markdown(
            """
        ### 1. LightGCN (SIGIR 2020)
        - **Linear Graph Convolution**: Removes non-linear feature transformation and self-loops:
          $$E^{(k+1)} = \\tilde{A} E^{(k)}, \\quad \\text{where } \\tilde{A} = D^{-\\frac{1}{2}} A D^{-\\frac{1}{2}}$$
        - **Layer Combination**: $E = \\frac{1}{L+1} \\sum_{k=0}^L E^{(k)}$
        - **Loss Function (BPR)**:
          $$\\mathcal{L}_{\\text{BPR}} = \\sum_{(u,i,j) \\in \\mathcal{D}} -\\ln \\sigma(\\hat{y}_{ui} - \\hat{y}_{uj}) + \\lambda \\|\\Theta_0\\|_2^2$$

        ---

        ### 2. SGL — Self-Supervised Graph Learning (SIGIR 2021)
        - **Topological Edge Dropout Augmentation**: Generates dual augmented subgraphs $G_1, G_2$ by dropping edges with probability $p_{\\text{drop}}$:
          $$z_u^{(1)} = \\text{GNN}(G_1, u), \\quad z_u^{(2)} = \\text{GNN}(G_2, u)$$
        - **InfoNCE Contrastive Loss**:
          $$\\mathcal{L}_{\\text{SSL}} = -\\sum_{u \\in \\mathcal{B}} \\log \\frac{\\exp(\\text{sim}(z_u^{(1)}, z_u^{(2)}) / \\tau)}{\\sum_{v \\in \\mathcal{B}} \\exp(\\text{sim}(z_u^{(1)}, z_v^{(2)}) / \\tau)}$$

        ---

        ### 3. SimGCL — Simple Graph Contrastive Learning (SIGIR 2022)
        - **Direct Embedding Noise Perturbation**: Eliminates costly graph reconstruction by directly adding bounded uniform noise during propagation:
          $$e^{(k)\\prime} = e^{(k)} + \\epsilon \\cdot \\frac{\\Delta}{\\|\\Delta\\|_2}, \\quad \\Delta \\sim \\mathcal{U}(0, 1)$$
        - **Computational Advantage**: Operates in $O(1)$ memory without constructing auxiliary adjacency matrices $\\tilde{A}_1, \\tilde{A}_2$.

        ---

        ### 4. Asymptotic Computational Complexity Comparison
        """
        )

        complexity_df = pd.DataFrame([
            {
                "Model": "LightGCN",
                "Graph Construction": "$O(|\\mathcal{E}|)$",
                "Forward Propagation": "$O(L \\cdot |\\mathcal{E}| \\cdot d)$",
                "Contrastive Loss": "None ($0$)",
                "Memory Overhead": "Base ($1\\times$)",
            },
            {
                "Model": "SGL",
                "Graph Construction": "$O(3 \\cdot |\\mathcal{E}|)$",
                "Forward Propagation": "$O(3L \\cdot |\\mathcal{E}| \\cdot d)$",
                "Contrastive Loss": "$O(B^2 \\cdot d)$",
                "Memory Overhead": "High ($3\\times$ adjacency)",
            },
            {
                "Model": "SimGCL",
                "Graph Construction": "$O(|\\mathcal{E}|)$",
                "Forward Propagation": "$O(3L \\cdot |\\mathcal{E}| \\cdot d)$",
                "Contrastive Loss": "$O(B^2 \\cdot d)$",
                "Memory Overhead": "Low ($1\\times$ adjacency)",
            },
        ])
        st.table(complexity_df)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "⚡ Advanced Graph Contrastive Learning Suite | Academic Research Project"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
