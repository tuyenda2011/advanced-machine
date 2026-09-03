"""
Advanced Graph Contrastive Learning Dashboard
============================================
Interactive Research Suite for Recommendation Systems
"""

import os
import pickle
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from src.models.adaptive_gcl import AdaptiveGCL
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.serving.ann_indexer import VectorIndexer
from src.utils.config import load_config
from src.utils.checkpoints import find_checkpoint


# Custom CSS
st.markdown("""
<style>
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }

    /* Better expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* Metric cards spacing */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Button hover effect */
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* Hide default footer */
    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Page Configuration
st.set_page_config(
    page_title="Graph Contrastive Learning Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Helper Functions
# ==============================================================================
@st.cache_data
def load_processed_data():
    """Load preprocessed data with caching."""
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
def load_trained_model(model_name: str, num_users: int, num_items: int, sparsity: float = 1.0, seed: int = 42):
    """Load trained model with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(model_name, "configs")
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]

    if model_name == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif model_name == "xsimgcl":
        xsim_cfg = config.get("xsimgcl", {})
        model = XSimGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            contrastive_weight=xsim_cfg.get("contrastive_weight", 0.1),
        )
    elif model_name == "directau":
        dau_cfg = config.get("directau", {})
        model = DirectAU(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            gamma=dau_cfg.get("gamma", 1.0), t=dau_cfg.get("t", 2.0),
        )
    elif model_name == "adaptive_gcl":
        ada_cfg = config.get("adaptive_gcl", {})
        text_emb_path = "data/processed/item_text_embeddings.pt"
        text_features = None
        if os.path.exists(text_emb_path):
            text_features = torch.load(text_emb_path, map_location="cpu", weights_only=False)
            text_dim = text_features.shape[1]
        else:
            text_dim = ada_cfg.get("text_dim", 384)
        model = AdaptiveGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            text_dim=text_dim, text_features=text_features,
            ssl_temp=ada_cfg.get("ssl_temp", 0.2),
            ssl_reg=ada_cfg.get("ssl_reg", 0.1),
            dirichlet_reg=ada_cfg.get("dirichlet_reg", 0.01),
        )

    # Use centralized checkpoint finder
    checkpoint_path = find_checkpoint(model_name, sparsity, seed)

    if checkpoint_path and os.path.exists(checkpoint_path):
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


# ==============================================================================
# Main Application
# ==============================================================================
def main():
    st.title("⚡ Graph Contrastive Learning Dashboard")
    st.caption("Interactive Research Suite for Top-K Recommendation Systems")

    # Load data
    train_df, val_df, test_df, mappings = load_processed_data()

    if train_df is None:
        st.error("⚠️ Dataset not found. Please run `python scripts/prepare_data.py` first.")
        return

    num_users = mappings["stats"]["num_users"]
    num_items = mappings["stats"]["num_items"]
    item_metadata = mappings["item_metadata"]
    user2id = mappings["user2id"]
    item_pop = train_df["i_idx"].value_counts().to_dict()

    # Model configuration
    model_configs = {
        "lightgcn": {"name": "LightGCN", "desc": "Pure CF Baseline (SIGIR '20)"},
        "xsimgcl": {"name": "XSimGCL", "desc": "Contrastive SSL (TKDE '23)"},
        "directau": {"name": "DirectAU", "desc": "Alignment & Uniformity (KDD '22)"},
        "adaptive_gcl": {"name": "AdaptiveGCL", "desc": "Multimodal Gated GCL"},
    }

    # Tabs
    tabs = st.tabs([
        "🎯 Recommendations",
        "📊 Benchmark",
        "🌐 Geometry",
        "📉 Sparsity",
        "🔮 Zero-Shot",
        "📘 Theory"
    ])

    # ===========================================================================
    # TAB 1: Interactive Recommendation
    # ===========================================================================
    with tabs[0]:
        st.header("Interactive Top-K Recommendation")

        col_user, col_info = st.columns([1, 2])

        with col_user:
            id2user = {v: k for k, v in user2id.items()}
            u_idx = st.number_input(
                "Select User Index",
                min_value=0,
                max_value=max(0, num_users - 1),
                value=min(100, num_users - 1),
            )
            reviewer_id = id2user.get(u_idx, f"User_{u_idx}")
            st.info(f"**Reviewer:** {reviewer_id}")

        with col_info:
            user_history = train_df[train_df["u_idx"] == u_idx]
            st.metric("User History", f"{len(user_history)} rated items")

            with st.expander("View Purchase History"):
                history_items = user_history.sort_values(by="timestamp", ascending=False).head(10)
                h_data = []
                for _, row in history_items.iterrows():
                    info = item_metadata.get(row["i_idx"], {})
                    h_data.append({
                        "Product": info.get("title", "Unknown")[:50],
                        "Brand": info.get("brand", "Unknown"),
                    })
                st.dataframe(pd.DataFrame(h_data), use_container_width=True, hide_index=True)

        st.divider()

        # Model selection
        st.subheader("Select Models to Compare")
        selected_models = st.multiselect(
            "Models",
            options=list(model_configs.keys()),
            default=["lightgcn", "xsimgcl"],
            format_func=lambda x: f"{model_configs[x]['name']} - {model_configs[x]['desc']}"
        )

        col_brand, col_cat, col_ann = st.columns(3)
        with col_brand:
            brands = ["All Brands"] + sorted([
                v.get("brand", "") for v in item_metadata.values()
                if v.get("brand") and v.get("brand") != "Unknown"
            ])[:50]
            selected_brand = st.selectbox("Filter by Brand", options=brands, index=0)

        with col_cat:
            cats = ["All Categories"] + sorted([
                v.get("categories", "").split(" > ")[-1]
                for v in item_metadata.values() if v.get("categories")
            ])[:50]
            selected_cat = st.selectbox("Filter by Category", options=cats, index=0)

        with col_ann:
            use_ann = st.checkbox("Use ANN Search", value=True)

        st.divider()

        # Generate button
        if st.button("🚀 Generate Recommendations", type="primary"):
            if not selected_models:
                st.warning("Please select at least one model")
            else:
                seen_items = set(user_history["i_idx"])

                for model_name in selected_models:
                    config = model_configs[model_name]
                    st.subheader(f"{config['name']} - {config['desc']}")

                    try:
                        with st.spinner(f"Loading {config['name']}..."):
                            model, u_embeds, i_embeds, device = load_trained_model(
                                model_name, num_users, num_items
                            )

                        start_t = time.perf_counter()
                        u_vec = u_embeds[u_idx:u_idx + 1]

                        # Build filter
                        has_filter = (selected_brand != "All Brands") or (selected_cat != "All Categories")

                        def make_filter_fn(brand, cat):
                            def filter_fn(meta):
                                if brand != "All Brands" and meta.get("brand") != brand:
                                    return False
                                if cat != "All Categories" and cat not in meta.get("categories", ""):
                                    return False
                                return True
                            return filter_fn

                        if use_ann and not has_filter:
                            indexer = VectorIndexer(embedding_dim=i_embeds.shape[1], use_hnsw=True)
                            indexer.build_index(i_embeds, metadata=item_metadata)
                            ann_results = indexer.query_topk(u_vec, k=10, excluded_items=seen_items)
                            latency_ms = (time.perf_counter() - start_t) * 1000.0
                            topk_ids = [r[0] for r in ann_results]
                            topk_scores = [r[1] for r in ann_results]
                        else:
                            scores = torch.matmul(u_vec, i_embeds.T).squeeze(0)
                            if seen_items:
                                scores[torch.tensor(list(seen_items), device=device)] = -1e9

                            if has_filter:
                                filter_fn = make_filter_fn(selected_brand, selected_cat)
                                valid_indices = [
                                    idx for idx in range(num_items)
                                    if filter_fn(item_metadata.get(idx, {}))
                                ]
                                if valid_indices:
                                    mask = torch.ones(num_items, dtype=torch.bool, device=device)
                                    mask[torch.tensor(valid_indices, device=device)] = False
                                    scores[mask] = -1e9

                            topk_scores, topk_indices = torch.topk(scores, k=10)
                            latency_ms = (time.perf_counter() - start_t) * 1000.0
                            topk_ids = topk_indices.cpu().numpy()
                            topk_scores = topk_scores.cpu().numpy()

                        # Metrics
                        if len(topk_ids) > 1:
                            rec_embeds = F.normalize(i_embeds[torch.tensor(topk_ids, device=device)], dim=-1)
                            sim_mat = torch.matmul(rec_embeds, rec_embeds.T)
                            triu_idx = torch.triu_indices(len(topk_ids), len(topk_ids), offset=1)
                            ild_score = (1.0 - sim_mat[triu_idx[0], triu_idx[1]]).mean().item()
                        else:
                            ild_score = 0.0

                        novelty_bits = np.mean([
                            -np.log2((item_pop.get(int(iid), 0) + 1) / float(num_users))
                            for iid in topk_ids
                        ]) if topk_ids else 0.0

                        # Display recommendations
                        rec_data = []
                        for rank, (idx, score) in enumerate(zip(topk_ids, topk_scores), start=1):
                            info = item_metadata.get(idx, {})
                            rec_data.append({
                                "Rank": rank,
                                "Product": info.get("title", "Unknown")[:45] + "...",
                                "Brand": info.get("brand", "Unknown"),
                                "Score": f"{score:.3f}",
                            })

                        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Latency", f"{latency_ms:.2f} ms")
                        with m2:
                            st.metric("Diversity (ILD)", f"{ild_score:.3f}")
                        with m3:
                            st.metric("Novelty", f"{novelty_bits:.2f} bits")

                    except Exception as e:
                        st.error(f"Error loading {config['name']}: {str(e)}")

    # ===========================================================================
    # TAB 2: Benchmark Results
    # ===========================================================================
    with tabs[1]:
        st.header("Benchmark Results & Statistical Analysis")

        agg_csv = os.path.join("results", "aggregated", "benchmark_summary.csv")

        if os.path.exists(agg_csv):
            df_res = pd.read_csv(agg_csv)
            st.dataframe(df_res, use_container_width=True)

            st.subheader("Statistical Significance Tests")
            sig_csv = os.path.join("results", "aggregated", "statistical_significance.csv")
            if os.path.exists(sig_csv):
                df_sig = pd.read_csv(sig_csv)
                st.dataframe(df_sig, use_container_width=True)
                st.caption("Levels: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")

            st.subheader("LaTeX Export")
            tex_file = os.path.join("results", "aggregated", "benchmark_table.tex")
            if os.path.exists(tex_file):
                with open(tex_file, "r", encoding="utf-8") as f:
                    tex_code = f.read()
                with st.expander("View LaTeX Code"):
                    st.code(tex_code, language="latex")

            st.subheader("Visualization Charts")
            c1, c2, c3 = st.columns(3)
            with c1:
                if os.path.exists("results/figures/recall_10_by_model.png"):
                    st.image("results/figures/recall_10_by_model.png", caption="Recall@10")
            with c2:
                if os.path.exists("results/figures/diversity_10_by_model.png"):
                    st.image("results/figures/diversity_10_by_model.png", caption="Diversity@10")
            with c3:
                if os.path.exists("results/figures/novelty_10_by_model.png"):
                    st.image("results/figures/novelty_10_by_model.png", caption="Novelty@10")
        else:
            st.info("📁 No benchmark results yet. Run `python scripts/benchmark_all.py` first.")

    # ===========================================================================
    # TAB 3: Representation Geometry
    # ===========================================================================
    with tabs[2]:
        st.header("Representation Geometry Analysis")

        st.markdown("Based on **Wang & Isola (ICML 2020)**, contrastive learning on the hypersphere:")
        st.markdown("- **Alignment**: Distance between positive user-item pairs")
        st.markdown("- **Uniformity**: Distribution uniformity of all representations")

        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("results/figures/alignment_vs_uniformity.png"):
                st.image("results/figures/alignment_vs_uniformity.png",
                        caption="Alignment vs Uniformity Pareto Frontier")
            else:
                st.info("Run `scripts/generate_plots.py` to generate figures")

        with c2:
            if os.path.exists("results/figures/beyond_accuracy_radar.png"):
                st.image("results/figures/beyond_accuracy_radar.png",
                        caption="6-Dimensional Radar Profile")
            else:
                st.info("Run `scripts/generate_plots.py` to generate figures")

    # ===========================================================================
    # TAB 4: Sparsity Analysis
    # ===========================================================================
    with tabs[3]:
        st.header("Sparsity Robustness & Cold-Start Analysis")

        st.markdown("Graph Contrastive Learning is designed to rescue **Tail (Cold-Start)** users "
                   "with minimal interaction history.")

        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("results/figures/sparsity_recall_10_curve.png"):
                st.image("results/figures/sparsity_recall_10_curve.png",
                        caption="Sparsity vs Recall@10")
            else:
                st.info("Run benchmark to generate sparsity curves")

        with c2:
            if os.path.exists("results/figures/subgroup_tail_vs_head.png"):
                st.image("results/figures/subgroup_tail_vs_head.png",
                        caption="Tail vs Head Performance")
            else:
                st.info("Run benchmark to generate subgroup analysis")

        drop_csv = os.path.join("results", "aggregated", "sparsity_drop25_summary.csv")
        if os.path.exists(drop_csv):
            st.subheader("Performance Degradation at 25% Sparsity")
            st.dataframe(pd.read_csv(drop_csv), use_container_width=True)

    # ===========================================================================
    # TAB 5: Zero-Shot Recommender
    # ===========================================================================
    with tabs[4]:
        st.header("Zero-Shot Product Recommendation")

        st.markdown("Test multimodal generalization on **brand new products** with **ZERO** historical data. "
                   "Uses Sentence-Transformers to project semantic text into CF space.")

        with st.container():
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                input_title = st.text_input(
                    "Product Title",
                    value="Sony WH-1000XM5 Wireless Headphones",
                )
                input_desc = st.text_area(
                    "Description / Specifications",
                    value="Industry-leading noise cancellation, 30-hour battery, crystal clear calls",
                )
            with col_t2:
                input_brand = st.text_input("Brand", value="Sony")
                input_cat = st.text_input("Category", value="Electronics > Audio > Headphones")

            if st.button("🔍 Find Target Customers", type="primary"):
                with st.spinner("Encoding text semantics..."):
                    try:
                        from sentence_transformers import SentenceTransformer

                        text_str = f"{input_title} | {input_brand} | {input_cat} | {input_desc}"
                        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                        text_vec_np = encoder.encode([text_str], normalize_embeddings=True)
                        text_vec = torch.from_numpy(text_vec_np).float()

                        model, u_embeds, _, device = load_trained_model(
                            "adaptive_gcl", num_users, num_items
                        )

                        if hasattr(model, "zero_shot_embed"):
                            item_zero_shot = model.zero_shot_embed(text_vec).to(device)
                        else:
                            item_zero_shot = F.normalize(torch.randn((1, 64), device=device), dim=-1)

                        user_scores = torch.matmul(u_embeds, item_zero_shot.T).squeeze(-1)
                        topk_users, topk_scores = torch.topk(user_scores, k=10)

                        st.success("✅ Zero-shot prediction complete!")

                        rows = []
                        for rank, (u_id, sc) in enumerate(
                            zip(topk_users.cpu().numpy(), topk_scores.cpu().numpy()), start=1
                        ):
                            r_id = id2user.get(int(u_id), f"User_{u_id}")
                            u_hist = len(train_df[train_df["u_idx"] == u_id])
                            rows.append({
                                "Rank": rank,
                                "User": r_id,
                                "Score": f"{sc:.4f}",
                                "History": f"{u_hist} items",
                            })

                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    except Exception as ex:
                        st.error(f"Error: {str(ex)}")

    # ===========================================================================
    # TAB 6: Theoretical Foundations
    # ===========================================================================
    with tabs[5]:
        st.header("Theoretical Foundations & Complexity")

        # Model cards
        model_info = [
            {
                "name": "LightGCN (SIGIR '20)",
                "formula": "E^{(k+1)} = \\tilde{A} E^{(k)}",
                "loss": "\\mathcal{L}_{BPR} = -\\ln \\sigma(\\hat{y}_{ui} - \\hat{y}_{uj})",
                "desc": "Linear graph convolution without feature transformation"
            },
            {
                "name": "XSimGCL (TKDE '23)",
                "formula": "E' = E + \\epsilon \\cdot \\frac{\\Delta}{\\|\\Delta\\|}",
                "loss": "\\mathcal{L}_{CL} = -\\log \\frac{exp(sim)}{sum}",
                "desc": "Final-layer perturbation for efficiency"
            },
            {
                "name": "DirectAU (KDD '22)",
                "formula": "\\mathcal{L} = \\mathcal{L}_{align} + \\gamma \\mathcal{L}_{uniform}",
                "loss": "No negative sampling required",
                "desc": "Direct optimization of alignment and uniformity"
            },
            {
                "name": "AdaptiveGCL (Multimodal)",
                "formula": "H_i = E_i + W_p X_i",
                "loss": "\\mathcal{L} = \\mathcal{L}_{BPR} + \\mathcal{L}_{semantic}",
                "desc": "Cross-modal semantic alignment"
            },
        ]

        for info in model_info:
            with st.expander(f"📐 {info['name']}"):
                st.markdown(f"**{info['desc']}**")
                st.latex(info['formula'])
                st.markdown(f"**Loss:**")
                st.latex(info['loss'])

        st.subheader("Computational Complexity")

        complexity_data = [
            {"Model": "LightGCN", "Forward": "O(L·E·d)", "Contrastive": "None", "Speed": "1.0×"},
            {"Model": "SGL", "Forward": "O(3L·E·d)", "Contrastive": "2× GNN", "Speed": "0.35×"},
            {"Model": "XSimGCL", "Forward": "O(L·E·d)", "Contrastive": "O(N·d)", "Speed": "0.90×"},
            {"Model": "DirectAU", "Forward": "O(L·E·d)", "Contrastive": "None", "Speed": "1.20×"},
            {"Model": "AdaptiveGCL", "Forward": "O(L·E·d)", "Contrastive": "O(B·d)", "Speed": "0.85×"},
        ]

        st.dataframe(pd.DataFrame(complexity_data), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("⚡ Advanced Graph Contrastive Learning Suite | Academic Research")


if __name__ == "__main__":
    main()
