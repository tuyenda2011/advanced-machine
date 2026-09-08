"""
Advanced Graph Contrastive Learning Dashboard
============================================
Interactive Research Suite for Recommendation Systems
"""

import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.data.bundle import BundleError, resolve_bundle
from src.data.sparsity import create_sparse_train_set
from src.data.text_encoder import (
    DEFAULT_ENCODER,
    PINNED_REVISION,
    build_user_history_features,
    format_item_text,
    load_training_text,
)
from src.evaluation.evaluator import EVALUATION_PROTOCOL
from src.evaluation.metrics import compute_intra_list_diversity
from src.models.adaptive_gcl import AdaptiveGCL
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.serving.ann_indexer import VectorIndexer
from src.serving.metadata_display import (
    ALL_BRANDS_LABEL,
    ALL_CATEGORIES_LABEL,
    brand_filter_value,
    category_filter_value,
    display_brand,
    display_brand_source,
    display_category,
    display_title,
    format_brand_filter,
    format_category_filter,
    metadata_counts,
    unique_filter_options,
)
from src.serving.recommendations import recommend_exact
from src.utils.checkpoints import (
    get_checkpoint_path,
    get_run_fingerprint,
)
from src.utils.config import load_config
from src.utils.paths import PROJECT_ROOT, find_latest_run_root

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
class MissingCheckpointError(FileNotFoundError):
    """Raised when a requested model has not been trained for this run."""


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint does not match the current code/data identity."""


class TextCacheCompatibilityError(RuntimeError):
    """Raised when the content embedding cache is stale or invalid."""


def load_demo_text(processed_dir: str, mappings: dict):
    """Load the verified text artifact and expose a Demo-specific error."""
    try:
        if (
            str(processed_dir).replace("\\", "/") == "data/processed"
            and (PROJECT_ROOT / "data" / "current.json").exists()
        ):
            processed_dir = str(resolve_bundle().train_dir)
        return load_training_text(processed_dir, mappings)
    except (BundleError, ValueError) as exc:
        raise TextCacheCompatibilityError(
            "Text embedding cache không tương thích với metadata hiện tại. "
            "Hãy chạy lại prepare_data.py."
        ) from exc


def _file_state(path: str | Path) -> dict[str, int | str] | None:
    """Return a cheap cache token for a file without loading large artifacts."""
    target = Path(path)
    if not target.exists():
        return None
    stat = target.stat()
    return {
        "path": str(target.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def get_processed_data_cache_token(
    processed_dir: str | None = None,
    manifest_path: str | Path | None = None,
) -> str:
    """Hash manifest and artifact states before entering Streamlit cache."""
    if processed_dir is None or str(processed_dir).replace("\\", "/") == "data/processed":
        resolved = resolve_bundle()
        processed = resolved.train_dir
        manifest = resolved.manifest_path
    else:
        processed = Path(processed_dir)
        manifest = Path(manifest_path or "data/manifest.json")
    manifest_hash = (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else "missing"
    )
    files = [
        processed / name
        for name in (
            "train.parquet",
            "val.parquet",
            "test.parquet",
            "mappings.pkl",
            "item_text_embeddings.pt",
            "item_text_embeddings.pt.json",
            "disliked_interactions.parquet",
        )
    ]
    payload = {
        "processed_dir": str(processed.resolve()),
        "manifest_sha256": manifest_hash,
        "files": [_file_state(path) for path in files],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@st.cache_data
def load_processed_data_cached(processed_dir: str, cache_token: str):
    """Load preprocessed data with caching."""
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


def load_processed_data(
    processed_dir: str | None = None,
    manifest_path: str | Path | None = None,
):
    """Load processed artifacts using a token computed before the cached call."""
    try:
        if processed_dir is None or str(processed_dir).replace("\\", "/") == "data/processed":
            resolved = resolve_bundle()
            processed = resolved.train_dir
            manifest = resolved.manifest_path
        else:
            processed = Path(processed_dir)
            manifest = Path(manifest_path or "data/manifest.json")
        cache_token = get_processed_data_cache_token(str(processed), manifest)
    except BundleError:
        return None, None, None, None
    return load_processed_data_cached(str(processed), cache_token)


@st.cache_resource
def load_trained_model_cached(
    model_name: str,
    num_users: int,
    num_items: int,
    sparsity: float,
    seed: int,
    data_token: str,
    checkpoint_path: str,
    checkpoint_token: str,
    run_token: str,
):
    """Load trained model with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(model_name, "configs")
    emb_dim = config["model"]["embedding_dim"]
    num_layers = config["model"]["num_layers"]
    train_df, _, _, mappings = load_processed_data()
    train_df_sparse = create_sparse_train_set(train_df, sparsity, seed)

    if not os.path.exists(checkpoint_path):
        raise MissingCheckpointError(
            f"Chưa có checkpoint cho {model_name}, density={sparsity}, seed={seed}. "
            "Hãy train đúng cấu hình này trước khi mở Demo."
        )

    if model_name == "lightgcn":
        model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers)
    elif model_name == "xsimgcl":
        xsim_cfg = config.get("xsimgcl", {})
        model = XSimGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            contrastive_weight=xsim_cfg.get("contrastive_weight", 0.1),
            temperature=xsim_cfg.get("temperature", 0.2),
            epsilon=xsim_cfg.get("epsilon", 0.1),
            contrastive_layer=xsim_cfg.get("contrastive_layer", 1),
        )
    elif model_name == "directau":
        dau_cfg = config.get("directau", {})
        model = DirectAU(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            gamma=dau_cfg.get("gamma", 1.0), t=dau_cfg.get("t", 2.0),
            profile=dau_cfg.get("profile", "project_cosine"),
        )
    elif model_name == "adaptive_gcl":
        ada_cfg = config.get("adaptive_gcl", {})
        try:
            text_dir = str(resolve_bundle().train_dir)
        except BundleError as exc:
            raise TextCacheCompatibilityError(
                "Không tìm thấy bundle dữ liệu đang hoạt động. Hãy chạy prepare_data.py."
            ) from exc
        text_features, item_text_mask = load_demo_text(
            text_dir, mappings
        )
        text_dim = text_features.shape[1]
        user_history_features, user_text_mask = build_user_history_features(
            train_df_sparse, text_features, num_users, item_text_mask
        )
        model = AdaptiveGCL(
            num_users, num_items, embedding_dim=emb_dim, num_layers=num_layers,
            text_dim=text_dim, text_features=text_features,
            ssl_temp=ada_cfg.get("ssl_temp", 0.2),
            ssl_reg=ada_cfg.get("ssl_reg", 0.1),
            dirichlet_reg=ada_cfg.get("dirichlet_reg", 0.0),
            node_dropout=ada_cfg.get("node_dropout", 0.0),
            tau_plus=ada_cfg.get("tau_plus", 0.0),
            user_history_features=user_history_features,
            item_text_mask=item_text_mask,
            user_text_mask=user_text_mask,
            use_item_text=ada_cfg.get("use_item_text", True),
            user_semantic_weight=ada_cfg.get("user_semantic_weight", 0.5),
            layer_aggregation=ada_cfg.get("layer_aggregation", "learnable"),
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_fingerprint = get_run_fingerprint(model_name, sparsity, seed, config)
        stored_fingerprint = ckpt.get("config", {}).get("experiment_fingerprint")
        if stored_fingerprint != expected_fingerprint:
            raise CheckpointCompatibilityError(
                f"Checkpoint của {model_name} đã cũ so với data/config hiện tại. "
                "Hãy train lại model này trước khi dùng Demo."
            )
        model.load_state_dict(ckpt["model_state_dict"])
    except CheckpointCompatibilityError:
        raise
    except (
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        OSError,
        pickle.PickleError,
    ) as exc:
        raise CheckpointCompatibilityError(
            f"Không thể đọc checkpoint của {model_name}. "
            "Checkpoint có thể hỏng hoặc không tương thích; hãy train lại model."
        ) from exc

    model.to(device)
    model.eval()

    from src.data.graph import get_norm_adj_tensor
    norm_adj = get_norm_adj_tensor(train_df_sparse, num_users, num_items, device)

    with torch.no_grad():
        u_embeds, i_embeds = model(norm_adj)

    return model, u_embeds, i_embeds, device


def get_checkpoint_cache_token(
    model_name: str, sparsity: float = 1.0, seed: int = 42
) -> dict:
    """Return checkpoint and run identity used as a model-cache key."""
    config = load_config(model_name, "configs")
    run_root = find_latest_run_root(
        model_name=model_name, sparsity=sparsity, seed=seed
    )
    checkpoint_path = get_checkpoint_path(
        model_name,
        sparsity,
        seed,
        root=str(run_root) if run_root is not None else "results",
    )
    return {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_state": _file_state(checkpoint_path),
        "run_fingerprint": get_run_fingerprint(model_name, sparsity, seed, config),
    }


def load_trained_model(
    model_name: str,
    num_users: int,
    num_items: int,
    sparsity: float = 1.0,
    seed: int = 42,
):
    """Load a model after computing data, checkpoint and config cache tokens."""
    data_token = get_processed_data_cache_token()
    checkpoint = get_checkpoint_cache_token(model_name, sparsity, seed)
    if checkpoint["checkpoint_state"] is None:
        raise MissingCheckpointError(
            f"Chưa có checkpoint cho {model_name}, density={sparsity}, seed={seed}. "
            "Hãy train đúng cấu hình này trước khi mở Demo."
        )
    return load_trained_model_cached(
        model_name,
        num_users,
        num_items,
        sparsity,
        seed,
        data_token,
        checkpoint["checkpoint_path"],
        json.dumps(checkpoint["checkpoint_state"], sort_keys=True),
        checkpoint["run_fingerprint"],
    )


def clear_dashboard_cache() -> None:
    """Clear only dashboard data and model caches."""
    load_processed_data_cached.clear()
    load_trained_model_cached.clear()


# ===============================================================================
# Main Application
# ===============================================================================
def main():
    st.title("⚡ Graph Contrastive Learning Dashboard")
    st.caption("Interactive Research Suite for Top-K Recommendation Systems")

    # Load data
    train_df, _val_df, _test_df, mappings = load_processed_data()

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
                        "ASIN": info.get("original_id", "Unknown ASIN"),
                        "Product": display_title(info, max_length=50),
                        "Brand": display_brand(info),
                        "Brand source": display_brand_source(info),
                        "Category": display_category(info, max_length=50),
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
            brands = [ALL_BRANDS_LABEL] + unique_filter_options(
                item_metadata, brand_filter_value
            )
            selected_brand = st.selectbox(
                "Filter by Brand",
                options=brands,
                index=0,
                format_func=format_brand_filter,
            )

        with col_cat:
            cats = [ALL_CATEGORIES_LABEL] + unique_filter_options(
                item_metadata, category_filter_value
            )
            selected_cat = st.selectbox(
                "Filter by Category",
                options=cats,
                index=0,
                format_func=format_category_filter,
            )

        with col_ann:
            use_ann = st.checkbox("Use ANN Search", value=True)

        counts = metadata_counts(item_metadata.values())
        st.caption(
            f"Metadata: {counts['missing_titles']:,} sản phẩm chưa có title; "
            f"{counts['missing_brands']:,} chưa rõ hãng. Các item này vẫn được giữ trong catalog."
        )

        st.divider()

        # Generate button
        if st.button("🚀 Generate Recommendations", type="primary"):
            if not selected_models:
                st.warning("Please select at least one model")
            else:
                seen_items = set(user_history["i_idx"])
                try:
                    diversity_features, diversity_mask = load_demo_text(
                        "data/processed", mappings
                    )
                except TextCacheCompatibilityError as exc:
                    st.error(str(exc))
                    st.stop()

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
                        has_filter = (
                            selected_brand != ALL_BRANDS_LABEL
                            or selected_cat != ALL_CATEGORIES_LABEL
                        )

                        def make_filter_fn(brand, cat):
                            def filter_fn(meta):
                                return (
                                    (brand == ALL_BRANDS_LABEL or brand_filter_value(meta) == brand)
                                    and (cat == ALL_CATEGORIES_LABEL or category_filter_value(meta) == cat)
                                )
                            return filter_fn

                        if use_ann and not has_filter:
                            indexer = VectorIndexer(embedding_dim=i_embeds.shape[1], use_hnsw=True,
                                                    scoring_metric=model.scoring_metric)
                            indexer.build_index(i_embeds, metadata=item_metadata)
                            ann_results = indexer.query_topk(u_vec, k=10, excluded_items=seen_items)
                            latency_ms = (time.perf_counter() - start_t) * 1000.0
                            topk_ids = [r[0] for r in ann_results]
                            topk_scores = [r[1] for r in ann_results]
                        else:
                            results = recommend_exact(
                                model, u_idx, u_embeds, i_embeds, k=10,
                                excluded_items=seen_items, metadata=item_metadata,
                                filter_fn=make_filter_fn(selected_brand, selected_cat) if has_filter else None,
                            )
                            latency_ms = (time.perf_counter() - start_t) * 1000.0
                            topk_ids = [item for item, _ in results]
                            topk_scores = [score for _, score in results]

                        if not topk_ids:
                            st.info("No products match the selected filters and viewing history.")
                            continue

                        # Metrics
                        ild_score = compute_intra_list_diversity(
                            torch.as_tensor(topk_ids, dtype=torch.long).reshape(1, -1),
                            diversity_features, k=len(topk_ids), item_mask=diversity_mask,
                        ) if len(topk_ids) > 1 else float("nan")

                        novelty_bits = np.mean([
                            -np.log2((item_pop.get(int(iid), 0) + 1) / float(num_users))
                            for iid in topk_ids
                        ]) if len(topk_ids) else 0.0

                        # Display recommendations
                        rec_data = []
                        for rank, (idx, score) in enumerate(zip(topk_ids, topk_scores), start=1):
                            info = item_metadata.get(idx, {})
                            rec_data.append({
                                "Rank": rank,
                                "ASIN": info.get("original_id", "Unknown ASIN"),
                                "Product": display_title(info, max_length=45),
                                "Brand": display_brand(info),
                                "Brand source": display_brand_source(info),
                                "Category": display_category(info, max_length=45),
                                "Score": f"{score:.3f}",
                            })

                        st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Latency", f"{latency_ms:.2f} ms")
                        with m2:
                            st.metric("Content Diversity (shared MiniLM)", f"{ild_score:.3f}" if np.isfinite(ild_score) else "N/A")
                        with m3:
                            st.metric("Novelty", f"{novelty_bits:.2f} bits")

                    except MissingCheckpointError as exc:
                        st.warning(str(exc))
                    except TextCacheCompatibilityError as exc:
                        st.error(str(exc))
                    except CheckpointCompatibilityError as exc:
                        st.error(str(exc))
                    except Exception as e:  # noqa: BLE001 - UI boundary must keep other models usable.
                        st.error(f"Error loading {config['name']}: {e!s}")

    # ===========================================================================
    # TAB 2: Benchmark Results
    # ===========================================================================
    with tabs[1]:
        st.header("Benchmark Results & Statistical Analysis")

        benchmark_root = find_latest_run_root(
            required_relative_path=os.path.join("aggregated", "benchmark_summary.csv"),
            preferred_kinds=("benchmark", "train_all"),
        ) or PROJECT_ROOT / "results"
        benchmark_root = Path(benchmark_root)
        agg_csv = benchmark_root / "aggregated" / "benchmark_summary.csv"

        if os.path.exists(agg_csv):
            df_res = pd.read_csv(agg_csv)
            if "evaluation_protocol" not in df_res or not df_res["evaluation_protocol"].eq(EVALUATION_PROTOCOL).all():
                st.warning("Kết quả dùng giao thức đánh giá cũ hoặc không đồng nhất. Hãy tạo lại benchmark trước khi so sánh metric.")
            st.dataframe(df_res, use_container_width=True)

            st.subheader("Statistical Significance Tests")
            sig_csv = benchmark_root / "aggregated" / "statistical_significance.csv"
            if os.path.exists(sig_csv):
                df_sig = pd.read_csv(sig_csv)
                st.dataframe(df_sig, use_container_width=True)
                st.caption("Xem Holm-adjusted p-value; kết quả dưới 5 seed chỉ mang tính thăm dò.")

            st.subheader("LaTeX Export")
            tex_file = benchmark_root / "aggregated" / "benchmark_table.tex"
            if os.path.exists(tex_file):
                with open(tex_file, "r", encoding="utf-8") as f:
                    tex_code = f.read()
                with st.expander("View LaTeX Code"):
                    st.code(tex_code, language="latex")

            st.subheader("Visualization Charts")
            c1, c2, c3 = st.columns(3)
            with c1:
                figure = benchmark_root / "figures" / "recall_10_by_model.png"
                if figure.exists():
                    st.image(str(figure), caption="Recall@10")
            with c2:
                figure = benchmark_root / "figures" / "diversity_10_by_model.png"
                if figure.exists():
                    st.image(str(figure), caption="Diversity@10")
            with c3:
                figure = benchmark_root / "figures" / "novelty_10_by_model.png"
                if figure.exists():
                    st.image(str(figure), caption="Novelty@10")
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
            figure = benchmark_root / "figures" / "alignment_vs_uniformity.png"
            if figure.exists():
                st.image(str(figure),
                        caption="Alignment vs Uniformity Pareto Frontier")
            else:
                st.info("Run `scripts/generate_plots.py` to generate figures")

        with c2:
            figure = benchmark_root / "figures" / "beyond_accuracy_radar.png"
            if figure.exists():
                st.image(str(figure),
                        caption="6-Dimensional Radar Profile")
            else:
                st.info("Run `scripts/generate_plots.py` to generate figures")

    # ===========================================================================
    # TAB 4: Sparsity Analysis
    # ===========================================================================
    with tabs[3]:
        st.header("Sparsity Robustness & User Activity Analysis")

        st.markdown("**Tail (Low-Activity)** groups users by low training degree. "
                   "This is not an evaluation of unseen-user or unseen-item cold-start.")

        c1, c2 = st.columns(2)
        with c1:
            figure = benchmark_root / "figures" / "sparsity_recall_10_curve.png"
            if figure.exists():
                st.image(str(figure),
                        caption="Sparsity vs Recall@10")
            else:
                st.info("Run benchmark to generate sparsity curves")

        with c2:
            figure = benchmark_root / "figures" / "subgroup_tail_vs_head.png"
            if figure.exists():
                st.image(str(figure),
                        caption="Tail vs Head Performance")
            else:
                st.info("Run benchmark to generate subgroup analysis")

        drop_csv = benchmark_root / "aggregated" / "sparsity_drop25_summary.csv"
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

                        text_str = format_item_text({"title": input_title, "brand": input_brand, "categories": input_cat})
                        if input_desc.strip():
                            text_str += " | " + input_desc.strip()
                        if not text_str.strip():
                            raise ValueError("Provide actual product text for zero-shot inference")
                        encoder = SentenceTransformer(DEFAULT_ENCODER, revision=PINNED_REVISION)
                        text_vec_np = encoder.encode([text_str], normalize_embeddings=True)
                        text_vec = torch.from_numpy(text_vec_np).float()

                        model, u_embeds, _, device = load_trained_model(
                            "adaptive_gcl", num_users, num_items
                        )

                        if hasattr(model, "zero_shot_embed"):
                            item_zero_shot = model.zero_shot_embed(text_vec).to(device)
                        else:
                            raise CheckpointCompatibilityError(
                                "Model adaptive_gcl hiện tại không hỗ trợ zero-shot; "
                                "hãy train lại đúng phiên bản model."
                            )

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

                    except MissingCheckpointError as ex:
                        st.warning(str(ex))
                    except TextCacheCompatibilityError as ex:
                        st.error(str(ex))
                    except CheckpointCompatibilityError as ex:
                        st.error(str(ex))
                    except Exception as ex:  # noqa: BLE001 - UI boundary reports inference failures.
                        st.error(f"Error: {ex!s}")

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
                "formula": "E^{(l)} = \\tilde{A}E^{(l-1)} + \\epsilon \\cdot sign(E^{(l)}) \\odot \\bar{\\Delta}^{(l)}",
                "loss": "\\mathcal{L}_{CL} = -\\log \\frac{exp(sim)}{sum}",
                "desc": "Layer-wise perturbation with final-to-intermediate contrast"
            },
            {
                "name": "DirectAU (KDD '22)",
                "formula": "\\mathcal{L} = \\mathcal{L}_{align} + \\gamma \\mathcal{L}_{uniform}",
                "loss": "No negative sampling required",
                "desc": "Direct optimization of alignment and uniformity"
            },
            {
                "name": "AdaptiveGCL (Course-project model)",
                "formula": "H_i = g_i \\odot E_i + (1-g_i) \\odot W_p X_i",
                "loss": "\\mathcal{L} = \\mathcal{L}_{BPR} + \\mathcal{L}_{semantic} + \\mathcal{L}_{hard} + \\mathcal{L}_{dir}",
                "desc": "Gated text fusion with semantic and graph regularization"
            },
        ]

        for info in model_info:
            with st.expander(f"📐 {info['name']}"):
                st.markdown(f"**{info['desc']}**")
                st.latex(info['formula'])
                st.markdown("**Loss:**")
                st.latex(info['loss'])

        st.subheader("Computational Complexity")

        complexity_data = [
            {"Model": "LightGCN", "Forward": "O(L·E·d)", "Contrastive": "None", "Speed": "1.0×"},
            {"Model": "XSimGCL", "Forward": "O(L·E·d)", "Contrastive": "O(B²·d)", "Speed": "Measure"},
            {"Model": "DirectAU", "Forward": "O(L·E·d)", "Contrastive": "O(B²·d)", "Speed": "Measure"},
            {"Model": "AdaptiveGCL", "Forward": "O(L·E·d)", "Contrastive": "O(B²·d)", "Speed": "Measure"},
        ]

        st.dataframe(pd.DataFrame(complexity_data), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("⚡ Advanced Graph Contrastive Learning Suite | Course Project")

    with st.sidebar:
        st.subheader("Dữ liệu và cache")
        if st.button("Tải lại dữ liệu / model cache"):
            clear_dashboard_cache()
            st.rerun()


if __name__ == "__main__":
    main()
