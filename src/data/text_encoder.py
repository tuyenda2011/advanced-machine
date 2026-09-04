import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_user_history_features(
    train_df: pd.DataFrame,
    item_text_features: torch.Tensor,
    num_users: int,
) -> torch.Tensor:
    """Mean-pool item text features from each user's training history."""
    text_features = item_text_features.float().cpu()
    user_features = torch.zeros(
        (num_users, text_features.shape[1]), dtype=text_features.dtype
    )
    user_indices = torch.from_numpy(train_df["u_idx"].to_numpy(copy=True)).long()
    item_indices = torch.from_numpy(train_df["i_idx"].to_numpy(copy=True)).long()
    counts = torch.zeros(num_users, dtype=text_features.dtype)

    chunk_size = 100_000
    for start in range(0, len(user_indices), chunk_size):
        end = start + chunk_size
        user_chunk = user_indices[start:end]
        item_chunk = item_indices[start:end]
        user_features.index_add_(0, user_chunk, text_features[item_chunk])
        counts.index_add_(
            0,
            user_chunk,
            torch.ones(len(user_chunk), dtype=text_features.dtype),
        )

    return user_features / counts.clamp_min(1.0).unsqueeze(1)


def format_item_text(meta: dict) -> str:
    """Format item metadata dictionary into structured semantic text description."""
    title = meta.get("title", "").strip() or "Unknown Electronics Product"
    brand = meta.get("brand", "").strip()
    category = meta.get("categories", "").strip()

    parts = [title]
    if brand and brand != "Unknown":
        parts.append(f"Brand: {brand}")
    if category and category != "Unknown":
        parts.append(f"Category: {category}")

    return " | ".join(parts)


def encode_item_metadata(
    item_metadata: Dict[int, dict],
    num_items: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    force_recompute: bool = False,
    allow_fallback: bool = False,
) -> torch.Tensor:
    """Extract dense semantic text embeddings for all mapped items from metadata.

    Args:
        item_metadata: Mapping from contiguous item integer index to metadata dict.
        num_items: Total number of items in mapped dataset.
        model_name: Pretrained SentenceTransformer model identifier.
        batch_size: Mini-batch size for transformer inference.
        device: 'cuda' or 'cpu'. Defaults to auto-detect.
        save_path: Optional file path to cache embeddings (.pt).
        force_recompute: Whether to ignore cached tensor and recompute.

    Returns:
        torch.Tensor of shape (num_items, feature_dim), L2-normalized.
    """
    if save_path and os.path.exists(save_path) and not force_recompute:
        logger.info(f"Loading cached item text embeddings from {save_path}...")
        embeddings = torch.load(save_path, map_location="cpu", weights_only=False)
        if isinstance(embeddings, torch.Tensor) and embeddings.shape[0] == num_items:
            return embeddings
        logger.warning(
            f"Cached embedding size {embeddings.shape if isinstance(embeddings, torch.Tensor) else None} "
            f"does not match num_items={num_items}. Recomputing..."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Compiling text descriptions for {num_items:,} items...")
    text_corpus = []
    for i_idx in range(num_items):
        meta = item_metadata.get(i_idx, {})
        text_corpus.append(format_item_text(meta))

    logger.info(f"Encoding {len(text_corpus):,} items with {model_name} on {device}...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        embeddings_np = model.encode(
            text_corpus,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        embeddings_tensor = torch.from_numpy(embeddings_np).float()
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"SentenceTransformer encoding failed for {model_name}"
            ) from e
        logger.warning(f"SentenceTransformer encoding failed or offline ({e}). Using TF-IDF fallback...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        tfidf_mat = vectorizer.fit_transform(text_corpus)
        svd = TruncatedSVD(n_components=min(128, tfidf_mat.shape[1] - 1), random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat)
        embeddings_tensor = F.normalize(torch.from_numpy(svd_mat).float(), dim=-1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings_tensor, save_path)
        logger.info(f"Saved {embeddings_tensor.shape} item text embeddings to {save_path}")

    return embeddings_tensor
