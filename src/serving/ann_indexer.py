import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("Faiss is not installed. VectorIndexer will use PyTorch matrix multiplication fallback.")


class VectorIndexer:
    """High-performance Vector Search Engine for sub-millisecond Top-K recommendation using Faiss / HNSW."""

    def __init__(self, embedding_dim: int = 64, use_hnsw: bool = True, m: int = 32,
                 scoring_metric: str = "cosine"):
        if scoring_metric not in {"cosine", "dot_product"}:
            raise ValueError("scoring_metric must be cosine or dot_product")
        self.scoring_metric = scoring_metric
        self.embedding_dim = embedding_dim
        self.use_hnsw = use_hnsw
        self.m = m
        self.index: Any = None
        self.item_embeddings_np: np.ndarray | None = None
        self.item_embeddings_torch: torch.Tensor | None = None
        self.metadata: dict[int, dict] = {}
        self.num_items = 0

    def build_index(
        self,
        item_embeddings: torch.Tensor,
        metadata: dict[int, dict] | None = None,
    ):
        """Build an inner-product index, normalizing only for cosine scoring."""
        embeds = torch.as_tensor(item_embeddings).detach().cpu().float().clone()
        if embeds.ndim != 2 or not torch.isfinite(embeds).all():
            raise ValueError("Item embeddings must be a finite matrix")
        if self.scoring_metric == "cosine":
            embeds = F.normalize(embeds, dim=-1)
        self.item_embeddings_torch = embeds.contiguous()
        self.item_embeddings_np = self.item_embeddings_torch.numpy()
        self.num_items, self.embedding_dim = embeds.shape
        self.metadata = metadata or {}
        self.index = None

        if FAISS_AVAILABLE:
            if self.use_hnsw:
                # HNSW (Hierarchical Navigable Small World) Index for sub-millisecond approximate nearest neighbor
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, self.m, faiss.METRIC_INNER_PRODUCT)
                self.index.hnsw.efSearch = 64
                self.index.hnsw.efConstruction = 64
            else:
                # Exact Inner Product Flat Index
                self.index = faiss.IndexFlatIP(self.embedding_dim)

            self.index.add(self.item_embeddings_np)
            logger.info(
                f"Built Faiss index ({'HNSW' if self.use_hnsw else 'FlatIP'}) for {self.num_items:,} items."
            )
        else:
            logger.info(f"Using PyTorch fallback index for {self.num_items:,} items.")

    def query_topk(
        self,
        user_vector: torch.Tensor,
        k: int = 10,
        excluded_items: set[int] | None = None,
        filter_fn: Callable[[dict], bool] | None = None,
    ) -> list[tuple[int, float]]:
        """Query top-K items for a user embedding vector with collision exclusion and metadata filtering.

        Args:
            user_vector: User representation tensor of shape (dim,) or (1, dim).
            k: Number of recommendations to return.
            excluded_items: Set of item indices to ignore (e.g. already purchased).
            filter_fn: Optional predicate function f(item_meta) -> bool.

        Returns:
            List of (item_index, similarity_score) tuples.
        """
        if k < 0:
            raise ValueError("k must be nonnegative")
        if self.item_embeddings_torch is None:
            raise RuntimeError("Build the index before querying")
        if k == 0 or self.num_items == 0:
            return []
        excluded_items = excluded_items or set()
        query = torch.as_tensor(user_vector).detach().cpu().float().reshape(1, -1)
        if query.shape[1] != self.embedding_dim or not torch.isfinite(query).all():
            raise ValueError("Query must be finite and match the embedding dimension")
        if self.scoring_metric == "cosine":
            query = F.normalize(query, dim=-1)
        u_vec = query.contiguous().numpy()

        # Exact search within the filtered catalog avoids losing matches outside
        # a fixed ANN candidate window, including empty metadata/filter matches.
        if filter_fn is not None:
            eligible = [i for i in range(self.num_items)
                        if i not in excluded_items and filter_fn(self.metadata.get(i, {}))]
            if not eligible:
                return []
            candidates = torch.tensor(eligible, dtype=torch.long)
            scores = (query @ self.item_embeddings_torch[candidates].T)[0]
            values, positions = scores.topk(min(k, len(eligible)))
            return list(zip(candidates[positions].tolist(), values.tolist()))

        # Retrieve extra candidates to account for excluded items and metadata filtering
        search_k = min(self.num_items, max(k * 5, k + len(excluded_items) + 100))

        if FAISS_AVAILABLE and self.index is not None:
            distances, indices = self.index.search(u_vec, search_k)
            cand_indices = indices[0]
            cand_scores = distances[0]
        else:
            # PyTorch fallback
            u_t = torch.from_numpy(u_vec)
            scores = torch.matmul(u_t, self.item_embeddings_torch.T).squeeze(0)
            cand_scores, cand_indices = torch.topk(scores, search_k)
            cand_scores = cand_scores.numpy()
            cand_indices = cand_indices.numpy()

        results = []
        for item_idx, score in zip(cand_indices, cand_scores):
            item_idx = int(item_idx)
            if item_idx < 0 or item_idx in excluded_items:
                continue

            results.append((item_idx, float(score)))
            if len(results) >= k:
                break

        return results

    def measure_latency_ms(self, user_vector: torch.Tensor, num_runs: int = 50) -> float:
        """Measure average query latency in milliseconds."""
        # Warmup
        self.query_topk(user_vector, k=10)

        t0 = time.perf_counter()
        for _ in range(num_runs):
            self.query_topk(user_vector, k=10)
        t1 = time.perf_counter()

        return ((t1 - t0) / num_runs) * 1000.0
