import logging
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

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

    def __init__(self, embedding_dim: int = 64, use_hnsw: bool = True, m: int = 32):
        self.embedding_dim = embedding_dim
        self.use_hnsw = use_hnsw
        self.m = m
        self.index = None
        self.item_embeddings_np: Optional[np.ndarray] = None
        self.item_embeddings_torch: Optional[torch.Tensor] = None
        self.metadata: Dict[int, dict] = {}
        self.num_items = 0

    def build_index(
        self,
        item_embeddings: torch.Tensor,
        metadata: Optional[Dict[int, dict]] = None,
    ):
        """Construct normalized Faiss index from PyTorch item embedding tensor."""
        if isinstance(item_embeddings, torch.Tensor):
            # Normalize to unit sphere for exact Cosine Similarity via Inner Product
            norm_embs = F.normalize(item_embeddings.detach().cpu().float(), dim=-1)
            self.item_embeddings_torch = norm_embs
            self.item_embeddings_np = norm_embs.numpy().astype(np.float32)
        else:
            self.item_embeddings_np = np.ascontiguousarray(item_embeddings, dtype=np.float32)
            faiss.normalize_L2(self.item_embeddings_np)
            self.item_embeddings_torch = torch.from_numpy(self.item_embeddings_np)

        self.num_items, self.embedding_dim = self.item_embeddings_np.shape
        if metadata is not None:
            self.metadata = metadata

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
        excluded_items: Optional[Set[int]] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ) -> List[Tuple[int, float]]:
        """Query top-K items for a user embedding vector with collision exclusion and metadata filtering.

        Args:
            user_vector: User representation tensor of shape (dim,) or (1, dim).
            k: Number of recommendations to return.
            excluded_items: Set of item indices to ignore (e.g. already purchased).
            filter_fn: Optional predicate function f(item_meta) -> bool.

        Returns:
            List of (item_index, similarity_score) tuples.
        """
        if excluded_items is None:
            excluded_items = set()

        if isinstance(user_vector, torch.Tensor):
            u_vec = F.normalize(user_vector.detach().cpu().float().view(1, -1), dim=-1).numpy().astype(np.float32)
        else:
            u_vec = np.ascontiguousarray(user_vector, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(u_vec)

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

            if filter_fn is not None and self.metadata:
                meta = self.metadata.get(item_idx, {})
                if not filter_fn(meta):
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
