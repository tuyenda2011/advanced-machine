from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseRecommender


class SemanticGCL(BaseRecommender):
    """Semantic-Enhanced Graph Contrastive Learning for Recommender Systems.

    Combines ID collaborative embeddings with Dense Semantic Text Features projected
    via an alignment MLP. Integrates Cross-Modal Semantic Contrastive Loss (InfoNCE)
    and enables Zero-Shot Item Representation for cold-start items.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        text_dim: int = 384,
        text_features: Optional[torch.Tensor] = None,
        ssl_temp: float = 0.2,
        ssl_reg: float = 0.1,
        node_dropout: float = 0.0,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.text_dim = text_dim
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg

        if not 0.0 <= node_dropout < 1.0:
            raise ValueError(f"node_dropout must be in [0, 1), got {node_dropout}")
        self.node_dropout = node_dropout
        # Cached propagation graph: validated once per adjacency object, reused across steps.
        self.register_buffer("_cached_norm_adj", None, persistent=False)
        self._adj_cache_key = None

        # Text projection layer to align text feature space with CF embedding space
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Register item text features as buffer if provided
        if text_features is not None:
            if text_features.shape[0] != num_items:
                raise ValueError(
                    f"text_features rows ({text_features.shape[0]}) must equal num_items ({num_items})"
                )
            self.register_buffer("text_features", text_features.float())
        else:
            # Fallback zero features
            self.register_buffer("text_features", torch.zeros((num_items, text_dim)))

        self._init_semantic_weights()

    def _init_semantic_weights(self):
        for m in self.text_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_text_features(self, text_features: torch.Tensor):
        """Update or set text feature buffer."""
        self.register_buffer("text_features", text_features.float().to(self.user_embedding.weight.device))

    def get_projected_text_embeddings(self) -> torch.Tensor:
        """Project dense text features into CF embedding space."""
        return self.text_proj(self.text_features)

    def zero_shot_embed(self, new_text_features: torch.Tensor) -> torch.Tensor:
        """Compute zero-shot item embeddings for brand new/unseen items given their text vectors."""
        device = self.user_embedding.weight.device
        proj = self.text_proj(new_text_features.float().to(device))
        return F.normalize(proj, dim=-1)

    def _validate_connectivity(self, norm_adj: torch.Tensor, strict: bool = False) -> None:
        """Check if any user or item node is isolated in the graph.
        
        Logs warning by default, raises ValueError only if strict=True.
        """
        row_idx = norm_adj.coalesce().indices()[0]
        present_users = torch.unique(row_idx[row_idx < self.num_users])
        present_items = torch.unique(row_idx[row_idx >= self.num_users]) - self.num_users
        n_missing_users = self.num_users - present_users.numel()
        n_missing_items = self.num_items - present_items.numel()
        if n_missing_users or n_missing_items:
            msg = (
                f"Adjacency graph has disconnected nodes: {n_missing_users} users and "
                f"{n_missing_items} items have zero interactions."
            )
            if strict:
                raise ValueError(msg)
            import logging
            logging.getLogger(__name__).warning(msg)

    def _apply_node_dropout(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """Drop all edges incident to randomly selected users/items (training only)."""
        adj = norm_adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        device = indices.device

        drop_u = torch.rand(self.num_users, device=device) < self.node_dropout
        drop_i = torch.rand(self.num_items, device=device) < self.node_dropout

        row, col = indices[0], indices[1]
        row_is_user = row < self.num_users
        col_is_user = col < self.num_users
        row_off = (row - self.num_users).clamp(min=0)
        col_off = (col - self.num_users).clamp(min=0)
        drop_row = torch.where(row_is_user, drop_u[row.clamp(min=0)], drop_i[row_off])
        drop_col = torch.where(col_is_user, drop_u[col.clamp(min=0)], drop_i[col_off])
        keep = ~(drop_row | drop_col)

        return torch.sparse_coo_tensor(
            indices[:, keep], values[keep], adj.size(), device=device, dtype=values.dtype
        ).coalesce()

    def _get_propagation_adj(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """Return cached adjacency (validated once); resample node dropout every pass."""
        if self._adj_cache_key is not norm_adj:
            self._validate_connectivity(norm_adj, strict=False)
            self._cached_norm_adj = norm_adj.coalesce()
            self._adj_cache_key = norm_adj
        adj = self._cached_norm_adj
        if self.training and self.node_dropout > 0:
            adj = self._apply_node_dropout(adj)
        return adj

    def forward(
        self,
        norm_adj: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform graph convolution propagation on semantic-enhanced initial embeddings.

        Args:
            norm_adj: PyTorch Sparse COO normalized bipartite adjacency tensor.
            cached_proj_text: Optional precomputed projected text features of shape (num_items, emb_dim).

        Returns:
            Tuple of (final_user_embeds, final_item_embeds).
        """
        # Resolve validated/cached adjacency (applies node dropout during training)
        norm_adj = self._get_propagation_adj(norm_adj)

        # 1. Base ID embeddings
        u_emb_0 = self.user_embedding.weight
        i_emb_0 = self.item_embedding.weight

        # 2. Fuse item ID embeddings with projected semantic text features
        if cached_proj_text is not None:
            proj_text = cached_proj_text
        else:
            proj_text = self.text_proj(self.text_features)
        fused_i_emb_0 = i_emb_0 + proj_text

        # 3. Stack initial graph state E0 = [Users; Items]
        all_emb = torch.cat([u_emb_0, fused_i_emb_0], dim=0)
        embs_list = [all_emb]

        # 4. Multi-layer Graph Convolution
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            embs_list.append(all_emb)

        # 5. Layer Aggregation: Mean pooling across layer states
        final_embs = torch.stack(embs_list, dim=1).mean(dim=1)
        final_users, final_items = torch.split(final_embs, [self.num_users, self.num_items], dim=0)

        return final_users, final_items

    def compute_semantic_ssl_loss(
        self,
        batch_items: torch.Tensor,
        final_items: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Cross-Modal InfoNCE loss between graph topological embeddings and text semantic features."""
        # Graph topological item embeddings for the batch
        graph_i_emb = F.normalize(final_items[batch_items], dim=-1)

        # Projected text semantic feature for the batch
        if cached_proj_text is not None:
            proj_batch = cached_proj_text[batch_items]
        else:
            proj_batch = self.text_proj(self.text_features[batch_items])
        text_i_emb = F.normalize(proj_batch, dim=-1)

        # Positive pairs similarity: (B,)
        pos_sim = torch.sum(graph_i_emb * text_i_emb, dim=-1) / self.ssl_temp

        # All-pairs similarity matrix: (B, B)
        sim_matrix = torch.matmul(graph_i_emb, text_i_emb.T) / self.ssl_temp

        # InfoNCE loss
        loss = -torch.mean(pos_sim - torch.logsumexp(sim_matrix, dim=-1))
        return self.ssl_reg * loss

