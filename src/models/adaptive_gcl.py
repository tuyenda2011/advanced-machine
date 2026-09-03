import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class AdaptiveGCL(BaseRecommender):
    """Adaptive Gated Graph Contrastive Learning for Recommender Systems (AdaptiveGCL).

    Key Innovations:
    1. Adaptive Multimodal Gating: Learns dynamic element-wise gate g in (0, 1) between
       Collaborative ID and Semantic Text embeddings to eliminate Modality Competition.
    2. User Semantic Profiler: Enriches user representations by aggregating historical text semantics.
    3. Learnable Layer-Attention: Dynamically weights multi-hop propagation layers (alpha_0, ..., alpha_L)
       to suppress Graph Oversmoothing.
    4. Zero-Shot Representation: Seamless zero-shot cold item embedding for long-tail items.
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
        dirichlet_reg: float = 0.01,
        node_dropout: float = 0.0,
        user_history_features: Optional[torch.Tensor] = None,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.text_dim = text_dim
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.dirichlet_reg = dirichlet_reg
        self.node_dropout = node_dropout

        # Cached propagation graph buffer
        self.register_buffer("_cached_norm_adj", None, persistent=False)
        self._adj_cache_key = None

        # 1. Text Projection MLP: maps text_dim -> embedding_dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # 2. Adaptive Multimodal Gating MLP: inputs [e_id || e_text] -> gate in (0, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

        # 3. User Semantic Profiler MLP (if user text history is present)
        self.user_semantic_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # 4. Learnable Layer-Attention: assigns importance weights to each layer [E0, E1, ..., EL]
        self.layer_attention_weights = nn.Parameter(torch.zeros(num_layers + 1))

        # Register item text features as buffer if provided
        if text_features is not None:
            if text_features.shape[0] != num_items:
                raise ValueError(
                    f"text_features rows ({text_features.shape[0]}) must match num_items ({num_items})"
                )
            self.register_buffer("text_features", text_features.float())
        else:
            self.register_buffer("text_features", torch.zeros((num_items, text_dim)))

        # Register user history features if provided
        if user_history_features is not None:
            self.register_buffer("user_history_features", user_history_features.float())
        else:
            self.register_buffer("user_history_features", None)

        self._init_adaptive_weights()

    def _init_adaptive_weights(self):
        """Initialize projection and gating layers using Xavier uniform initialization."""
        for module in [self.text_proj, self.gate_mlp, self.user_semantic_mlp]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Initialize layer attention uniformly
        nn.init.zeros_(self.layer_attention_weights)

    def set_text_features(self, text_features: torch.Tensor):
        """Update or set text feature tensor buffer."""
        device = self.user_embedding.weight.device
        self.register_buffer("text_features", text_features.float().to(device))

    def get_gated_item_embeddings(
        self, cached_proj_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptively gated initial item representations and gating weights."""
        i_id_emb = self.item_embedding.weight
        if cached_proj_text is not None:
            proj_text = cached_proj_text
        else:
            proj_text = self.text_proj(self.text_features)

        # Compute element-wise adaptive gate g in (0, 1)
        gate_input = torch.cat([i_id_emb, proj_text], dim=-1)
        g = self.gate_mlp(gate_input)

        # Fused item embedding: g * ID + (1 - g) * Text
        fused_items = g * i_id_emb + (1.0 - g) * proj_text
        return fused_items, g

    def get_user_initial_embeddings(self) -> torch.Tensor:
        """Compute user initial embeddings, optionally enriched with semantic history."""
        u_emb = self.user_embedding.weight
        if self.user_history_features is not None:
            user_sem = self.user_semantic_mlp(self.user_history_features)
            u_emb = u_emb + 0.5 * user_sem
        return u_emb

    def zero_shot_embed(self, new_text_features: torch.Tensor) -> torch.Tensor:
        """Compute zero-shot item embeddings for brand new/unseen items given their text vectors."""
        device = self.user_embedding.weight.device
        proj = self.text_proj(new_text_features.float().to(device))
        return F.normalize(proj, dim=-1)

    def _apply_node_dropout(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """Apply random node dropout during training to create contrastive views."""
        adj = norm_adj.coalesce()
        indices = adj.indices()
        values = adj.values()
        device = indices.device
        num_total_nodes = self.num_users + self.num_items

        # Unified node dropout mask across all user + item nodes
        drop_nodes = torch.rand(num_total_nodes, device=device) < self.node_dropout

        row, col = indices[0], indices[1]
        keep = ~(drop_nodes[row] | drop_nodes[col])

        return torch.sparse_coo_tensor(
            indices[:, keep], values[keep], adj.size(), device=device, dtype=values.dtype
        ).coalesce()

    def _get_propagation_adj(self, norm_adj: torch.Tensor) -> torch.Tensor:
        """Resolve cached adjacency matrix with proper identity tracking.

        Uses object id() for cache key to prevent unnecessary coalesce operations.
        """
        if self._adj_cache_key != id(norm_adj):
            self._cached_norm_adj = norm_adj.coalesce()
            self._adj_cache_key = id(norm_adj)
        adj = self._cached_norm_adj
        if self.training and self.node_dropout > 0:
            adj = self._apply_node_dropout(adj)
        return adj

    def forward(
        self,
        norm_adj: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform adaptive gated graph propagation with learnable layer attention.

        Args:
            norm_adj: PyTorch Sparse COO normalized bipartite adjacency tensor.
            cached_proj_text: Optional precomputed projected text features of shape (num_items, emb_dim).

        Returns:
            Tuple of (final_user_embeds, final_item_embeds).
        """
        norm_adj = self._get_propagation_adj(norm_adj)

        # 1. Initial Gated Item State and User State
        u_emb_0 = self.get_user_initial_embeddings()
        i_emb_0, _ = self.get_gated_item_embeddings(cached_proj_text=cached_proj_text)

        # 2. Stack initial graph state E0 = [Users; Items]
        all_emb = torch.cat([u_emb_0, i_emb_0], dim=0)
        layer_embs = [all_emb]

        # 3. Multi-layer Graph Convolution
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            layer_embs.append(all_emb)

        # 4. Learnable Layer Attention Aggregation: E = sum(alpha_l * E_l)
        stacked_embs = torch.stack(layer_embs, dim=1)  # (N_nodes, num_layers + 1, dim)
        attn_weights = F.softmax(self.layer_attention_weights, dim=0)  # (num_layers + 1,)
        final_embs = torch.sum(stacked_embs * attn_weights.view(1, -1, 1), dim=1)

        final_users, final_items = torch.split(final_embs, [self.num_users, self.num_items], dim=0)
        return final_users, final_items

    def compute_dirichlet_energy(
        self, norm_adj: torch.Tensor, final_embs: torch.Tensor
    ) -> torch.Tensor:
        """Compute Dirichlet Energy: E_dir = 0.5 * Tr(X^T (I - A) X).

        Higher energy indicates better distinction among node representations (less oversmoothing).
        """
        adj = norm_adj.coalesce()
        norm_embs = F.normalize(final_embs, dim=-1)
        # Lap_X = X - A * X
        ax = torch.sparse.mm(adj, norm_embs)
        diff = norm_embs - ax
        dirichlet_energy = 0.5 * torch.sum(norm_embs * diff)
        return dirichlet_energy

    def compute_semantic_ssl_loss(
        self,
        batch_items: torch.Tensor,
        final_items: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Cross-Modal InfoNCE loss between graph topological embeddings and text features."""
        graph_i_emb = F.normalize(final_items[batch_items], dim=-1)

        if cached_proj_text is not None:
            proj_batch = cached_proj_text[batch_items]
        else:
            proj_batch = self.text_proj(self.text_features[batch_items])
        text_i_emb = F.normalize(proj_batch, dim=-1)

        pos_sim = torch.sum(graph_i_emb * text_i_emb, dim=-1) / self.ssl_temp
        sim_matrix = torch.matmul(graph_i_emb, text_i_emb.T) / self.ssl_temp

        loss = -torch.mean(pos_sim - torch.logsumexp(sim_matrix, dim=-1))
        return self.ssl_reg * loss
