import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.debiased_infonce import DebiasedInfoNCELoss
from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class AdaptiveGCL(BaseRecommender):
    """Experimental multimodal graph recommender for the course project.

    It combines gated ID/text item representations, pooled text histories for
    users, learnable layer aggregation, and semantic contrastive supervision.
    ``zero_shot_embed`` exposes a text-only representation for separate cold-item
    experiments; the standard benchmark remains a warm-start ranking protocol.
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
        dirichlet_reg: float = 0.0,
        node_dropout: float = 0.0,
        tau_plus: float = 0.0,
        user_history_features: Optional[torch.Tensor] = None,
        item_text_mask: Optional[torch.Tensor] = None,
        user_text_mask: Optional[torch.Tensor] = None,
        use_item_text: bool = True,
        user_semantic_weight: float = 0.5,
        layer_aggregation: str = "learnable",
    ):
        if not isinstance(use_item_text, bool):
            raise ValueError("use_item_text must be a boolean")
        if not math.isfinite(user_semantic_weight) or user_semantic_weight < 0:
            raise ValueError("user_semantic_weight must be finite and nonnegative")
        if not math.isfinite(ssl_reg) or ssl_reg < 0:
            raise ValueError("ssl_reg must be finite and nonnegative")
        if layer_aggregation not in {"learnable", "mean"}:
            raise ValueError("layer_aggregation must be learnable or mean")
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.use_item_text = use_item_text
        self.user_semantic_weight = user_semantic_weight
        self.layer_aggregation = layer_aggregation
        self.text_dim = text_dim
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.dirichlet_reg = dirichlet_reg
        self.node_dropout = node_dropout
        self.debiased_ssl = DebiasedInfoNCELoss(
            temperature=ssl_temp,
            tau_plus=tau_plus,
        )

        # Cached propagation graph buffer
        self.register_buffer("_cached_norm_adj", None, persistent=False)
        self._adj_cache_key: Optional[int] = None

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
            nn.Linear(text_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Global layer logits, shared by all nodes; keep the state_dict key.
        self.layer_attention_weights = nn.Parameter(torch.zeros(num_layers + 1))

        # Register item text features as buffer if provided
        if text_features is not None and item_text_mask is None:
            raise ValueError("item_text_mask is required with text_features")
        if user_history_features is not None and user_text_mask is None:
            raise ValueError("user_text_mask is required with user_history_features")
        for label, mask, count in (
            ("item", item_text_mask, num_items),
            ("user", user_text_mask, num_users),
        ):
            if mask is not None and (
                mask.dtype != torch.bool or mask.shape != (count,)
            ):
                raise ValueError(f"Invalid {label} text mask")
            self.register_buffer(
                f"{label}_text_mask",
                mask.clone()
                if mask is not None
                else torch.zeros(count, dtype=torch.bool),
                persistent=False,
            )
        if text_features is not None:
            if text_features.shape != (num_items, text_dim):
                raise ValueError(
                    f"text_features must have shape ({num_items}, {text_dim})"
                )
            self.register_buffer(
                "text_features", text_features.float(), persistent=False
            )
        else:
            self.register_buffer(
                "text_features", torch.zeros((num_items, text_dim)), persistent=False
            )

        # Register user history features if provided
        if user_history_features is not None:
            if user_history_features.shape != (num_users, text_dim):
                raise ValueError(
                    f"user_history_features must have shape ({num_users}, {text_dim})"
                )
            self.register_buffer(
                "user_history_features",
                user_history_features.float(),
                persistent=False,
            )
        else:
            self.register_buffer("user_history_features", None, persistent=False)

        self._init_adaptive_weights()

    def optimizer_param_groups(self, mlp_weight_decay: float = 0.0) -> list[dict]:
        """Adam L2 on MLP matrices only; ID regularization remains in BPR."""
        if not math.isfinite(mlp_weight_decay) or mlp_weight_decay < 0:
            raise ValueError("mlp_weight_decay must be finite and nonnegative")
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            is_mlp = name.startswith(("text_proj.", "gate_mlp.", "user_semantic_mlp."))
            (decay if is_mlp and parameter.ndim == 2 else no_decay).append(parameter)
        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": mlp_weight_decay},
        ]

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

    def set_text_features(
        self, text_features: torch.Tensor, item_text_mask: torch.Tensor
    ):
        """Update or set text feature tensor buffer."""
        device = self.user_embedding.weight.device
        if (
            item_text_mask.dtype != torch.bool
            or item_text_mask.shape != (self.num_items,)
            or text_features.shape != (self.num_items, self.text_dim)
        ):
            raise ValueError("Invalid updated text features/mask")
        self.item_text_mask = item_text_mask.to(device)
        self.register_buffer(
            "text_features", text_features.float().to(device), persistent=False
        )

    def get_gated_item_embeddings(
        self, cached_proj_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptively gated initial item representations and gating weights."""
        i_id_emb = self.item_embedding.weight
        if not self.use_item_text:
            return i_id_emb, torch.ones_like(i_id_emb)
        if cached_proj_text is not None:
            proj_text = cached_proj_text
        else:
            proj_text = self.text_proj(self.text_features)

        # Compute element-wise adaptive gate g in (0, 1)
        gate_input = torch.cat([i_id_emb, proj_text], dim=-1)
        g = self.gate_mlp(gate_input)

        # Fused item embedding: g * ID + (1 - g) * Text
        fused_items = g * i_id_emb + (1.0 - g) * proj_text
        fused_items = torch.where(self.item_text_mask[:, None], fused_items, i_id_emb)
        g = torch.where(self.item_text_mask[:, None], g, torch.ones_like(g))
        return fused_items, g

    def get_user_initial_embeddings(self) -> torch.Tensor:
        """Compute user initial embeddings, optionally enriched with semantic history."""
        u_emb = self.user_embedding.weight
        if self.user_history_features is not None and self.user_semantic_weight > 0:
            user_sem = self.user_semantic_mlp(self.user_history_features)
            u_emb = u_emb + self.user_semantic_weight * torch.where(
                self.user_text_mask[:, None], user_sem, torch.zeros_like(user_sem)
            )
        return u_emb

    def zero_shot_embed(self, new_text_features: torch.Tensor) -> torch.Tensor:
        """Experimental text-only embeddings; not evidence of cold-start accuracy."""
        if not self.use_item_text and self.ssl_reg == 0:
            raise RuntimeError(
                "Text projection is inactive in this ablation; zero-shot is unavailable"
            )
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
            indices[:, keep],
            values[keep],
            adj.size(),
            device=device,
            dtype=values.dtype,
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
        """Perform gated propagation with global layer aggregation (not per-node attention).

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

        # 4. Global weighted aggregation or fixed uniform aggregation.
        stacked_embs = torch.stack(layer_embs, dim=1)  # (N_nodes, num_layers + 1, dim)
        if self.layer_aggregation == "mean":
            final_embs = stacked_embs.mean(dim=1)
        else:
            attn_weights = F.softmax(self.layer_attention_weights, dim=0)
            final_embs = torch.sum(stacked_embs * attn_weights.view(1, -1, 1), dim=1)

        final_users, final_items = torch.split(
            final_embs, [self.num_users, self.num_items], dim=0
        )
        return final_users, final_items

    def compute_dirichlet_energy(
        self, norm_adj: torch.Tensor, final_embs: torch.Tensor
    ) -> torch.Tensor:
        """Compute node-averaged energy: Tr(X^T (I - A) X) / (2N).

        X has normalized rows. High energy does not guarantee high rank or
        useful rankings: bipartite sign-separated rank-one embeddings can
        maximize it. This is an experimental diagnostic, not a collapse test.
        """
        adj = norm_adj.coalesce()
        norm_embs = F.normalize(final_embs, dim=-1)
        # Lap_X = X - A * X
        ax = torch.sparse.mm(adj, norm_embs)
        diff = norm_embs - ax
        dirichlet_energy = 0.5 * torch.sum(norm_embs * diff) / norm_embs.size(0)
        return dirichlet_energy

    def compute_dirichlet_regularization(
        self, norm_adj: torch.Tensor, final_embs: torch.Tensor
    ) -> torch.Tensor:
        """Optional energy-maximization experiment, disabled by default."""
        energy = self.compute_dirichlet_energy(norm_adj, final_embs)
        return -self.dirichlet_reg * energy

    def compute_semantic_ssl_loss(
        self,
        batch_items: torch.Tensor,
        final_items: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Cross-Modal InfoNCE loss between graph topological embeddings and text features."""
        if self.ssl_reg == 0:
            return final_items.sum() * 0.0
        unique_items = torch.unique(batch_items)
        unique_items = unique_items[self.item_text_mask[unique_items]]
        if unique_items.numel() < 2:
            return final_items.sum() * 0.0
        graph_i_emb = final_items[unique_items]

        if cached_proj_text is not None:
            proj_batch = cached_proj_text[unique_items]
        else:
            proj_batch = self.text_proj(self.text_features[unique_items])

        loss = self.debiased_ssl.compute_debiased_contrastive_loss(
            graph_i_emb, proj_batch
        )
        return self.ssl_reg * loss
