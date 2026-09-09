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
        fusion_mode: str = "convex",
        residual_alpha_init: float = 0.1,
        residual_alpha_max: float = 1.0,
        ssl_item_mask: Optional[torch.Tensor] = None,
        user_semantic_gate: bool = False,
        ssl_target: str = "projected",
    ):
        if not isinstance(use_item_text, bool):
            raise ValueError("use_item_text must be a boolean")
        if not math.isfinite(user_semantic_weight) or user_semantic_weight < 0:
            raise ValueError("user_semantic_weight must be finite and nonnegative")
        if not math.isfinite(ssl_reg) or ssl_reg < 0:
            raise ValueError("ssl_reg must be finite and nonnegative")
        if layer_aggregation not in {"learnable", "mean", "anchored"}:
            raise ValueError("layer_aggregation must be learnable, mean or anchored")
        if not isinstance(user_semantic_gate, bool):
            raise ValueError("user_semantic_gate must be boolean")
        if ssl_target not in {"projected", "frozen_text"}:
            raise ValueError("ssl_target must be projected or frozen_text")
        if fusion_mode not in {"convex", "residual", "bounded_residual"}:
            raise ValueError("fusion_mode must be convex, residual or bounded_residual")
        if (
            not math.isfinite(residual_alpha_init)
            or not math.isfinite(residual_alpha_max)
            or residual_alpha_init <= 0
            or residual_alpha_max <= 0
            or residual_alpha_init >= residual_alpha_max
        ):
            raise ValueError(
                "residual_alpha_init and residual_alpha_max must be finite with "
                "0 < residual_alpha_init < residual_alpha_max"
            )
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.use_item_text = use_item_text
        self.user_semantic_weight = user_semantic_weight
        self.layer_aggregation = layer_aggregation
        self.user_semantic_gate = user_semantic_gate
        self.ssl_target = ssl_target
        self.text_dim = text_dim
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.dirichlet_reg = dirichlet_reg
        self.node_dropout = node_dropout
        self.fusion_mode = fusion_mode
        self.residual_alpha_init = residual_alpha_init
        self.residual_alpha_max = residual_alpha_max
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
        if fusion_mode in {"residual", "bounded_residual"}:
            initial_ratio = residual_alpha_init / residual_alpha_max
            initial_logit = math.log(initial_ratio / (1.0 - initial_ratio))
            # Deterministic scalar initialization; convex mode keeps its
            # original parameter set and RNG sequence.
            self.residual_alpha_logit = nn.Parameter(torch.tensor(initial_logit))

        # Register item text features as buffer if provided
        if text_features is not None and item_text_mask is None:
            raise ValueError("item_text_mask is required with text_features")
        if user_history_features is not None and user_text_mask is None:
            raise ValueError("user_text_mask is required with user_history_features")
        for label, mask, count in (
            ("item", item_text_mask, num_items),
            ("ssl_item", ssl_item_mask if ssl_item_mask is not None else item_text_mask, num_items),
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
        # Optional heads are initialized after legacy parameters so their
        # addition does not change the shared control's initialization.
        if user_semantic_gate:
            self.user_gate = nn.Linear(embedding_dim * 2, 1)
            nn.init.zeros_(self.user_gate.weight)
            nn.init.constant_(self.user_gate.bias, math.log(0.1 / 0.9))
        if ssl_target == "frozen_text":
            self.ssl_graph_proj = nn.Linear(embedding_dim, text_dim, bias=False)
            nn.init.xavier_uniform_(self.ssl_graph_proj.weight)

    def optimizer_param_groups(self, mlp_weight_decay: float = 0.0) -> list[dict]:
        """Adam L2 on MLP matrices only; ID regularization remains in BPR."""
        if not math.isfinite(mlp_weight_decay) or mlp_weight_decay < 0:
            raise ValueError("mlp_weight_decay must be finite and nonnegative")
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            is_mlp = name.startswith(("text_proj.", "gate_mlp.", "user_semantic_mlp.", "user_gate.", "ssl_graph_proj."))
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
        # Keep the legacy setter's single-mask behavior; callers with a
        # separate SSL policy can override it explicitly via set_ssl_item_mask.
        self.ssl_item_text_mask = item_text_mask.to(device)
        self.register_buffer(
            "text_features", text_features.float().to(device), persistent=False
        )

    def set_ssl_item_mask(self, ssl_item_mask: torch.Tensor) -> None:
        """Update the item eligibility mask used by semantic SSL only."""
        if ssl_item_mask.dtype != torch.bool or ssl_item_mask.shape != (self.num_items,):
            raise ValueError("Invalid ssl item text mask")
        self.ssl_item_text_mask = ssl_item_mask.to(self.user_embedding.weight.device)

    @property
    def residual_alpha(self) -> torch.Tensor | None:
        """Return the learned residual scale, or ``None`` for convex fusion."""
        if self.fusion_mode not in {"residual", "bounded_residual"}:
            return None
        return self.residual_alpha_max * torch.sigmoid(self.residual_alpha_logit)

    def _fusion_components(
        self,
        item_indices: torch.Tensor | None = None,
        cached_proj_text: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute item fusion and its contributions with one shared formula."""
        if item_indices is None:
            item_indices = torch.arange(
                self.num_items, device=self.item_embedding.weight.device
            )
        item_id = self.item_embedding.weight[item_indices]
        if not self.use_item_text:
            return (
                item_id,
                torch.ones_like(item_id),
                item_id,
                torch.zeros_like(item_id),
                None,
            )
        if cached_proj_text is not None:
            proj_text = cached_proj_text[item_indices]
        else:
            proj_text = self.text_proj(self.text_features[item_indices])
        gate_input = torch.cat([item_id, proj_text], dim=-1)
        gate = self.gate_mlp(gate_input)
        usable = self.item_text_mask[item_indices][:, None]
        gate = torch.where(usable, gate, torch.ones_like(gate))
        if self.fusion_mode in {"residual", "bounded_residual"}:
            alpha = self.residual_alpha
            if self.fusion_mode == "bounded_residual":
                proj_text = self._bound_semantic_norm(proj_text, item_id)
            text_contribution = alpha * (1.0 - gate) * proj_text
            text_contribution = torch.where(
                usable, text_contribution, torch.zeros_like(text_contribution)
            )
            id_contribution = item_id
        else:
            id_contribution = gate * item_id
            text_contribution = (1.0 - gate) * proj_text
        fused = id_contribution + text_contribution
        return fused, gate, id_contribution, text_contribution, proj_text

    def get_gated_item_embeddings(
        self, cached_proj_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptively gated initial item representations and gating weights."""
        fused, gate, _, _, _ = self._fusion_components(
            cached_proj_text=cached_proj_text
        )
        return fused, gate

    @staticmethod
    def _bound_semantic_norm(semantic: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Cap each semantic row by its ID norm without growing small rows.

        Detach the ID budget so semantic gradients cannot enlarge it directly.
        Keep the semantic norm differentiable to remove the radial incentive
        to grow a projection that has already reached the cap.
        """
        budget = reference.detach().norm(dim=-1, keepdim=True)
        scale = (budget / semantic.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        return semantic * scale

    def _user_semantic_contribution(self, user_id, user_sem, usable):
        if self.fusion_mode == "bounded_residual":
            user_sem = self._bound_semantic_norm(user_sem, user_id)
        if self.user_semantic_gate:
            gate = torch.sigmoid(self.user_gate(torch.cat([
                F.normalize(user_id, dim=-1), F.normalize(user_sem, dim=-1)
            ], dim=-1)))
            user_sem = gate * user_sem
        return self.user_semantic_weight * torch.where(
            usable[:, None], user_sem, torch.zeros_like(user_sem)
        )

    def get_user_initial_embeddings(self) -> torch.Tensor:
        """Compute user initial embeddings, optionally enriched with semantic history."""
        u_emb = self.user_embedding.weight
        if self.user_history_features is not None and self.user_semantic_weight > 0:
            user_sem = self.user_semantic_mlp(self.user_history_features)
            u_emb = u_emb + self._user_semantic_contribution(u_emb, user_sem, self.user_text_mask)
        return u_emb

    def zero_shot_embed(self, new_text_features: torch.Tensor) -> torch.Tensor:
        """Experimental text-only embeddings; not evidence of cold-start accuracy."""
        if not self.use_item_text and (self.ssl_reg == 0 or self.ssl_target == "frozen_text"):
            raise RuntimeError(
                "Text projection is inactive in this ablation; zero-shot is unavailable"
            )
        device = self.user_embedding.weight.device
        proj = self.text_proj(new_text_features.float().to(device))
        return F.normalize(proj, dim=-1)

    @torch.no_grad()
    def collect_diagnostics(
        self,
        item_indices: Optional[torch.Tensor] = None,
        user_indices: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Collect lightweight, deterministic representation diagnostics.

        The caller supplies fixed indices and invokes this after validation.
        No random operation, parameter update or persistent cache mutation is
        performed, so enabling diagnostics cannot change the training path.
        """
        device = self.user_embedding.weight.device
        if item_indices is None:
            item_indices = torch.arange(min(self.num_items, 4096), device=device)
        else:
            item_indices = item_indices.to(device=device, dtype=torch.long)
        if user_indices is None:
            user_indices = torch.arange(min(self.num_users, 4096), device=device)
        else:
            user_indices = user_indices.to(device=device, dtype=torch.long)

        diagnostics: dict[str, float] = {}
        layer_weights = self.get_layer_weights()
        entropy = -(layer_weights * torch.log(layer_weights.clamp_min(1e-12))).sum()
        for index, value in enumerate(layer_weights):
            diagnostics[f"layer_weight_{index}"] = float(value.item())
        diagnostics["layer_weight_entropy"] = float(entropy.item())
        if item_indices.numel():
            diagnostics["ssl_eligible_item_fraction"] = float(
                self.ssl_item_text_mask[item_indices].float().mean().item()
            )
        if self.residual_alpha is not None:
            diagnostics["residual_alpha"] = float(self.residual_alpha.item())
            diagnostics["residual_alpha_max"] = float(self.residual_alpha_max)

        if item_indices.numel() and self.use_item_text:
            _, gate, id_contrib, text_contrib, item_text = self._fusion_components(
                item_indices=item_indices
            )
            usable = self.item_text_mask[item_indices]
            usable_gate = gate[usable]
            if usable_gate.numel():
                diagnostics["gate_mean"] = float(usable_gate.mean().item())
                diagnostics["gate_p10"] = float(torch.quantile(usable_gate, 0.10).item())
                diagnostics["gate_p90"] = float(torch.quantile(usable_gate, 0.90).item())
                diagnostics["gate_near_zero_fraction"] = float((usable_gate < 0.1).float().mean().item())
                diagnostics["gate_near_one_fraction"] = float((usable_gate > 0.9).float().mean().item())
            item_id = self.item_embedding.weight[item_indices]
            diagnostics["item_id_norm_mean"] = float(item_id.norm(dim=-1).mean().item())
            diagnostics["item_text_norm_mean"] = float(item_text.norm(dim=-1).mean().item())
            diagnostics["item_id_contribution_norm_mean"] = float(id_contrib.norm(dim=-1).mean().item())
            diagnostics["item_text_contribution_norm_mean"] = float(text_contrib.norm(dim=-1).mean().item())
            diagnostics["item_text_to_id_ratio"] = float(
                (text_contrib.norm(dim=-1) / item_id.norm(dim=-1).clamp_min(1e-12)).mean().item()
            )
            diagnostics["usable_item_text_fraction"] = float(usable.float().mean().item())
        else:
            diagnostics["gate_mean"] = 1.0
            diagnostics["gate_p10"] = 1.0
            diagnostics["gate_p90"] = 1.0
            diagnostics["gate_near_zero_fraction"] = 0.0
            diagnostics["gate_near_one_fraction"] = 1.0
            diagnostics.setdefault("ssl_eligible_item_fraction", 0.0)

        if (
            user_indices.numel()
            and self.user_history_features is not None
            and self.user_semantic_weight > 0
        ):
            user_id = self.user_embedding.weight[user_indices]
            user_sem = self.user_semantic_mlp(self.user_history_features[user_indices])
            usable_users = self.user_text_mask[user_indices]
            weighted_sem = self._user_semantic_contribution(user_id, user_sem, usable_users)
            diagnostics["user_id_norm_mean"] = float(user_id.norm(dim=-1).mean().item())
            diagnostics["user_semantic_norm_mean"] = float(user_sem.norm(dim=-1).mean().item())
            diagnostics["user_semantic_weighted_norm_mean"] = float(weighted_sem.norm(dim=-1).mean().item())
            diagnostics["user_semantic_to_id_ratio"] = float(
                (weighted_sem.norm(dim=-1) / user_id.norm(dim=-1).clamp_min(1e-12)).mean().item()
            )
            diagnostics["usable_user_text_fraction"] = float(usable_users.float().mean().item())
        return diagnostics

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

    def get_layer_weights(self) -> torch.Tensor:
        uniform = torch.full_like(self.layer_attention_weights, 1.0 / (self.num_layers + 1))
        if self.layer_aggregation == "mean":
            return uniform
        learned = F.softmax(self.layer_attention_weights, dim=0)
        # Half of the mass always covers all graph depths; learned weights
        # cannot bypass message passing by concentrating entirely on layer 0.
        return 0.5 * uniform + 0.5 * learned if self.layer_aggregation == "anchored" else learned

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
        attn_weights = self.get_layer_weights()
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
        _, weighted = self.compute_semantic_ssl_components(
            batch_items, final_items, cached_proj_text=cached_proj_text
        )
        return weighted

    def compute_semantic_ssl_components(
        self,
        batch_items: torch.Tensor,
        final_items: torch.Tensor,
        cached_proj_text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw and weighted semantic SSL losses for diagnostics."""
        if self.ssl_reg == 0:
            zero = final_items.sum() * 0.0
            return zero, zero
        unique_items = torch.unique(batch_items)
        unique_items = unique_items[self.ssl_item_text_mask[unique_items]]
        if unique_items.numel() < 2:
            zero = final_items.sum() * 0.0
            return zero, zero
        graph_i_emb = final_items[unique_items]

        if self.ssl_target == "frozen_text":
            # The target remains the frozen encoder output. Fusion's learned
            # projection cannot move both sides of the supervision together.
            raw_loss = self.debiased_ssl.compute_debiased_contrastive_loss(
                self.ssl_graph_proj(graph_i_emb), self.text_features[unique_items].detach()
            )
            return raw_loss, self.ssl_reg * raw_loss
        if cached_proj_text is not None:
            proj_batch = cached_proj_text[unique_items]
        else:
            proj_batch = self.text_proj(self.text_features[unique_items])

        raw_loss = self.debiased_ssl.compute_debiased_contrastive_loss(
            graph_i_emb, proj_batch
        )
        return raw_loss, self.ssl_reg * raw_loss
