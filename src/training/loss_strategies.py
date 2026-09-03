"""Loss computation strategies for different recommendation models.

Implements the Strategy Pattern to decouple Trainer from model-specific loss computations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from src.losses.bpr import BPRLoss
from src.losses.contrastive import InfoNCELoss
from src.losses.directau import DirectAULoss


class LossOutput(NamedTuple):
    """Output from loss computation."""
    total_loss: torch.Tensor
    bpr_loss: float
    cl_loss: float
    extra_losses: Dict[str, torch.Tensor]


class LossStrategy(ABC):
    """Abstract base class for loss computation strategies."""

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        """Compute loss for the current batch.

        Args:
            model: The recommendation model
            norm_adj: Normalized adjacency matrix
            u_batch: User indices for the batch
            pos_batch: Positive item indices
            neg_batch: Negative item indices (may be None for DirectAU)
            config: Configuration dictionary

        Returns:
            LossOutput containing total loss and components
        """
        pass

    @property
    @abstractmethod
    def requires_negative_samples(self) -> bool:
        """Whether this strategy requires negative sampling."""
        pass


class BPRStrategy(LossStrategy):
    """Standard BPR loss strategy for LightGCN."""

    def __init__(self, weight_decay: float = 1e-4):
        self.bpr_loss_fn = BPRLoss(weight_decay=weight_decay)
        self.weight_decay = weight_decay

    @property
    def requires_negative_samples(self) -> bool:
        return True

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        if neg_batch is None:
            raise ValueError("BPRStrategy requires negative samples")

        u_embeds, i_embeds = model(norm_adj)
        pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
        neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)
        neg_emb0 = model.item_embedding(neg_batch)

        total_loss, bpr_loss = self.bpr_loss_fn(
            pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
        )

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=bpr_loss.item(),
            cl_loss=0.0,
            extra_losses={},
        )


class XSimGCLStrategy(LossStrategy):
    """BPR + Contrastive SSL strategy for XSimGCL."""

    def __init__(
        self,
        weight_decay: float = 1e-4,
        contrastive_weight: float = 0.1,
        temperature: float = 0.2,
    ):
        self.bpr_loss_fn = BPRLoss(weight_decay=weight_decay)
        self.cl_loss_fn = InfoNCELoss(temperature=temperature)
        self.weight_decay = weight_decay
        self.contrastive_weight = contrastive_weight

    @property
    def requires_negative_samples(self) -> bool:
        return True

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        if neg_batch is None:
            raise ValueError("XSimGCLStrategy requires negative samples")

        u_embeds, i_embeds = model(norm_adj)
        pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
        neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)
        neg_emb0 = model.item_embedding(neg_batch)

        total_loss, bpr_loss = self.bpr_loss_fn(
            pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
        )

        # Contrastive SSL loss
        u_p1, i_p1 = model.forward_perturbed(u_embeds, i_embeds)
        u_p2, i_p2 = model.forward_perturbed(u_embeds, i_embeds)

        cl_loss = self.cl_loss_fn(
            u_p1[u_batch], u_p2[u_batch], i_p1[pos_batch], i_p2[pos_batch]
        )
        total_loss = total_loss + self.contrastive_weight * cl_loss

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=bpr_loss.item(),
            cl_loss=cl_loss.item(),
            extra_losses={"cl_loss": cl_loss},
        )


class DirectAUStrategy(LossStrategy):
    """DirectAU loss strategy (no negative sampling)."""

    def __init__(self, gamma: float = 1.0, t: float = 2.0, weight_decay: float = 1e-4):
        self.directau_loss_fn = DirectAULoss(gamma=gamma, t=t, weight_decay=weight_decay)
        self.gamma = gamma
        self.t = t
        self.weight_decay = weight_decay

    @property
    def requires_negative_samples(self) -> bool:
        return False

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        u_embeds, i_embeds = model(norm_adj)
        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)

        total_loss, align_loss, unif_loss = self.directau_loss_fn(
            u_embeds[u_batch], i_embeds[pos_batch], u_emb0, pos_emb0
        )

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=align_loss.item(),
            cl_loss=unif_loss.item(),
            extra_losses={"align_loss": align_loss, "unif_loss": unif_loss},
        )


class AdaptiveGCLStrategy(LossStrategy):
    """BPR + Semantic SSL + Dirichlet Energy for AdaptiveGCL."""

    def __init__(
        self,
        weight_decay: float = 1e-4,
        ssl_temp: float = 0.2,
        ssl_reg: float = 0.1,
        dirichlet_reg: float = 0.01,
    ):
        self.bpr_loss_fn = BPRLoss(weight_decay=weight_decay)
        self.weight_decay = weight_decay
        self.ssl_temp = ssl_temp
        self.ssl_reg = ssl_reg
        self.dirichlet_reg = dirichlet_reg

    @property
    def requires_negative_samples(self) -> bool:
        return True

    def compute_loss(
        self,
        model: nn.Module,
        norm_adj: torch.Tensor,
        u_batch: torch.Tensor,
        pos_batch: torch.Tensor,
        neg_batch: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> LossOutput:
        if neg_batch is None:
            raise ValueError("AdaptiveGCLStrategy requires negative samples")

        u_embeds, i_embeds = model(norm_adj)
        pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
        neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

        u_emb0 = model.user_embedding(u_batch)
        pos_emb0 = model.item_embedding(pos_batch)
        neg_emb0 = model.item_embedding(neg_batch)

        total_loss, bpr_loss = self.bpr_loss_fn(
            pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
        )

        # Semantic SSL loss
        if hasattr(model, "compute_semantic_ssl_loss"):
            cl_loss = model.compute_semantic_ssl_loss(pos_batch, i_embeds)
            total_loss = total_loss + cl_loss
        else:
            cl_loss = torch.tensor(0.0, device=u_batch.device)

        # Dirichlet Energy regularization
        extra_losses = {"cl_loss": cl_loss}
        if hasattr(model, "dirichlet_reg") and model.dirichlet_reg > 0:
            all_final = torch.cat([u_embeds, i_embeds], dim=0)
            dir_energy = model.compute_dirichlet_energy(norm_adj, all_final)
            dir_loss = model.dirichlet_reg * torch.clamp(1.0 - dir_energy, min=0.0)
            total_loss = total_loss + dir_loss
            extra_losses["dir_loss"] = dir_loss

        return LossOutput(
            total_loss=total_loss,
            bpr_loss=bpr_loss.item(),
            cl_loss=cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss,
            extra_losses=extra_losses,
        )


def get_loss_strategy(model_name: str, config: Dict[str, Any]) -> LossStrategy:
    """Factory function to get the appropriate loss strategy for a model.

    Args:
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        Appropriate LossStrategy instance
    """
    train_cfg = config.get("training", {})
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    if model_name == "lightgcn":
        return BPRStrategy(weight_decay=weight_decay)

    elif model_name == "xsimgcl":
        xsim_cfg = config.get("xsimgcl", {})
        return XSimGCLStrategy(
            weight_decay=weight_decay,
            contrastive_weight=xsim_cfg.get("contrastive_weight", 0.1),
            temperature=xsim_cfg.get("temperature", 0.2),
        )

    elif model_name == "directau":
        dau_cfg = config.get("directau", {})
        return DirectAUStrategy(
            gamma=dau_cfg.get("gamma", 1.0),
            t=dau_cfg.get("t", 2.0),
            weight_decay=weight_decay,
        )

    elif model_name == "adaptive_gcl":
        ada_cfg = config.get("adaptive_gcl", {})
        return AdaptiveGCLStrategy(
            weight_decay=weight_decay,
            ssl_temp=ada_cfg.get("ssl_temp", 0.2),
            ssl_reg=ada_cfg.get("ssl_reg", 0.1),
            dirichlet_reg=ada_cfg.get("dirichlet_reg", 0.01),
        )

    else:
        # Default to BPR
        return BPRStrategy(weight_decay=weight_decay)
