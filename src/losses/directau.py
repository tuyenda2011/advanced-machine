import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAULoss(nn.Module):
    """Direct Alignment and Uniformity (DirectAU) Loss (KDD '22).

    Directly optimizes the alignment of positive user-item pairs and the uniformity
    of user and item representations on the unit hypersphere without negative sampling.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        t: float = 2.0,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.gamma = gamma
        self.t = t
        self.weight_decay = weight_decay

    def compute_alignment_loss(
        self, u_norm: torch.Tensor, i_norm: torch.Tensor
    ) -> torch.Tensor:
        """Compute alignment loss: ||u - i||^2."""
        return (u_norm - i_norm).norm(p=2, dim=-1).pow(2).mean()

    def compute_uniformity_loss(self, embeds_norm: torch.Tensor) -> torch.Tensor:
        """Compute uniformity loss: log E_{x,y} [ exp(-t * ||x - y||^2) ]."""
        batch_size = embeds_norm.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeds_norm.device)

        # Pairwise distance: ||x - y||^2 = 2 - 2 * <x, y>
        sim_matrix = torch.matmul(embeds_norm, embeds_norm.T)
        dist_sq = 2.0 - 2.0 * sim_matrix.clamp(min=-1.0, max=1.0)

        # Exclude self-distances (diagonal elements)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeds_norm.device)
        exp_dist = torch.exp(-self.t * dist_sq[mask])

        return torch.log(exp_dist.mean() + 1e-12)

    def forward(
        self,
        u_embeds: torch.Tensor,
        i_embeds: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DirectAU combined loss.

        Args:
            u_embeds: Final user embeddings in mini-batch (batch_size, dim)
            i_embeds: Final positive item embeddings in mini-batch (batch_size, dim)
            u_emb0: Initial user embedding table weights for regularization
            pos_emb0: Initial item embedding table weights for regularization

        Returns:
            total_loss: align + gamma * uniform + L2 reg
            align_loss: alignment loss
            uniform_loss: combined user and item uniformity loss
        """
        u_norm = F.normalize(u_embeds, dim=-1)
        i_norm = F.normalize(i_embeds, dim=-1)

        align_loss = self.compute_alignment_loss(u_norm, i_norm)
        u_unif = self.compute_uniformity_loss(u_norm)
        i_unif = self.compute_uniformity_loss(i_norm)
        uniform_loss = (u_unif + i_unif) / 2.0

        # L2 Regularization
        reg_loss = (u_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2)) / (
            2.0 * u_embeds.size(0)
        )

        total_loss = align_loss + self.gamma * uniform_loss + self.weight_decay * reg_loss
        return total_loss, align_loss, uniform_loss
