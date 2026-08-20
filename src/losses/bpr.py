import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) Loss with L2 Regularization on initial embeddings."""

    def __init__(self, weight_decay: float = 1e-4):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
        u_emb0: torch.Tensor,
        pos_emb0: torch.Tensor,
        neg_emb0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute BPR Loss and L2 Regularization Loss.

        Args:
            pos_scores: Inner product scores for positive user-item pairs (batch_size,)
            neg_scores: Inner product scores for negative user-item pairs (batch_size,)
            u_emb0: Initial user embeddings (batch_size, dim)
            pos_emb0: Initial positive item embeddings (batch_size, dim)
            neg_emb0: Initial negative item embeddings (batch_size, dim)

        Returns:
            total_loss: BPR loss + L2 regularization loss
            bpr_loss: Pure BPR loss for logging
        """
        # BPR Loss: -log(sigmoid(pos_score - neg_score)) = softplus(-(pos_score - neg_score))
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # L2 Regularization on initial embeddings (e0)
        reg_loss = (
            u_emb0.norm(2).pow(2)
            + pos_emb0.norm(2).pow(2)
            + neg_emb0.norm(2).pow(2)
        ) / (2.0 * pos_scores.shape[0])

        total_loss = bpr_loss + self.weight_decay * reg_loss
        return total_loss, bpr_loss
