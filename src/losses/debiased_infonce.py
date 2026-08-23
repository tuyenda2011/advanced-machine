import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DebiasedInfoNCELoss(nn.Module):
    """Debiased Contrastive InfoNCE Loss (NeurIPS '20 / RecSys SSL).

    Addresses False-Negative sampling bias where unobserved items in the denominator
    contain true positive user interests. Uses positive prior tau_plus to downweight
    false negatives and optionally amplifies known explicit hard negatives (1-2 stars).
    """

    def __init__(
        self,
        temperature: float = 0.2,
        tau_plus: float = 0.1,
        hard_negative_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.hard_negative_weight = hard_negative_weight

    def compute_debiased_contrastive_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute debiased InfoNCE loss for a batch of query-positive pairs.

        Args:
            query: Query representations (B, dim)
            positive: Positive key representations (B, dim)
            negatives: Optional explicit negative pool (M, dim). If None, all in-batch keys are used.
            hard_negatives: Optional explicit hard negatives (B, dim) from 1-2 star feedback.

        Returns:
            Scalar loss tensor.
        """
        q = F.normalize(query, dim=-1)
        k_pos = F.normalize(positive, dim=-1)

        # Positive pair cosine similarities: (B,)
        pos_sim = torch.sum(q * k_pos, dim=-1) / self.temperature
        pos_exp = torch.exp(pos_sim)

        if negatives is None:
            # Use all in-batch items as negative candidate pool (B, B)
            sim_matrix = torch.matmul(q, k_pos.T) / self.temperature
            exp_sim = torch.exp(sim_matrix)

            # Sum over candidate negatives
            n_samples = q.shape[0]
            # Average exponential similarity over pool
            avg_neg = torch.mean(exp_sim, dim=-1)  # (B,)
        else:
            k_neg = F.normalize(negatives, dim=-1)
            sim_matrix = torch.matmul(q, k_neg.T) / self.temperature
            exp_sim = torch.exp(sim_matrix)
            n_samples = k_neg.shape[0]
            avg_neg = torch.mean(exp_sim, dim=-1)

        # Debiasing transformation: remove positive expectation from negative pool
        # g_tilde = max((avg_neg - tau_plus * pos_exp) / (1 - tau_plus), exp(-1/tau))
        lower_bound = torch.exp(torch.tensor(-1.0 / self.temperature, device=query.device))
        debiased_neg = (avg_neg - self.tau_plus * pos_exp) / (1.0 - self.tau_plus)
        debiased_neg = torch.clamp(debiased_neg, min=lower_bound.item())

        # Total negative denominator score
        total_neg_score = n_samples * debiased_neg

        # Incorporate explicit hard negative penalty if present
        if hard_negatives is not None:
            k_hard = F.normalize(hard_negatives, dim=-1)
            hard_sim = torch.sum(q * k_hard, dim=-1) / self.temperature
            hard_exp = torch.exp(hard_sim)
            total_neg_score = total_neg_score + self.hard_negative_weight * hard_exp

        # Loss: -log (pos / (pos + debiased_neg))
        loss = -torch.log(pos_exp / (pos_exp + total_neg_score + 1e-8))
        return torch.mean(loss)

    def forward(
        self,
        u_view1: torch.Tensor,
        u_view2: torch.Tensor,
        i_view1: torch.Tensor,
        i_view2: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dual user and item debiased contrastive loss."""
        user_loss = self.compute_debiased_contrastive_loss(u_view1, u_view2)
        item_loss = self.compute_debiased_contrastive_loss(
            i_view1, i_view2, hard_negatives=hard_negatives
        )
        return user_loss + item_loss
