from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseRecommender


class XSimGCL(BaseRecommender):
    """XSimGCL: Extreme Simple Graph Contrastive Learning for Recommendation (TKDE '23).

    Eliminates redundant layer-wise perturbations by applying uniform noise perturbation
    directly to the aggregated final representation, drastically reducing training computation FLOPS.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        contrastive_weight: float = 0.1,
        temperature: float = 0.2,
        epsilon: float = 0.1,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard LightGCN forward propagation layer aggregation."""
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embeddings = [ego_embeddings]

        for layer in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_final, item_final

    def forward_perturbed(
        self, user_final: torch.Tensor, item_final: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply normalized random uniform noise perturbation directly to final representations."""
        # Perturb users
        u_noise = torch.rand_like(user_final)
        u_noise = F.normalize(u_noise, dim=-1)
        user_perturbed = user_final + self.epsilon * u_noise

        # Perturb items
        i_noise = torch.rand_like(item_final)
        i_noise = F.normalize(i_noise, dim=-1)
        item_perturbed = item_final + self.epsilon * i_noise

        return user_perturbed, item_perturbed
