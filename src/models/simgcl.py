from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseRecommender


class SimGCL(BaseRecommender):
    """SimGCL: Simple Graph Contrastive Learning for Recommendation (SIGIR '22).

    Constructs contrastive views using direct uniform embedding perturbation noise instead of graph augmentation.
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
        """Standard LightGCN forward propagation without noise perturbation."""
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

    def forward_perturbed(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation with layer-wise random uniform noise perturbation."""
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embeddings = [ego_embeddings]

        for layer in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)

            # Generate uniform noise in [0, 1) and normalize to unit norm
            random_noise = torch.rand_like(ego_embeddings)
            random_noise = F.normalize(random_noise, dim=1)

            # Add scaled noise: e_k = e_k + epsilon * noise
            ego_embeddings = ego_embeddings + self.epsilon * random_noise
            all_embeddings.append(ego_embeddings)

        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_final, item_final
