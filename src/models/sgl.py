from typing import Tuple
import torch
import torch.nn as nn
from src.models.base import BaseRecommender


class SGL(BaseRecommender):
    """SGL: Self-Supervised Graph Learning for Recommendation (SIGIR '21).

    Uses Edge Dropout graph augmentations to construct dual contrastive views G1 and G2.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        ssl_weight: float = 0.1,
        temperature: float = 0.2,
        drop_ratio: float = 0.1,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.ssl_weight = ssl_weight
        self.temperature = temperature
        self.drop_ratio = drop_ratio

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard LightGCN forward propagation on original graph."""
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

    def forward_view(self, norm_adj_view: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propagation on an augmented view graph."""
        return self.forward(norm_adj_view)
