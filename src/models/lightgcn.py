from typing import Tuple
import torch
import torch.nn as nn
from src.models.base import BaseRecommender


class LightGCN(BaseRecommender):
    """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR '20)."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform LightGCN layer aggregation over normalized bipartite graph adjacency matrix.

        Returns:
            final_user_embeddings: (num_users, dim)
            final_item_embeddings: (num_items, dim)
        """
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embeddings = [ego_embeddings]

        for layer in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        # Layer averaging aggregation: E = (E0 + E1 + ... + EL) / (L + 1)
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        user_final, item_final = torch.split(
            final_embeddings, [self.num_users, self.num_items], dim=0
        )
        return user_final, item_final
