from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseRecommender


class DirectAU(BaseRecommender):
    """DirectAU: Direct Alignment and Uniformity for Collaborative Filtering (KDD '22).

    Optimizes representation alignment of positive pairs and uniformity across hypersphere directly.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        gamma: float = 1.0,
        t: float = 2.0,
    ):
        super().__init__(num_users, num_items, embedding_dim, num_layers)
        self.gamma = gamma
        self.t = t

    def forward(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform LightGCN layer aggregation over normalized bipartite graph adjacency matrix."""
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

    def get_user_rating_scores(
        self, u_indices: torch.Tensor, all_users: torch.Tensor, all_items: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity prediction scores for given users against all items on unit sphere."""
        u_embeds = F.normalize(all_users[u_indices], dim=-1)
        i_embeds = F.normalize(all_items, dim=-1)
        scores = torch.matmul(u_embeds, i_embeds.T)
        return scores
