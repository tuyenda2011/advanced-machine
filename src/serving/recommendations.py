"""Exact recommendations with model scoring and explicit candidate filtering."""

from collections.abc import Callable

import torch

from src.models.base import BaseRecommender


@torch.no_grad()
def recommend_exact(
    model: BaseRecommender,
    user_index: int,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    k: int = 10,
    excluded_items: set[int] | None = None,
    metadata: dict[int, dict] | None = None,
    filter_fn: Callable[[dict], bool] | None = None,
) -> list[tuple[int, float]]:
    """Return up to k eligible items; an empty filter match returns no items."""
    if k < 0:
        raise ValueError("k must be nonnegative")
    excluded = excluded_items or set()
    metadata = metadata or {}
    eligible = [
        item for item in range(item_embeddings.shape[0])
        if item not in excluded
        and (filter_fn is None or filter_fn(metadata.get(item, {})))
    ]
    if k == 0 or not eligible:
        return []
    device = user_embeddings.device
    user = torch.tensor([user_index], dtype=torch.long, device=device)
    scores = model.get_user_rating_scores(user, user_embeddings, item_embeddings)[0]
    candidates = torch.tensor(eligible, dtype=torch.long, device=device)
    values, positions = scores[candidates].topk(min(k, len(eligible)))
    return list(zip(candidates[positions].tolist(), values.tolist()))
