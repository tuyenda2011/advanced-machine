from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


def compute_topk_metrics(
    pos_items_list: List[List[int]],
    topk_predictions: torch.Tensor,
    k_list: List[int] = [10, 20],
) -> Dict[str, float]:
    """Compute standard ranking accuracy metrics (Recall@K, NDCG@K, MRR@K).

    Args:
        pos_items_list: Ground truth item ID lists for each user in evaluation batch
        topk_predictions: Tensor of shape (num_eval_users, max_k) containing recommended item IDs
        k_list: List of K values (e.g., [10, 20])

    Returns:
        Dict mapping metric names to mean values.
    """
    max_k = max(k_list)
    topk_predictions = topk_predictions[:, :max_k].cpu().numpy()
    num_users = len(pos_items_list)

    results = {}
    for k in k_list:
        recalls = []
        ndcgs = []
        mrrs = []

        discount = 1.0 / np.log2(np.arange(2, k + 2))

        for u in range(num_users):
            target_set = set(pos_items_list[u])
            if not target_set:
                continue

            preds_k = topk_predictions[u, :k]
            hits = np.array([1 if item in target_set else 0 for item in preds_k], dtype=np.float32)

            # Recall@K (standard definition: hits / |targets|)
            num_hits = hits.sum()
            recall = num_hits / len(target_set)
            recalls.append(recall)

            # NDCG@K
            dcg = np.sum(hits * discount)
            idcg = np.sum(discount[: min(len(target_set), k)])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)

            # MRR@K
            first_hit = np.where(hits == 1)[0]
            mrr = 1.0 / (first_hit[0] + 1) if len(first_hit) > 0 else 0.0
            mrrs.append(mrr)

        results[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        results[f"MRR@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0

    return results


def compute_intra_list_diversity(
    topk_predictions: torch.Tensor,
    item_embeddings: torch.Tensor,
    k: int = 10,
    sample_users: int = 1000,
) -> float:
    """Compute Intra-List Diversity (ILD) based on cosine distance between recommended items.

    ILD@K = (1 / |U|) * sum_{u} [ (2 / (K*(K-1))) * sum_{i < j} (1 - cos(e_i, e_j)) ]

    Args:
        topk_predictions: Tensor (num_eval_users, max_k) of recommended item IDs
        item_embeddings: Tensor (num_items, dim) of learned item representations
        k: Top-K cutoff
        sample_users: Max users to sample for fast computation

    Returns:
        float: Mean intra-list diversity score in [0, 2] (higher is more diverse)
    """
    preds_k = topk_predictions[:, :k]
    num_users = preds_k.size(0)

    if num_users > sample_users:
        perm = torch.randperm(num_users)[:sample_users]
        preds_k = preds_k[perm]
        num_users = sample_users

    # Normalize item embeddings for fast cosine similarity
    norm_item_embeds = F.normalize(item_embeddings.to(preds_k.device), dim=-1)

    # Gather embeddings: shape (num_users, k, dim)
    rec_embeds = norm_item_embeds[preds_k]

    # Compute batch pairwise cosine similarity: shape (num_users, k, k)
    sim_matrix = torch.bmm(rec_embeds, rec_embeds.transpose(1, 2))

    # Mask upper triangle (i < j)
    k_pairs = k * (k - 1) / 2.0
    if k_pairs <= 0:
        return 0.0

    triu_indices = torch.triu_indices(k, k, offset=1)
    pairwise_sims = sim_matrix[:, triu_indices[0], triu_indices[1]] # (num_users, k_pairs)
    pairwise_dists = 1.0 - pairwise_sims # Cosine distance

    user_ild = pairwise_dists.mean(dim=1)
    return float(user_ild.mean().item())


def compute_novelty(
    topk_predictions: torch.Tensor,
    item_popularity_dict: Dict[int, int],
    num_total_users: int,
    k: int = 10,
) -> float:
    """Compute Recommendation Novelty / Serendipity based on Self-Information.

    Novelty@K = (1 / (|U| * K)) * sum_{u} sum_{i in R_u} -log2( P(i) )
    where P(i) = (count(i) + 1) / num_total_users

    Args:
        topk_predictions: Tensor (num_eval_users, max_k) of recommended item IDs
        item_popularity_dict: Dictionary mapping item ID to interaction count in training set
        num_total_users: Total number of users
        k: Top-K cutoff

    Returns:
        float: Mean novelty score in bits (higher indicates more novel / tail items)
    """
    preds_k = topk_predictions[:, :k].cpu().numpy()
    num_users = preds_k.shape[0]
    if num_users == 0:
        return 0.0

    user_novelties = []
    for u in range(num_users):
        u_preds = preds_k[u]
        self_info = [
            -np.log2((item_popularity_dict.get(int(item), 0) + 1) / float(num_total_users))
            for item in u_preds
        ]
        user_novelties.append(np.mean(self_info))

    return float(np.mean(user_novelties))


def compute_coverage_and_gini(
    topk_predictions: torch.Tensor,
    num_total_items: int,
    k: int = 10,
) -> Tuple[float, float]:
    """Compute Catalog Coverage and Gini Index of recommendation distribution.

    - Coverage@K = |Unique items recommended| / num_total_items
    - Gini@K = Concentration inequality of recommendations (0 = perfect equality, 1 = extreme bias)

    Returns:
        Tuple[coverage, gini_index]
    """
    preds_k = topk_predictions[:, :k].cpu().numpy().flatten()
    unique_items, counts = np.unique(preds_k, return_counts=True)

    # Coverage
    coverage = float(len(unique_items) / float(num_total_items)) if num_total_items > 0 else 0.0

    # Gini Coefficient across entire catalog
    full_counts = np.zeros(num_total_items, dtype=np.float64)
    full_counts[unique_items] = counts
    sorted_counts = np.sort(full_counts)

    n = num_total_items
    index = np.arange(1, n + 1)
    sum_counts = np.sum(sorted_counts)
    if sum_counts == 0:
        gini = 0.0
    else:
        gini = float((2.0 * np.sum(index * sorted_counts) - (n + 1) * sum_counts) / (n * sum_counts))

    return coverage, gini
