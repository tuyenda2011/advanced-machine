import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import (
    compute_coverage_and_gini,
    compute_intra_list_diversity,
    compute_novelty,
    compute_topk_metrics,
)


class Evaluator:
    """Full-ranking evaluator for Top-K recommendation with training history masking and beyond-accuracy metrics."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        k_list: List[int] = [10, 20],
        batch_size: int = 1024,
        candidate_items: Optional[Set[int]] = None,
        popularity_df: Optional[pd.DataFrame] = None,
        diversity_features: Optional[torch.Tensor] = None,
        diversity_mask: Optional[torch.Tensor] = None,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.k_list = k_list
        self.batch_size = batch_size
        self.diversity_features = None
        self.diversity_mask = None
        if diversity_features is not None:
            if diversity_features.ndim != 2 or diversity_features.shape[0] != num_items:
                raise ValueError("Diversity features must have one row per item")
            if diversity_mask is None or diversity_mask.dtype != torch.bool or diversity_mask.shape != (num_items,):
                raise ValueError("Diversity requires a boolean usable-text mask")
            if not torch.isfinite(diversity_features).all():
                raise ValueError("Diversity features must be finite")
            if (diversity_features[diversity_mask].norm(dim=1) == 0).any():
                raise ValueError("Usable diversity features must be nonzero")
            self.diversity_features = diversity_features.detach().cpu().clone()
            self.diversity_mask = diversity_mask.detach().cpu().clone()

        # Build training history set per user for masking
        self.train_history: Dict[int, Set[int]] = (
            train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        )

        # Precompute item popularity for Novelty metric
        popularity_source = popularity_df if popularity_df is not None else train_df
        self.item_popularity: Dict[int, int] = (
            popularity_source["i_idx"].value_counts().to_dict()
        )

        # Build ground truth target list for eval set users
        eval_grouped = eval_df.groupby("u_idx")["i_idx"].apply(list).to_dict()
        self.eval_users = sorted(list(eval_grouped.keys()))
        self.ground_truth = [eval_grouped[u] for u in self.eval_users]
        if candidate_items is None:
            self.excluded_candidates: List[int] = []
        else:
            self.excluded_candidates = sorted(
                set(range(num_items)) - set(candidate_items)
            )

    # Constant for masking seen items
    MASK_VALUE: float = -1e9

    @torch.no_grad()
    def get_predictions(
        self,
        final_user_embeds: torch.Tensor,
        final_item_embeds: torch.Tensor,
        device: torch.device,
        show_progress: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Compute full top-K prediction tensor for all evaluation users with optional progress bar."""
        max_k = max(self.k_list)
        all_topk_preds = []

        start_time = time.perf_counter()
        num_eval_users = len(self.eval_users)

        batch_range = range(0, num_eval_users, self.batch_size)
        if show_progress:
            batch_range = tqdm(
                batch_range,
                desc="Top-K Inference",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )

        for i in batch_range:
            batch_u_idx = self.eval_users[i : i + self.batch_size]
            u_tensors = torch.tensor(batch_u_idx, dtype=torch.long, device=device)

            # Compute rating scores matrix (batch_users, num_items)
            scores = torch.matmul(final_user_embeds[u_tensors], final_item_embeds.T)

            if self.excluded_candidates:
                excluded_tensor = torch.tensor(
                    self.excluded_candidates, dtype=torch.long, device=device
                )
                scores[:, excluded_tensor] = self.MASK_VALUE

            # Mask each user's own history. A batch-wide union would hide valid
            # targets whenever another user happened to have seen the same item.
            seen_rows = []
            seen_items = []
            for row, u in enumerate(batch_u_idx):
                seen = self.train_history.get(u)
                if seen:
                    seen_rows.extend([row] * len(seen))
                    seen_items.extend(seen)

            if seen_rows:
                row_tensor = torch.tensor(seen_rows, dtype=torch.long, device=device)
                item_tensor = torch.tensor(seen_items, dtype=torch.long, device=device)
                scores[row_tensor, item_tensor] = self.MASK_VALUE

            # Retrieve top-K recommended item IDs
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            all_topk_preds.append(topk_indices.cpu())

        if device.type == "cuda":
            torch.cuda.synchronize()

        if show_progress:
            import sys
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()

        total_inference_time = time.perf_counter() - start_time
        avg_user_latency_ms = (total_inference_time / max(1, num_eval_users)) * 1000.0

        topk_preds_tensor = torch.cat(all_topk_preds, dim=0) if all_topk_preds else torch.empty(0, max_k)
        return topk_preds_tensor, avg_user_latency_ms


    @torch.no_grad()
    def evaluate(
        self,
        final_user_embeds: torch.Tensor,
        final_item_embeds: torch.Tensor,
        device: torch.device,
        include_beyond_accuracy: bool = False,
        show_progress: bool = False,
    ) -> Tuple[Dict[str, float], float]:
        """Perform full-ranking evaluation across all target users.

        Returns:
            metrics_dict: Dictionary of evaluated metrics
            avg_user_latency_ms: Average latency per user in milliseconds
        """
        topk_preds_tensor, avg_user_latency_ms = self.get_predictions(
            final_user_embeds, final_item_embeds, device, show_progress=show_progress
        )

        metrics = compute_topk_metrics(self.ground_truth, topk_preds_tensor, self.k_list)

        if include_beyond_accuracy and topk_preds_tensor.size(0) > 0:
            for k in self.k_list:
                # Shared content space only; never silently use model embeddings.
                metrics[f"Diversity@{k}"] = float("nan")
                if self.diversity_features is not None:
                    metrics[f"Diversity@{k}"] = compute_intra_list_diversity(
                        topk_preds_tensor, self.diversity_features, k=k,
                        item_mask=self.diversity_mask,
                    )
                    metrics[f"DiversityValidTextFraction@{k}"] = float(
                        self.diversity_mask[topk_preds_tensor[:, :k]].float().mean()
                    )

                # Novelty (Self-Information)
                novelty = compute_novelty(
                    topk_preds_tensor, self.item_popularity, self.num_users, k=k
                )
                metrics[f"Novelty@{k}"] = novelty

                # Catalog Coverage & Gini Coefficient
                cov, gini = compute_coverage_and_gini(
                    topk_preds_tensor, self.num_items, k=k
                )
                metrics[f"Coverage@{k}"] = cov
                metrics[f"Gini@{k}"] = gini

        return metrics, avg_user_latency_ms
