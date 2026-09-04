import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Force UTF-8 encoding for Windows Command Prompt/PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm


from src.data.graph import create_edge_dropout_views, get_norm_adj_tensor
from src.evaluation.evaluator import Evaluator
from src.evaluation.representation import (
    compute_alignment_and_uniformity,
    compute_svd_spectrum,
)
from src.evaluation.subgroup import evaluate_degree_subgroups
from src.losses.bpr import BPRLoss
from src.losses.contrastive import InfoNCELoss
from src.losses.directau import DirectAULoss
from src.losses.hard_bpr import HardNegativeBPRLoss
from src.models.adaptive_gcl import AdaptiveGCL
from src.models.base import BaseRecommender
from src.models.directau import DirectAU
from src.models.lightgcn import LightGCN
from src.models.xsimgcl import XSimGCL
from src.training.early_stopping import EarlyStopping, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

# Prefer the current AMP API while retaining compatibility with older PyTorch.
try:
    from torch.amp import GradScaler
    MODERN_AMP_API = True
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
        MODERN_AMP_API = False
        AMP_AVAILABLE = True
    except ImportError:
        MODERN_AMP_API = False
        AMP_AVAILABLE = False


# =============================================================================
# Constants for training configuration
# =============================================================================
MASK_VALUE: float = -1e9  # Mask value for filtering seen items
GRADIENT_CLIP_VALUE: float = 1.0  # Gradient clipping max norm
DEFAULT_TERMINAL_WIDTH: int = 80  # Default terminal width for progress display
SYNC_CUDA: bool = False  # Whether to synchronize CUDA after each epoch (for accurate timing)


def sample_negative_items(
    users: np.ndarray,
    num_items: int,
    train_history: Dict[int, Set[int]],
    max_attempts: int = 100,
) -> np.ndarray:
    """Fast uniform random negative sampling with vectorized collision rejection.

    Optimized implementation that batches operations and minimizes Python loops.

    Args:
        users: Array of user indices
        num_items: Total number of items
        train_history: Dictionary mapping user to set of positive items
        max_attempts: Maximum retry attempts per negative sample to prevent infinite loops

    Returns:
        Array of negative item indices
    """
    n = len(users)
    neg_items = np.random.randint(0, num_items, size=n)

    # Pre-compute history sets for faster lookup
    user_histories = [train_history.get(u, set()) for u in users]

    # Vectorized collision detection using numpy
    collisions = np.array([
        neg_items[i] in user_histories[i]
        for i in range(n)
    ], dtype=bool)

    # Retry collisions with limit
    attempts = np.zeros(n, dtype=np.int32)
    while collisions.any() and (attempts < max_attempts).any():
        retry_mask = collisions & (attempts < max_attempts)
        neg_items[retry_mask] = np.random.randint(0, num_items, size=retry_mask.sum())

        # Update collision status
        for i in np.where(retry_mask)[0]:
            attempts[i] += 1
            collisions[i] = neg_items[i] in user_histories[i]

    return neg_items


def sample_hard_negative_items(
    users: np.ndarray,
    user_disliked_items: Dict[int, List[int]],
    train_history: Dict[int, Set[int]],
    num_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one valid explicit dislike per user when available."""
    if len(users) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)

    selected_by_user = np.full(int(users.max()) + 1, -1, dtype=np.int64)
    for user in np.unique(users):
        positives = train_history.get(int(user), set())
        candidates = [
            item
            for item in user_disliked_items.get(int(user), [])
            if 0 <= item < num_items and item not in positives
        ]
        if candidates:
            selected_by_user[int(user)] = candidates[np.random.randint(len(candidates))]

    hard_items = selected_by_user[users.astype(np.int64)]
    available = hard_items >= 0
    return hard_items, available


class Trainer:
    """Trainer for the four recommendation models used in this project."""

    def __init__(
        self,
        model: BaseRecommender,
        train_df: pd.DataFrame,
        val_evaluator: Evaluator,
        test_evaluator: Evaluator,
        config: Dict[str, Any],
        device: torch.device,
        user_disliked_items: Optional[Dict[int, List[int]]] = None,
    ):
        self.model = model.to(device)
        self.train_df = train_df
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.config = config
        self.device = device
        self.user_disliked_items = user_disliked_items or {}

        self.model_name = config["model_name"]
        self.num_users = model.num_users
        self.num_items = model.num_items

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.batch_size = train_cfg["batch_size"]
        self.lr = train_cfg["learning_rate"]
        self.weight_decay = train_cfg["weight_decay"]

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.bpr_loss_fn = BPRLoss(weight_decay=self.weight_decay)
        self.cl_loss_fn = InfoNCELoss(
            temperature=config.get(self.model_name, {}).get("temperature", 0.2)
        )
        if self.model_name == "directau":
            d_cfg = config.get("directau", {})
            self.directau_loss_fn = DirectAULoss(
                gamma=d_cfg.get("gamma", 1.0),
                t=d_cfg.get("t", 2.0),
                weight_decay=self.weight_decay,
            )
        elif self.model_name == "adaptive_gcl":
            adaptive_cfg = config.get("adaptive_gcl", {})
            self.hard_bpr_loss_fn = HardNegativeBPRLoss(
                alpha=adaptive_cfg.get("hard_neg_alpha", 0.2),
                margin=adaptive_cfg.get("hard_neg_margin", 0.5),
            )

        # Precompute train history mapping for negative sampling
        self.train_history: Dict[int, Set[int]] = (
            train_df.groupby("u_idx")["i_idx"].apply(set).to_dict()
        )

        # Precompute base graph normalized adjacency tensor
        self.norm_adj = get_norm_adj_tensor(
            train_df, self.num_users, self.num_items, device
        )

        # Training positive edge index for Alignment/Uniformity computation
        self.train_edge_index = torch.tensor(
            np.stack([train_df["u_idx"].values, train_df["i_idx"].values], axis=0),
            dtype=torch.long,
            device=device,
        )

        # Early stopping handler
        self.early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping_patience"],
            monitor="NDCG@10",
            mode="max",
        )

        # Mixed precision training (AMP)
        self.use_amp = AMP_AVAILABLE and device.type == "cuda"
        if self.use_amp:
            self.scaler = GradScaler("cuda") if MODERN_AMP_API else GradScaler()
            logger.info("AMP gradient scaling enabled for CUDA device")
        else:
            self.scaler = None

    def _backward_and_step(self, total_loss: torch.Tensor) -> None:
        """Backpropagate, clip gradients, and update model parameters."""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            # Gradients must be unscaled before their norm is clipped.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=GRADIENT_CLIP_VALUE
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=GRADIENT_CLIP_VALUE
            )
            self.optimizer.step()

    def train(self, checkpoint_path: str, resume: bool = False) -> Dict[str, Any]:
        """Execute full model training loop with tqdm visual progress bars, early stopping, and scientific metrics."""
        logger.info(
            f"Starting training for {self.model_name.upper()} on device {self.device}..."
        )

        latest_checkpoint_path = checkpoint_path.replace(".pt", "_latest.pt")
        start_epoch = 1

        if resume and os.path.exists(latest_checkpoint_path):
            try:
                loaded_epoch, best_sc, ckpt = load_checkpoint(
                    latest_checkpoint_path,
                    self.model,
                    self.optimizer,
                    device=self.device,
                    expected_fingerprint=self.config.get("experiment_fingerprint"),
                )
                start_epoch = loaded_epoch + 1
                self.early_stopping.best_score = best_sc
                self.early_stopping.best_epoch = ckpt.get("best_epoch", loaded_epoch)
                print(
                    f"🔄 [RESUME CHECKPOINT] Đã khôi phục trọng số! Tiếp tục train từ Epoch {start_epoch:02d}/{self.epochs:02d} (Val NDCG@10 đỉnh hiện tại: {best_sc:.4f})\n",
                    flush=True,
                )
            except Exception as ex:
                logger.warning(f"Error loading checkpoint for resume: {ex}. Starting from epoch 1.")

        user_array = self.train_df["u_idx"].values
        pos_item_array = self.train_df["i_idx"].values
        num_samples = len(user_array)

        start_train_time = time.perf_counter()
        epoch_times = []

        best_val_metrics = {}
        best_epoch = self.early_stopping.best_epoch

        history_dir = os.path.join("results", "history", self.model_name)
        os.makedirs(history_dir, exist_ok=True)
        history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
        history_csv_path = os.path.join(history_dir, history_csv_name)

        history_records = []
        if resume and os.path.exists(history_csv_path):
            history_records = pd.read_csv(history_csv_path).to_dict(orient="records")

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()

            # 1. Random negative sampling per epoch (only needed if model uses negative sampling)
            if self.model_name != "directau":
                neg_item_array = sample_negative_items(
                    user_array, self.num_items, self.train_history
                )
            else:
                neg_item_array = None

            if self.model_name == "adaptive_gcl" and self.user_disliked_items:
                hard_item_array, hard_item_available = sample_hard_negative_items(
                    user_array,
                    self.user_disliked_items,
                    self.train_history,
                    self.num_items,
                )
            else:
                hard_item_array = None
                hard_item_available = None

            # Shuffle mini-batches
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            total_loss_accum = 0.0
            bpr_loss_accum = 0.0
            cl_loss_accum = 0.0
            num_batches = 0

            for i in range(0, num_samples, self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                u_batch = torch.tensor(
                    user_array[batch_idx], dtype=torch.long, device=self.device
                )
                pos_batch = torch.tensor(
                    pos_item_array[batch_idx], dtype=torch.long, device=self.device
                )

                self.optimizer.zero_grad()

                if self.model_name == "directau":
                    u_embeds, i_embeds = self.model(self.norm_adj)
                    u_emb0 = self.model.user_embedding(u_batch)
                    pos_emb0 = self.model.item_embedding(pos_batch)

                    total_loss, align_loss, unif_loss = self.directau_loss_fn(
                        u_embeds[u_batch], i_embeds[pos_batch], u_emb0, pos_emb0
                    )
                    bpr_loss_accum += align_loss.item()
                    cl_loss_accum += unif_loss.item()
                else:
                    neg_batch = torch.tensor(
                        neg_item_array[batch_idx], dtype=torch.long, device=self.device
                    )

                    if self.model_name == "xsimgcl":
                        u_embeds, i_embeds, cl_u_embeds, cl_i_embeds = self.model(
                            self.norm_adj, perturbed=True
                        )
                    else:
                        u_embeds, i_embeds = self.model(self.norm_adj)

                    pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
                    neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

                    if self.model_name == "xsimgcl":
                        u_emb0 = u_embeds[u_batch]
                        pos_emb0 = i_embeds[pos_batch]
                        # Match the official SELFRec implementation, which
                        # regularizes the propagated user and positive item only.
                        neg_emb0 = None
                    else:
                        u_emb0 = self.model.user_embedding(u_batch)
                        pos_emb0 = self.model.item_embedding(pos_batch)
                        neg_emb0 = self.model.item_embedding(neg_batch)

                    total_loss, bpr_loss = self.bpr_loss_fn(
                        pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
                    )

                    cl_loss = torch.tensor(0.0, device=self.device)

                    if self.model_name == "xsimgcl":
                        cl_weight = self.config["xsimgcl"]["contrastive_weight"]
                        unique_users = torch.unique(u_batch)
                        unique_items = torch.unique(pos_batch)
                        cl_loss = (
                            self.cl_loss_fn.compute_view_loss(
                                u_embeds[unique_users], cl_u_embeds[unique_users]
                            )
                            + self.cl_loss_fn.compute_view_loss(
                                i_embeds[unique_items], cl_i_embeds[unique_items]
                            )
                        )
                        total_loss = total_loss + cl_weight * cl_loss

                    elif self.model_name == "adaptive_gcl":
                        cl_loss = self.model.compute_semantic_ssl_loss(pos_batch, i_embeds)
                        total_loss = total_loss + cl_loss

                        if hard_item_array is not None and hard_item_available is not None:
                            batch_hard_mask = hard_item_available[batch_idx]
                            if batch_hard_mask.any():
                                hard_mask = torch.tensor(
                                    batch_hard_mask, dtype=torch.bool, device=self.device
                                )
                                hard_batch = torch.tensor(
                                    hard_item_array[batch_idx][batch_hard_mask],
                                    dtype=torch.long,
                                    device=self.device,
                                )
                                total_loss = total_loss + self.hard_bpr_loss_fn.compute_hard_penalty(
                                    u_embeds[u_batch[hard_mask]],
                                    i_embeds[pos_batch[hard_mask]],
                                    i_embeds[hard_batch],
                                )

                        if self.model.dirichlet_reg > 0:
                            all_final = torch.cat([u_embeds, i_embeds], dim=0)
                            total_loss = total_loss + self.model.compute_dirichlet_regularization(
                                self.norm_adj, all_final
                            )

                    bpr_loss_accum += bpr_loss.item()
                    cl_loss_accum += cl_loss.item()


                self._backward_and_step(total_loss)

                total_loss_accum += total_loss.item()
                num_batches += 1

                # Live in-place batch progress
                curr_sample = min(i + self.batch_size, num_samples)
                pct = int(curr_sample / num_samples * 100)
                total_batches = (num_samples + self.batch_size - 1) // self.batch_size
                batch_status = f"[{self.model_name.upper()}] Epoch {epoch:02d}/{self.epochs:02d} | Batch {num_batches:>3d}/{total_batches} ({pct:>3d}%) | Loss: {total_loss.item():.4f}"
                sys.stdout.write(f"\r{batch_status:<70}")
                sys.stdout.flush()

            if SYNC_CUDA and self.device.type == "cuda":
                torch.cuda.synchronize()

            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)

            # Evaluation on Validation set every epoch
            self.model.eval()
            with torch.no_grad():
                val_u_embeds, val_i_embeds = self.model(self.norm_adj)
                val_metrics, _ = self.val_evaluator.evaluate(
                    val_u_embeds, val_i_embeds, self.device, include_beyond_accuracy=False
                )

            val_ndcg10 = val_metrics["NDCG@10"]
            val_rec10 = val_metrics["Recall@10"]
            avg_loss = total_loss_accum / max(1, num_batches)

            is_improved = self.early_stopping(
                val_ndcg10,
                epoch,
                self.model,
                checkpoint_path,
                optimizer=self.optimizer,
                val_metrics=val_metrics,
                config=self.config,
            )
            if is_improved:
                best_val_metrics = val_metrics
                best_epoch = epoch

            best_tag = " 🌟 [BEST]" if is_improved else ""

            # Overwrite line atomically without excessive spaces to prevent terminal wrapping
            epoch_summary = f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:4.1f}s] | Loss: {avg_loss:.4f} | Val Recall@10: {val_rec10:.4f} | Val NDCG@10: {val_ndcg10:.4f}{best_tag}"
            sys.stdout.write(f"\r{epoch_summary:<85}\n")
            sys.stdout.flush()

            # Save latest checkpoint at end of each epoch for resuming
            save_checkpoint(
                latest_checkpoint_path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_score=self.early_stopping.best_score,
                val_metrics=val_metrics,
                config=self.config,
            )

            # Record epoch training history
            history_records.append({
                "epoch": epoch,
                "train_loss": round(total_loss_accum / max(1, num_batches), 4),
                "bpr_loss": round(bpr_loss_accum / max(1, num_batches), 4),
                "cl_loss": round(cl_loss_accum / max(1, num_batches), 4),
                "val_ndcg_10": round(val_ndcg10, 4),
                "val_recall_10": round(val_metrics.get("Recall@10", 0.0), 4),
                "val_mrr_10": round(val_metrics.get("MRR@10", 0.0), 4),
                "epoch_time_sec": round(epoch_time, 2),
                "is_best": bool(is_improved),
            })

            # Save epoch history CSV
            history_dir = os.path.join("results", "history", self.model_name)
            os.makedirs(history_dir, exist_ok=True)
            history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
            history_csv_path = os.path.join(history_dir, history_csv_name)
            pd.DataFrame(history_records).to_csv(history_csv_path, index=False)


            if self.early_stopping.early_stop:
                logger.info(f"Stopping early at epoch {epoch}")
                break

        total_train_time = time.perf_counter() - start_train_time
        avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else 0.0

        # Load best checkpoint for final evaluation
        if os.path.exists(checkpoint_path):
            load_checkpoint(checkpoint_path, self.model, device=self.device)
            logger.info(f"Loaded best checkpoint for final evaluation: {checkpoint_path}")

        self.model.eval()
        with torch.no_grad():
            final_u_embeds, final_i_embeds = self.model(self.norm_adj)

            # 1. Full ranking evaluation including beyond-accuracy metrics with progress bar
            test_metrics, avg_user_latency_ms = self.test_evaluator.evaluate(
                final_u_embeds, final_i_embeds, self.device, include_beyond_accuracy=True, show_progress=True
            )

            # 2. Representation Geometry (Alignment & Uniformity)
            rep_metrics = compute_alignment_and_uniformity(
                final_u_embeds, final_i_embeds, self.train_edge_index
            )

            # 3. SVD Spectrum & Dimensional Collapse
            svd_user = compute_svd_spectrum(final_u_embeds)
            svd_item = compute_svd_spectrum(final_i_embeds)

            # 4. Degree-Stratified Subgroup Analysis (Head / Torso / Tail)
            subgroups = evaluate_degree_subgroups(
                self.test_evaluator,
                final_u_embeds,
                final_i_embeds,
                self.device,
                self.train_df,
            )

        throughput_users_per_sec = 1000.0 / avg_user_latency_ms if avg_user_latency_ms > 0 else 0.0

        summary_results = {
            "model_name": self.model_name,
            "best_epoch": best_epoch,
            "total_epochs": len(epoch_times),
            "total_train_time": total_train_time,
            "avg_epoch_time": avg_epoch_time,
            "inference_latency_ms_per_user": avg_user_latency_ms,
            "throughput_users_per_sec": throughput_users_per_sec,
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "representation_metrics": rep_metrics,
            "svd_metrics": {
                "user_effective_rank": svd_user["effective_rank"],
                "item_effective_rank": svd_item["effective_rank"],
                "user_singular_values": svd_user["singular_values"][:15],
                "item_singular_values": svd_item["singular_values"][:15],
            },
            "subgroup_metrics": subgroups,
        }

        logger.info(f"Training completed for {self.model_name.upper()}.")
        logger.info(
            f"Final Test Results -> Recall@10: {test_metrics.get('Recall@10', 0):.4f} | NDCG@10: {test_metrics.get('NDCG@10', 0):.4f} | Diversity@10: {test_metrics.get('Diversity@10', 0):.4f} | Novelty@10: {test_metrics.get('Novelty@10', 0):.4f}"
        )
        logger.info(
            f"Representation -> Alignment: {rep_metrics['alignment']:.4f} | Mean Uniformity: {rep_metrics['mean_uniformity']:.4f} | User Eff Rank: {svd_user['effective_rank']:.2f}"
        )

        return summary_results
