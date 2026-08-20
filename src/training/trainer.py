import logging
import os
import time
from typing import Any, Dict, Set, Tuple
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
from src.models.base import BaseRecommender
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.training.early_stopping import EarlyStopping, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)


def sample_negative_items(
    users: np.ndarray,
    num_items: int,
    train_history: Dict[int, Set[int]],
) -> np.ndarray:
    """Uniform random negative sampling for an array of user indices."""
    neg_items = np.random.randint(0, num_items, size=len(users))
    for idx, u in enumerate(users):
        user_history = train_history.get(u, set())
        while neg_items[idx] in user_history:
            neg_items[idx] = np.random.randint(0, num_items)
    return neg_items


class Trainer:
    """Generic PyTorch trainer for LightGCN, SGL, and SimGCL recommendation models with visual progress bars and robust checkpointing."""

    def __init__(
        self,
        model: BaseRecommender,
        train_df: pd.DataFrame,
        val_evaluator: Evaluator,
        test_evaluator: Evaluator,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_df = train_df
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        self.config = config
        self.device = device

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

    def train(self, checkpoint_path: str, resume: bool = False) -> Dict[str, Any]:
        """Execute full model training loop with tqdm visual progress bars, early stopping, and scientific metrics."""
        logger.info(
            f"Starting training for {self.model_name.upper()} on device {self.device}..."
        )

        latest_checkpoint_path = checkpoint_path.replace(".pt", "_latest.pt")
        start_epoch = 1

        if resume and os.path.exists(latest_checkpoint_path):
            loaded_epoch, best_sc, ckpt = load_checkpoint(
                latest_checkpoint_path, self.model, self.optimizer, device=self.device
            )
            start_epoch = loaded_epoch + 1
            self.early_stopping.best_score = best_sc
            self.early_stopping.best_epoch = loaded_epoch
            logger.info(f"Resuming training from epoch {start_epoch}/{self.epochs}")

        user_array = self.train_df["u_idx"].values
        pos_item_array = self.train_df["i_idx"].values
        num_samples = len(user_array)

        start_train_time = time.perf_counter()
        epoch_times = []

        best_val_metrics = {}
        best_epoch = self.early_stopping.best_epoch

        history_dir = os.path.join("results", "history")
        os.makedirs(history_dir, exist_ok=True)
        history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
        history_csv_path = os.path.join(history_dir, history_csv_name)

        history_records = []
        if resume and os.path.exists(history_csv_path):
            history_records = pd.read_csv(history_csv_path).to_dict(orient="records")

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()

            # 1. Random negative sampling per epoch
            neg_item_array = sample_negative_items(
                user_array, self.num_items, self.train_history
            )

            # Shuffle mini-batches
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            # SGL Edge Dropout graph view generation
            if self.model_name == "sgl":
                drop_ratio = self.config["sgl"]["drop_ratio"]
                norm_adj1, norm_adj2 = create_edge_dropout_views(
                    self.train_df,
                    self.num_users,
                    self.num_items,
                    drop_ratio,
                    self.device,
                )

            total_loss_accum = 0.0
            bpr_loss_accum = 0.0
            cl_loss_accum = 0.0
            num_batches = 0

            batch_pbar = tqdm(
                range(0, num_samples, self.batch_size),
                desc=f"[{self.model_name.upper()}] Epoch {epoch:02d}/{self.epochs:02d}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            )

            for i in batch_pbar:
                batch_idx = indices[i : i + self.batch_size]
                u_batch = torch.tensor(
                    user_array[batch_idx], dtype=torch.long, device=self.device
                )
                pos_batch = torch.tensor(
                    pos_item_array[batch_idx], dtype=torch.long, device=self.device
                )
                neg_batch = torch.tensor(
                    neg_item_array[batch_idx], dtype=torch.long, device=self.device
                )

                self.optimizer.zero_grad()

                # Main Graph Propagation
                u_embeds, i_embeds = self.model(self.norm_adj)

                pos_scores = (u_embeds[u_batch] * i_embeds[pos_batch]).sum(dim=-1)
                neg_scores = (u_embeds[u_batch] * i_embeds[neg_batch]).sum(dim=-1)

                # Initial embeddings for regularization
                u_emb0 = self.model.user_embedding(u_batch)
                pos_emb0 = self.model.item_embedding(pos_batch)
                neg_emb0 = self.model.item_embedding(neg_batch)

                total_loss, bpr_loss = self.bpr_loss_fn(
                    pos_scores, neg_scores, u_emb0, pos_emb0, neg_emb0
                )

                cl_loss = torch.tensor(0.0, device=self.device)

                if self.model_name == "sgl":
                    ssl_weight = self.config["sgl"]["ssl_weight"]
                    u_v1, i_v1 = self.model.forward_view(norm_adj1)
                    u_v2, i_v2 = self.model.forward_view(norm_adj2)

                    cl_loss = self.cl_loss_fn(
                        u_v1[u_batch], u_v2[u_batch], i_v1[pos_batch], i_v2[pos_batch]
                    )
                    total_loss = total_loss + ssl_weight * cl_loss

                elif self.model_name == "simgcl":
                    cl_weight = self.config["simgcl"]["contrastive_weight"]
                    u_p1, i_p1 = self.model.forward_perturbed(self.norm_adj)
                    u_p2, i_p2 = self.model.forward_perturbed(self.norm_adj)

                    cl_loss = self.cl_loss_fn(
                        u_p1[u_batch], u_p2[u_batch], i_p1[pos_batch], i_p2[pos_batch]
                    )
                    total_loss = total_loss + cl_weight * cl_loss

                total_loss.backward()
                self.optimizer.step()

                total_loss_accum += total_loss.item()
                bpr_loss_accum += bpr_loss.item()
                cl_loss_accum += cl_loss.item()
                num_batches += 1

                batch_pbar.set_postfix({
                    "Loss": f"{total_loss.item():.4f}",
                    "BPR": f"{bpr_loss.item():.4f}",
                })

            batch_pbar.close()

            if self.device.type == "cuda":
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
            tqdm.write(
                f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:4.1f}s] | Loss: {avg_loss:.4f} | Val Recall@10: {val_rec10:.4f} | Val NDCG@10: {val_ndcg10:.4f}{best_tag}"
            )

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
            history_dir = os.path.join("results", "history")
            os.makedirs(history_dir, exist_ok=True)
            history_csv_name = os.path.basename(checkpoint_path).replace(".pt", "_history.csv")
            history_csv_path = os.path.join(history_dir, history_csv_name)
            pd.DataFrame(history_records).to_csv(history_csv_path, index=False)

            if self.early_stopping.early_stop:
                logger.info(f"Stopping early at epoch {epoch}")
                break

        epoch_pbar.close()

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
