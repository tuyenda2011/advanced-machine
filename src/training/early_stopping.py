import logging
import os
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early Stopping handler to track validation metrics and save best model checkpoint."""

    def __init__(self, patience: int = 20, monitor: str = "NDCG@10", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def __call__(
        self, score: float, epoch: int, model: torch.nn.Module, checkpoint_path: str
    ) -> bool:
        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

            # Save checkpoint
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_score": score,
                },
                checkpoint_path,
            )
            logger.info(
                f"Epoch {epoch}: Validation {self.monitor} improved to {score:.4f}. Saved checkpoint to {checkpoint_path}"
            )
        else:
            self.counter += 1
            logger.info(
                f"Epoch {epoch}: Validation {self.monitor} did not improve ({score:.4f} vs best {self.best_score:.4f}). Counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch}!")

        return improved
