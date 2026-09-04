"""Centralized checkpoint path utilities for consistent checkpoint management across the project."""

import hashlib
import os
from typing import Optional


def get_experiment_fingerprint(
    model_name: str,
    config_dir: str = "configs",
) -> str:
    """Hash the data manifest, model config, and core implementation files."""
    paths = [
        "data/manifest.json",
        os.path.join(config_dir, "common.yaml"),
        os.path.join(config_dir, f"{model_name}.yaml"),
        os.path.join("src", "models", f"{model_name}.py"),
        os.path.join("src", "models", "base.py"),
        os.path.join("scripts", "train.py"),
        os.path.join("src", "training", "trainer.py"),
        os.path.join("src", "training", "early_stopping.py"),
        os.path.join("src", "evaluation", "evaluator.py"),
        os.path.join("src", "data", "negative_collector.py"),
        os.path.join("src", "data", "splitter.py"),
        os.path.join("src", "data", "sparsity.py"),
        os.path.join("src", "data", "text_encoder.py"),
        os.path.join("src", "utils", "config.py"),
        os.path.join("src", "utils", "config_schemas.py"),
        os.path.join("src", "losses", "bpr.py"),
        os.path.join("src", "losses", "contrastive.py"),
        os.path.join("src", "losses", "directau.py"),
        os.path.join("src", "losses", "debiased_infonce.py"),
        os.path.join("src", "losses", "hard_bpr.py"),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.replace("\\", "/").encode("utf-8"))
        with open(path, "rb") as file:
            digest.update(file.read())
    return digest.hexdigest()


def get_checkpoint_dir(model_name: str) -> str:
    """Get the checkpoint directory for a given model.

    Args:
        model_name: Name of the model (e.g., 'lightgcn', 'adaptive_gcl')

    Returns:
        Absolute path to the checkpoint directory
    """
    return os.path.join("results", "checkpoints", model_name)


def get_checkpoint_path(
    model_name: str,
    sparsity: float = 1.0,
    seed: int = 42,
    checkpoint_type: str = "run",
) -> str:
    """Get the standardized checkpoint path for a model run.

    Args:
        model_name: Name of the model
        sparsity: Sparsity ratio (0.25 to 1.0)
        seed: Random seed
        checkpoint_type: Type of checkpoint - 'run' (per-sparsity/seed) or 'best' (global best)

    Returns:
        Absolute path to the checkpoint file
    """
    checkpoint_dir = get_checkpoint_dir(model_name)
    sparsity_tag = f"s{int(sparsity * 100)}"

    if checkpoint_type == "best":
        return os.path.join(checkpoint_dir, f"{model_name}_best.pt")
    else:
        return os.path.join(checkpoint_dir, f"{model_name}_{sparsity_tag}_seed{seed}.pt")


def find_checkpoint(
    model_name: str,
    sparsity: float = 1.0,
    seed: int = 42,
) -> Optional[str]:
    """Find an existing checkpoint for the model, checking multiple possible locations.

    Priority order:
    1. Run-specific checkpoint (model_s{tag}_seed{seed}.pt)
    2. Global best checkpoint (model_best.pt)
    3. Legacy checkpoint locations (backwards compatibility)

    Args:
        model_name: Name of the model
        sparsity: Sparsity ratio
        seed: Random seed

    Returns:
        Path to the found checkpoint, or None if not found
    """
    candidates = [
        # Priority 1: Run-specific checkpoint
        get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="run"),
        # Priority 2: Global best checkpoint
        get_checkpoint_path(model_name, sparsity, seed, checkpoint_type="best"),
        # Priority 3: Legacy paths for backwards compatibility
        os.path.join("results", "checkpoints", f"{model_name}_best.pt"),
        os.path.join("results", "checkpoints", f"{model_name}_s{int(sparsity * 100)}_seed{seed}.pt"),
        os.path.join("artifacts", "checkpoints", f"{model_name}_s{int(sparsity * 100)}_seed{seed}.pt"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def ensure_checkpoint_dir(model_name: str) -> str:
    """Ensure the checkpoint directory exists and return its path.

    Args:
        model_name: Name of the model

    Returns:
        Path to the checkpoint directory
    """
    checkpoint_dir = get_checkpoint_dir(model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
