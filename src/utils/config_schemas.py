"""Config validation schemas to ensure type safety and constraints.

This module provides optional validation using pydantic if available,
otherwise falls back to basic Python validation.
"""

from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Check if pydantic is available
try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore
    Field = None
    field_validator = None
    logger.info("Pydantic not found. Using basic config validation.")


class ConfigValidator:
    """Simple config validator that works with or without pydantic."""

    @staticmethod
    def validate_split_ratios(v: List[float]) -> List[float]:
        """Validate split ratios sum to 1.0."""
        if len(v) != 3:
            raise ValueError("split_ratios must have exactly 3 elements (train, val, test)")
        if not abs(sum(v) - 1.0) < 1e-6:
            raise ValueError("split_ratios must sum to 1.0")
        if any(r <= 0 for r in v):
            raise ValueError("split_ratios must all be positive")
        return v

    @staticmethod
    def validate_embedding_dim(dim: int) -> int:
        """Validate embedding dimension is in valid range."""
        if not (8 <= dim <= 512):
            raise ValueError("embedding_dim must be between 8 and 512")
        return dim

    @staticmethod
    def validate_num_layers(layers: int) -> int:
        """Validate number of layers is in valid range."""
        if not (1 <= layers <= 16):
            raise ValueError("num_layers must be between 1 and 16")
        return layers

    @staticmethod
    def validate_learning_rate(lr: float) -> float:
        """Validate learning rate is in valid range."""
        if lr <= 0 or lr > 1.0:
            raise ValueError("learning_rate must be between 0 and 1.0")
        if lr < 1e-6:
            logger.warning(f"learning_rate {lr} is very small, may cause underflow")
        return lr

    @staticmethod
    def validate_batch_size(batch_size: int) -> int:
        """Validate batch size is positive."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size

    @staticmethod
    def validate_sparsity_levels(levels: List[float]) -> List[float]:
        """Validate sparsity levels are in valid range."""
        if not levels:
            raise ValueError("sparsity levels cannot be empty")
        if any(not (0 < level <= 1.0) for level in levels):
            raise ValueError("sparsity levels must be in range (0, 1.0]")
        return sorted(levels, reverse=True)


if PYDANTIC_AVAILABLE:
    class DatasetConfig(BaseModel):
        """Dataset configuration schema."""
        name: str
        reviews_url: str
        meta_url: str
        raw_dir: str = "data/raw"
        processed_dir: str = "data/processed"
        positive_rating_threshold: float = Field(default=4.0, ge=1.0, le=5.0)
        min_user_interactions: int = Field(default=5, ge=1)
        min_item_interactions: int = Field(default=5, ge=1)
        split_ratios: List[float] = Field(default=[0.8, 0.1, 0.1])
        split_seed: int = Field(default=42, ge=0)

        @field_validator("split_ratios")
        @classmethod
        def validate_split_ratios(cls, v: List[float]) -> List[float]:
            if len(v) != 3:
                raise ValueError("split_ratios must have exactly 3 elements")
            if not abs(sum(v) - 1.0) < 1e-6:
                raise ValueError("split_ratios must sum to 1.0")
            return v

    class ModelConfig(BaseModel):
        """Model architecture configuration schema."""
        embedding_dim: int = Field(default=64, ge=8, le=512)
        num_layers: int = Field(default=3, ge=1, le=16)

    class TrainingConfig(BaseModel):
        """Training hyperparameters schema."""
        batch_size: int = Field(default=2048, ge=1, le=65536)
        learning_rate: float = Field(default=0.001, gt=0, le=1.0)
        weight_decay: float = Field(default=1e-4, ge=0, le=1.0)
        epochs: int = Field(default=100, ge=1, le=10000)
        early_stopping_patience: int = Field(default=20, ge=1)
        seed: int = Field(default=42, ge=0)
        num_workers: int = Field(default=0, ge=0)

    class AdaptiveGCLConfig(BaseModel):
        """AdaptiveGCL-specific configuration schema."""
        text_dim: int = Field(default=384, ge=64, le=1024)
        ssl_temp: float = Field(default=0.2, gt=0, le=2.0)
        ssl_reg: float = Field(default=0.1, ge=0, le=10.0)
        dirichlet_reg: float = Field(default=0.01, ge=0, le=1.0)
        node_dropout: float = Field(default=0.0, ge=0, lt=1.0)
        tau_plus: float = Field(default=0.1, ge=0, lt=1.0)
        hard_neg_alpha: float = Field(default=0.2, ge=0)
        hard_neg_margin: float = Field(default=0.5, ge=0)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config dictionary with basic checks.

    Args:
        config: Raw config dictionary

    Returns:
        Validated config dictionary (unchanged if pydantic not available)
    """
    if not PYDANTIC_AVAILABLE:
        # Use basic validation
        validator = ConfigValidator()
        try:
            # Validate model config
            if "model" in config:
                model = config["model"]
                if "embedding_dim" in model:
                    model["embedding_dim"] = validator.validate_embedding_dim(model["embedding_dim"])
                if "num_layers" in model:
                    model["num_layers"] = validator.validate_num_layers(model["num_layers"])

            # Validate training config
            if "training" in config:
                training = config["training"]
                if "learning_rate" in training:
                    training["learning_rate"] = validator.validate_learning_rate(training["learning_rate"])
                if "batch_size" in training:
                    training["batch_size"] = validator.validate_batch_size(training["batch_size"])

            # Validate dataset config
            if "dataset" in config:
                dataset = config["dataset"]
                if "split_ratios" in dataset:
                    dataset["split_ratios"] = validator.validate_split_ratios(dataset["split_ratios"])

            # Validate sparsity config
            if "sparsity" in config and "levels" in config["sparsity"]:
                config["sparsity"]["levels"] = validator.validate_sparsity_levels(
                    config["sparsity"]["levels"]
                )

        except ValueError as e:
            logger.warning(f"Config validation warning: {e}")

        return config

    # Use pydantic validation
    from pydantic import ValidationError
    try:
        if "dataset" in config:
            config["dataset"] = DatasetConfig(**config["dataset"]).model_dump()
        if "model" in config:
            config["model"] = ModelConfig(**config["model"]).model_dump()
        if "training" in config:
            config["training"] = TrainingConfig(**config["training"]).model_dump()
    except ValidationError as e:
        logger.warning(f"Config validation error: {e}")

    return config


def validate_model_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Validate model-specific config.

    Args:
        config: Raw config dictionary
        model_name: Name of the model

    Returns:
        Validated config dictionary
    """
    if model_name == "adaptive_gcl" and "adaptive_gcl" in config:
        if PYDANTIC_AVAILABLE:
            from pydantic import ValidationError
            try:
                config["adaptive_gcl"] = AdaptiveGCLConfig(**config["adaptive_gcl"]).model_dump()
            except ValidationError as e:
                logger.warning(f"AdaptiveGCL config validation error: {e}")
        else:
            validator = ConfigValidator()
            ada_cfg = config["adaptive_gcl"]
            if "ssl_temp" in ada_cfg:
                ada_cfg["ssl_temp"] = max(0.001, min(2.0, ada_cfg["ssl_temp"]))
            if "ssl_reg" in ada_cfg:
                ada_cfg["ssl_reg"] = max(0, min(10.0, ada_cfg["ssl_reg"]))
            if "dirichlet_reg" in ada_cfg:
                ada_cfg["dirichlet_reg"] = max(0, min(1.0, ada_cfg["dirichlet_reg"]))

    return config
