import os
from typing import Any, Dict
import yaml

from src.utils.config_schemas import validate_config, validate_model_config


def load_config(
    model_name: str = "lightgcn",
    config_dir: str = "configs",
    validate: bool = True,
) -> Dict[str, Any]:
    """Load and merge common.yaml with model specific configuration file.

    Args:
        model_name: Name of the model to load specific config
        config_dir: Directory containing config files
        validate: Whether to validate config values (default: True)

    Returns:
        Validated config dictionary
    """
    common_path = os.path.join(config_dir, "common.yaml")
    if not os.path.exists(common_path):
        raise FileNotFoundError(f"Common config file not found at {common_path}")

    with open(common_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_config_path = os.path.join(config_dir, f"{model_name}.yaml")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
            if model_cfg:
                config.update(model_cfg)

    config["model_name"] = model_name

    # Apply validation
    if validate:
        config = validate_config(config)
        config = validate_model_config(config, model_name)

    return config
