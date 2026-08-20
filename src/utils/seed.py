import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set global random seed for python random, numpy, and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic algorithms if available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
