import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_sparse_train_set(
    train_df: pd.DataFrame, sparsity_ratio: float = 1.0, seed: int = 42
) -> pd.DataFrame:
    """Sample p% of training interactions per user, ensuring each user keeps at least 1 interaction edge.

    Validation and Test sets MUST remain 100% unchanged.
    """
    if sparsity_ratio >= 1.0:
        return train_df.copy()

    logger.info(f"Applying per-user sparsity sampling at ratio={sparsity_ratio} (seed={seed})...")
    np.random.seed(seed)

    sampled_list = []
    for _, group in train_df.groupby("u_idx", sort=False):
        n = len(group)
        n_keep = max(1, int(np.round(n * sparsity_ratio)))

        if n_keep >= n:
            sampled_list.append(group)
        else:
            indices = np.random.choice(group.index, size=n_keep, replace=False)
            sampled_list.append(group.loc[indices])

    sparse_train_df = (
        pd.concat(sampled_list, ignore_index=True)
        .sort_values(by=["u_idx", "timestamp"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Sparsity ratio {sparsity_ratio}: original edges={len(train_df)} -> retained edges={len(sparse_train_df)} ({len(sparse_train_df)/len(train_df):.2%})"
    )

    return sparse_train_df
