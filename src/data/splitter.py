import logging
from typing import Any, Dict, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def check_split_connectivity(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Check whether all users and items present in val/test exist in the train graph."""
    train_users = set(train_df["u_idx"].unique())
    train_items = set(train_df["i_idx"].unique())

    val_users = set(val_df["u_idx"].unique())
    val_items = set(val_df["i_idx"].unique())

    test_users = set(test_df["u_idx"].unique())
    test_items = set(test_df["i_idx"].unique())

    eval_users = val_users.union(test_users)
    eval_items = val_items.union(test_items)

    missing_users = eval_users - train_users
    missing_items = eval_items - train_items

    report = {
        "num_train_users": len(train_users),
        "num_train_items": len(train_items),
        "num_eval_users": len(eval_users),
        "num_eval_items": len(eval_items),
        "missing_users_count": len(missing_users),
        "missing_items_count": len(missing_items),
        "missing_users": sorted(list(missing_users)),
        "missing_items": sorted(list(missing_items)),
        "is_connected": len(missing_users) == 0 and len(missing_items) == 0,
    }

    if not report["is_connected"]:
        logger.warning(
            f"Graph split connectivity violation: {len(missing_users)} users and "
            f"{len(missing_items)} items present in eval splits are missing from train graph."
        )
    else:
        logger.info("Graph split connectivity check PASSED: 100% of eval nodes exist in train graph.")

    return report


def relocate_disconnected(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Move one interaction per isolated item/user from val or test into train to guarantee connectivity."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 1. Fix missing items
    train_items = set(train_df["i_idx"].unique())
    all_eval_items = set(val_df["i_idx"].unique()).union(set(test_df["i_idx"].unique()))
    missing_items = all_eval_items - train_items

    if missing_items:
        logger.info(f"Relocating interactions for {len(missing_items)} disconnected items to train...")
        val_trans_idx = []
        test_trans_idx = []
        for item in missing_items:
            v_matches = val_df[val_df["i_idx"] == item]
            if len(v_matches) > 0:
                val_trans_idx.append(v_matches.index[0])
            else:
                t_matches = test_df[test_df["i_idx"] == item]
                if len(t_matches) > 0:
                    test_trans_idx.append(t_matches.index[0])

        if val_trans_idx:
            transfer_val = val_df.loc[val_trans_idx]
            train_df = pd.concat([train_df, transfer_val], ignore_index=True)
            val_df = val_df.drop(index=val_trans_idx).reset_index(drop=True)

        if test_trans_idx:
            transfer_test = test_df.loc[test_trans_idx]
            train_df = pd.concat([train_df, transfer_test], ignore_index=True)
            test_df = test_df.drop(index=test_trans_idx).reset_index(drop=True)

    # 2. Fix missing users
    train_users = set(train_df["u_idx"].unique())
    all_eval_users = set(val_df["u_idx"].unique()).union(set(test_df["u_idx"].unique()))
    missing_users = all_eval_users - train_users

    if missing_users:
        logger.info(f"Relocating interactions for {len(missing_users)} disconnected users to train...")
        val_trans_u = []
        test_trans_u = []
        for user in missing_users:
            v_matches = val_df[val_df["u_idx"] == user]
            if len(v_matches) > 0:
                val_trans_u.append(v_matches.index[0])
            else:
                t_matches = test_df[test_df["u_idx"] == user]
                if len(t_matches) > 0:
                    test_trans_u.append(t_matches.index[0])

        if val_trans_u:
            transfer_val_u = val_df.loc[val_trans_u]
            train_df = pd.concat([train_df, transfer_val_u], ignore_index=True)
            val_df = val_df.drop(index=val_trans_u).reset_index(drop=True)

        if test_trans_u:
            transfer_test_u = test_df.loc[test_trans_u]
            train_df = pd.concat([train_df, transfer_test_u], ignore_index=True)
            test_df = test_df.drop(index=test_trans_u).reset_index(drop=True)

    return train_df, val_df, test_df


def chronological_per_user_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    enforce_connectivity: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform chronological per-user split: oldest interactions -> train, next -> val, latest -> test.

    Guarantees every user has at least 1 train, 1 val, and 1 test interaction.
    """
    logger.info(f"Performing chronological per-user split (val_ratio={val_ratio}, test_ratio={test_ratio})...")

    # Sort interactions by user and timestamp
    df_sorted = df.sort_values(by=["u_idx", "timestamp"]).reset_index(drop=True)

    train_list = []
    val_list = []
    test_list = []

    for _, group in df_sorted.groupby("u_idx", sort=False):
        n = len(group)
        if n < 3:
            # Fallback if a user has < 3 interactions
            n_test = 1
            n_val = 1
            n_train = max(1, n - 2)
        else:
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))
            n_train = n - n_val - n_test

            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - n_train - n_val

        u_train = group.iloc[:n_train]
        u_val = group.iloc[n_train : n_train + n_val]
        u_test = group.iloc[n_train + n_val :]

        train_list.append(u_train)
        val_list.append(u_val)
        test_list.append(u_test)

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    if enforce_connectivity:
        train_df, val_df, test_df = relocate_disconnected(train_df, val_df, test_df)

    check_split_connectivity(train_df, val_df, test_df)
    verify_no_leakage(train_df, val_df, test_df)

    logger.info(
        f"Split completed: Train={len(train_df)} edges, Val={len(val_df)} edges, Test={len(test_df)} edges."
    )

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> bool:
    """Verify that train, validation, and test edge sets have zero intersection."""
    train_pairs: Set[Tuple[int, int]] = set(zip(train_df["u_idx"], train_df["i_idx"]))
    val_pairs: Set[Tuple[int, int]] = set(zip(val_df["u_idx"], val_df["i_idx"]))
    test_pairs: Set[Tuple[int, int]] = set(zip(test_df["u_idx"], test_df["i_idx"]))

    intersection_train_val = train_pairs.intersection(val_pairs)
    intersection_train_test = train_pairs.intersection(test_pairs)
    intersection_val_test = val_pairs.intersection(test_pairs)

    assert (
        len(intersection_train_val) == 0
    ), f"DATA LEAKAGE DETECTED! Train and Val share {len(intersection_train_val)} interaction edges!"
    assert (
        len(intersection_train_test) == 0
    ), f"DATA LEAKAGE DETECTED! Train and Test share {len(intersection_train_test)} interaction edges!"
    assert (
        len(intersection_val_test) == 0
    ), f"DATA LEAKAGE DETECTED! Val and Test share {len(intersection_val_test)} interaction edges!"

    logger.info("PASSED DATA LEAKAGE CHECK: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅.")
    return True


def global_temporal_split(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    enforce_connectivity: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform global chronological cutoff split based on global timestamps.

    Oldest (1 - val_ratio - test_ratio) -> train, next val_ratio -> val, latest test_ratio -> test.
    """
    logger.info(f"Performing global chronological cutoff split (val={val_ratio}, test={test_ratio})...")
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)

    n_total = len(df_sorted)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_df = df_sorted.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[n_train + n_val :].copy().reset_index(drop=True)

    if enforce_connectivity:
        train_df, val_df, test_df = relocate_disconnected(train_df, val_df, test_df)

    check_split_connectivity(train_df, val_df, test_df)
    verify_no_leakage(train_df, val_df, test_df)
    return train_df, val_df, test_df


