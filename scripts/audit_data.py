import pickle
import torch
import pandas as pd
import numpy as np

with open("data/processed/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
stats = mappings["stats"]

train_df = pd.read_parquet("data/processed/train.parquet")
val_df = pd.read_parquet("data/processed/val.parquet")
test_df = pd.read_parquet("data/processed/test.parquet")
disliked_df = pd.read_parquet("data/processed/disliked_interactions.parquet")
text_emb = torch.load("data/processed/item_text_embeddings.pt", weights_only=False)

u_cnt = train_df.groupby("u_idx").size()
i_cnt = train_df.groupby("i_idx").size()

train_u, train_i = set(train_df["u_idx"]), set(train_df["i_idx"])
val_u, val_i = set(val_df["u_idx"]), set(val_df["i_idx"])
test_u, test_i = set(test_df["u_idx"]), set(test_df["i_idx"])
tot = len(train_df) + len(val_df) + len(test_df)

print("=== DATASET OVERVIEW ===")
print(f"Users: {stats['num_users']:,}")
print(f"Items: {stats['num_items']:,}")
print(f"Total 5-core interactions: {stats['num_interactions']:,}")
print(f"Density: {stats['density']:.6%}")

print("\n=== SPLIT PROPORTIONS ===")
print(f"Train: {len(train_df):,} ({len(train_df)/tot*100:.2f}%)")
print(f"Val:   {len(val_df):,} ({len(val_df)/tot*100:.2f}%)")
print(f"Test:  {len(test_df):,} ({len(test_df)/tot*100:.2f}%)")
print(f"Disliked Pool (1-2 stars): {len(disliked_df):,}")

print("\n=== GRAPH CONNECTIVITY (ZERO LEAKAGE) ===")
print(f"Disconnected Val Users: {len(val_u - train_u)}")
print(f"Disconnected Val Items: {len(val_i - train_i)}")
print(f"Disconnected Test Users: {len(test_u - train_u)}")
print(f"Disconnected Test Items: {len(test_i - train_i)}")

print("\n=== USER DEGREE IN TRAIN SET ===")
print(f"Min: {u_cnt.min()}, 25%: {u_cnt.quantile(0.25)}, Median: {u_cnt.median()}, Mean: {u_cnt.mean():.2f}, 75%: {u_cnt.quantile(0.75)}, Max: {u_cnt.max()}")

print("\n=== ITEM DEGREE IN TRAIN SET ===")
print(f"Min: {i_cnt.min()}, 25%: {i_cnt.quantile(0.25)}, Median: {i_cnt.median()}, Mean: {i_cnt.mean():.2f}, 75%: {i_cnt.quantile(0.75)}, Max: {i_cnt.max()}")

norms = torch.norm(text_emb, dim=-1)
print("\n=== TEXT EMBEDDINGS TENSOR ===")
print(f"Shape: {text_emb.shape}")
print(f"Has NaN: {torch.isnan(text_emb).any().item()}")
print(f"Has Inf: {torch.isinf(text_emb).any().item()}")
print(f"L2 Norm Min: {norms.min().item():.4f}, Max: {norms.max().item():.4f}, Mean: {norms.mean().item():.4f}")
