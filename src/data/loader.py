import ast
import gzip
import json
import logging
import os
from typing import Optional, Tuple
import urllib.request
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_REVIEWS_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz"
DEFAULT_META_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_amazon_electronics(
    raw_dir: str = "data/raw",
    reviews_url: str = DEFAULT_REVIEWS_URL,
    meta_url: str = DEFAULT_META_URL,
) -> str:
    """Download Amazon Electronics reviews and metadata with visual download progress bars."""
    os.makedirs(raw_dir, exist_ok=True)
    dataset_dir = os.path.join(raw_dir, "amazon-electronics")
    os.makedirs(dataset_dir, exist_ok=True)

    reviews_file = os.path.join(dataset_dir, "reviews_Electronics_5.json.gz")
    meta_file = os.path.join(dataset_dir, "meta_Electronics.json.gz")

    if os.path.exists(reviews_file) and os.path.exists(meta_file):
        logger.info(f"Amazon Electronics dataset already exists at {dataset_dir}")
        return dataset_dir

    if not os.path.exists(reviews_file):
        logger.info(f"Downloading Reviews from {reviews_url}...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading Reviews") as t:
            urllib.request.urlretrieve(reviews_url, filename=reviews_file, reporthook=t.update_to)
        logger.info("Reviews download completed.")

    if not os.path.exists(meta_file):
        logger.info(f"Downloading Meta from {meta_url}...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading Metadata") as t:
            urllib.request.urlretrieve(meta_url, filename=meta_file, reporthook=t.update_to)
        logger.info("Meta download completed.")

    return dataset_dir


def get_df_from_json_gz(path: str, desc: Optional[str] = None) -> pd.DataFrame:
    """Load json.gz into Pandas DataFrame with live tqdm progress bar."""
    file_name = os.path.basename(path)
    pbar_desc = desc if desc is not None else f"Parsing {file_name}"
    logger.info(f"Parsing JSON GZ file: {path}")

    data = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=pbar_desc, unit=" lines", dynamic_ncols=True):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data.append(json.loads(line_str))
            except Exception:
                try:
                    data.append(ast.literal_eval(line_str))
                except Exception:
                    continue
    return pd.DataFrame.from_dict(data)


def load_raw_data(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Amazon Electronics reviews and metadata into DataFrames with progress bars."""
    reviews_path = os.path.join(dataset_dir, "reviews_Electronics_5.json.gz")
    meta_path = os.path.join(dataset_dir, "meta_Electronics.json.gz")

    logger.info("Loading reviews into DataFrame...")
    ratings_df = get_df_from_json_gz(reviews_path, desc="Loading Reviews (1.68M)")

    logger.info("Loading metadata into DataFrame...")
    items_df = get_df_from_json_gz(meta_path, desc="Loading Metadata (498K)")

    return ratings_df, items_df
