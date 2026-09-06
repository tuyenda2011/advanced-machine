import ast
import gzip
import json
import logging
import os
import urllib.request
from typing import Any

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
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="Downloading Reviews"
        ) as t:
            urllib.request.urlretrieve(
                reviews_url, filename=reviews_file, reporthook=t.update_to
            )
        logger.info("Reviews download completed.")

    if not os.path.exists(meta_file):
        logger.info(f"Downloading Meta from {meta_url}...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc="Downloading Metadata"
        ) as t:
            urllib.request.urlretrieve(
                meta_url, filename=meta_file, reporthook=t.update_to
            )
        logger.info("Meta download completed.")

    return dataset_dir


def get_df_from_json_gz(path: str, desc: str | None = None) -> pd.DataFrame:
    """Load json.gz into Pandas DataFrame with live tqdm progress bar."""
    file_name = os.path.basename(path)
    pbar_desc = desc if desc is not None else f"Parsing {file_name}"
    logger.info(f"Parsing JSON GZ file: {path}")

    data = []
    ledger: dict[str, Any] = {
        "total_lines": 0,
        "blank_lines": 0,
        "parsed_rows": 0,
        "invalid_lines": 0,
        "error_examples": [],
    }
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=pbar_desc, unit=" lines", dynamic_ncols=True):
            ledger["total_lines"] += 1
            line_str = line.strip()
            if not line_str:
                ledger["blank_lines"] += 1
                continue
            try:
                record = json.loads(line_str)
            except (ValueError, TypeError):
                try:
                    record = ast.literal_eval(line_str)
                except (ValueError, SyntaxError, TypeError):
                    record = None
            if not isinstance(record, dict):
                ledger["invalid_lines"] += 1
                if len(ledger["error_examples"]) < 20:
                    ledger["error_examples"].append(
                        {
                            "line": ledger["total_lines"],
                            "reason": "not a parseable object",
                        }
                    )
                continue
            data.append(record)
            ledger["parsed_rows"] += 1
    if ledger["invalid_lines"]:
        logger.warning(
            "%s: %s invalid lines; examples (no review text): %s",
            file_name,
            ledger["invalid_lines"],
            ledger["error_examples"],
        )
    frame = pd.DataFrame.from_dict(data)
    frame.attrs["ingestion_ledger"] = ledger
    return frame


def load_raw_data(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw Amazon Electronics reviews and metadata into DataFrames with progress bars."""
    reviews_path = os.path.join(dataset_dir, "reviews_Electronics_5.json.gz")
    meta_path = os.path.join(dataset_dir, "meta_Electronics.json.gz")

    logger.info("Loading reviews into DataFrame...")
    ratings_df = get_df_from_json_gz(reviews_path, desc="Loading Reviews (1.68M)")

    logger.info("Loading metadata into DataFrame...")
    items_df = get_df_from_json_gz(meta_path, desc="Loading Metadata (498K)")

    return ratings_df, items_df
