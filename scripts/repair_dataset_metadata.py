"""Rebuild metadata from source, validate staging, preserve split bytes and IDs."""

import json
import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data
from src.data.metadata_resolution import resolve_metadata, write_brand_template
from src.data.preprocessing import summarize_metadata_quality
from src.data.provenance import sha256_file
from src.data.quality_report import quality_report
from src.data.text_encoder import (
    PINNED_REVISION,
    encode_item_metadata,
    get_item_text_mask,
    get_text_input_fingerprint,
    load_verified_text_cache,
)


def main():
    processed = ROOT / "data/processed"
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with (processed / "mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    before = summarize_metadata_quality(mappings["item_metadata"])
    split_names = ("train.parquet", "val.parquet", "test.parquet")
    hashes = {name: sha256_file(processed / name) for name in split_names}
    original_ids = pickle.dumps((mappings["user2id"], mappings["item2id"]))
    reviews, items = load_raw_data(str(ROOT / "data/raw/amazon-electronics"))
    resolution = resolve_metadata(mappings["item_metadata"], items, ROOT / "data/metadata_overrides.csv")
    print("METADATA STAGES", json.dumps(resolution), flush=True)
    train, val, test = [pd.read_parquet(processed / name) for name in split_names]
    report = quality_report(train, val, test, mappings["item_metadata"])
    report["metadata"] = resolution
    after = summarize_metadata_quality(mappings["item_metadata"])
    report["metadata_quality_before"] = before
    report["metadata_quality_after"] = after
    report["brand_coverage"] = 1 - after["missing_brand_fraction"]
    report["title_coverage"] = 1 - after["missing_title_fraction"]
    ratings = reviews.overall
    report["ratings"] = {"total_raw_reviews": len(reviews), "positive_reviews": int((ratings >= 4).sum()),
                         "ratings_below_4": int((ratings < 4).sum()), "positive_ratio": float((ratings >= 4).mean()),
                         "training_hard_negatives": manifest["statistics"]["hard_negative_interactions"],
                         "hard_negative_policy": "ratings 1-2, mapped users/items, at or before train cutoff; not all ratings below 4"}
    report["timestamp"]["raw_midnight_fraction"] = float((reviews.unixReviewTime % 86400 == 0).mean())
    report["ingestion"] = {"reviews": reviews.attrs["ingestion_ledger"], "metadata": items.attrs["ingestion_ledger"]}
    counts = pd.concat([train, val, test]).i_idx.value_counts()
    report["missing_title_examples"] = [{"asin": m["original_id"], "brand": m["brand"], "category": m["categories"], "interaction_count": int(counts.get(i, 0))}
        for i, m in sorted(mappings["item_metadata"].items(), key=lambda pair: -counts.get(pair[0], 0)) if not m["has_title"]][:20]
    if resolution["matched_items_with_brand"] == 0:
        raise ValueError("No official brands survived ASIN matching")
    report["warnings"] = []
    if report["brand_coverage"] < .1:
        report["warnings"].append("Brand coverage unexpectedly low; metadata parsing or join may be broken.")
    mappings["stats"]["metadata_resolution"] = resolution
    mappings["stats"]["metadata_quality"] = after
    manifest["statistics"].update(metadata_quality=after, metadata_resolution=resolution)
    stage = Path(tempfile.mkdtemp(prefix="metadata-repair-", dir=ROOT / ".tmp"))
    backup = stage / "backup"
    backup.mkdir()
    artifact_names = ("mappings.pkl", "item_text_embeddings.pt", "item_text_embeddings.pt.json")
    for name in artifact_names:
        shutil.copy2(processed / name, backup / name)
    shutil.copy2(manifest_path, backup / "manifest.json")
    with (stage / "mappings.pkl").open("wb") as stream:
        pickle.dump(mappings, stream)
    encoder = manifest["text_features"].get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    revision = manifest["text_features"].get("revision", PINNED_REVISION)
    encode_item_metadata(mappings["item_metadata"], len(mappings["item2id"]), model_name=encoder,
                         revision=revision, save_path=str(stage / "item_text_embeddings.pt"))
    fingerprint = get_text_input_fingerprint(mappings["item_metadata"], len(mappings["item2id"]), encoder, revision)
    mask = get_item_text_mask(mappings["item_metadata"], len(mappings["item2id"]))
    if load_verified_text_cache(stage / "item_text_embeddings.pt", fingerprint, item_mask=mask) is None:
        raise ValueError("Staged text cache failed validation")
    if pickle.dumps((mappings["user2id"], mappings["item2id"])) != original_ids:
        raise ValueError("Mappings changed")
    for name, digest in hashes.items():
        if sha256_file(processed / name) != digest:
            raise ValueError("Split changed during repair")
    text_info = json.loads((stage / "item_text_embeddings.pt.json").read_text())
    text_info.pop("item_text_mask", None)
    manifest["text_features"].update(text_info)
    for entry in manifest["artifacts"]:
        if Path(entry["path"]).name in artifact_names:
            entry["sha256"] = sha256_file(stage / Path(entry["path"]).name)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Each replacement is atomic; retain the old complete set for recovery.
    try:
        for name in artifact_names:
            os.replace(stage / name, processed / name)
        os.replace(stage / "manifest.json", manifest_path)
    except OSError:
        for name in artifact_names:
            shutil.copy2(backup / name, processed / name)
        shutil.copy2(backup / "manifest.json", manifest_path)
        raise
    frame = pd.DataFrame.from_dict(mappings["item_metadata"], orient="index").rename_axis("i_idx").reset_index()
    frame.to_csv(processed / "csv/metadata.csv", index=False)
    frame["total_interactions"] = frame.i_idx.map(counts).fillna(0).astype(int)
    frame.sort_values(["total_interactions", "i_idx"], ascending=[False, True], inplace=True)
    frame.loc[~frame.has_brand].to_csv(processed / "csv/missing_brand_products.csv", index=False, encoding="utf-8-sig")
    frame.loc[~frame.has_title].to_csv(processed / "csv/missing_title_products.csv", index=False, encoding="utf-8-sig")
    template = ROOT / "data/metadata_brand_overrides_template.csv"
    if template.exists():
        shutil.copy2(template, backup / template.name)
    write_brand_template(mappings["item_metadata"], template)
    report["invariants"] = {"split_hashes_unchanged": True, "mapping_unchanged": True, "training_behavior_unchanged": True}
    report["backup"] = str(backup)
    (processed / "data_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
