"""Apply verified metadata overrides to the active processed dataset."""

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.metadata_overrides import apply_brand_overrides
from src.data.preprocessing import summarize_metadata_quality
from src.data.provenance import sha256_file
from src.data.text_encoder import PINNED_REVISION, encode_item_metadata


def _write_metadata_csv(item_metadata, path):
    frame = pd.DataFrame.from_dict(item_metadata, orient="index")
    frame.index.name = "i_idx"
    frame.reset_index().to_csv(path, index=False)


def main(processed_dir=None, manifest_path=None, overrides_path=None):
    processed = Path(processed_dir or REPO_ROOT / "data" / "processed")
    manifest_file = Path(manifest_path or REPO_ROOT / "data" / "manifest.json")
    overrides = Path(overrides_path or REPO_ROOT / "data" / "metadata_overrides.csv")
    with (processed / "mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)

    override_report = apply_brand_overrides(mappings["item_metadata"], overrides)
    mappings["stats"]["metadata_overrides"] = override_report
    mappings["stats"]["metadata_quality"] = summarize_metadata_quality(
        mappings["item_metadata"]
    )

    stage = REPO_ROOT / ".tmp" / "metadata_override_stage"
    backup = (
        REPO_ROOT
        / ".tmp"
        / f"metadata_override_backup_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    )
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    backup.mkdir(parents=True)

    artifact_names = (
        "mappings.pkl",
        "item_text_embeddings.pt",
        "item_text_embeddings.pt.json",
    )
    for name in (*artifact_names, "manifest.json"):
        source = manifest_file if name == "manifest.json" else processed / name
        shutil.copy2(source, backup / name)

    with (stage / "mappings.pkl").open("wb") as stream:
        pickle.dump(mappings, stream)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    revision = manifest.get("text_features", {}).get("revision", PINNED_REVISION)
    encode_item_metadata(
        mappings["item_metadata"],
        num_items=len(mappings["item2id"]),
        save_path=str(stage / "item_text_embeddings.pt"),
        revision=revision,
    )

    _write_metadata_csv(mappings["item_metadata"], stage / "metadata.csv")
    manifest["statistics"]["metadata_overrides"] = override_report
    manifest["statistics"]["metadata_quality"] = mappings["stats"]["metadata_quality"]
    text_provenance = json.loads(
        (stage / "item_text_embeddings.pt.json").read_text(encoding="utf-8")
    )
    text_provenance.pop("item_text_mask", None)
    manifest["text_features"].update(text_provenance)
    for entry in manifest.get("artifacts", []):
        artifact = Path(entry["path"])
        candidate = stage / artifact.name
        if candidate.exists():
            entry["sha256"] = sha256_file(candidate)
    (stage / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for name in artifact_names:
        shutil.copy2(stage / name, processed / name)
    shutil.copy2(stage / "metadata.csv", processed / "csv" / "metadata.csv")
    shutil.copy2(stage / "manifest.json", manifest_file)

    missing_brand_csv = processed / "csv" / "missing_brand_products.csv"
    if missing_brand_csv.exists():
        missing = pd.read_csv(missing_brand_csv)
        resolved = {
            meta["original_id"]
            for meta in mappings["item_metadata"].values()
            if meta.get("has_brand")
        }
        missing = missing[~missing["original_id"].isin(resolved)]
        missing.to_csv(missing_brand_csv, index=False, encoding="utf-8-sig")

    print(json.dumps({
        "applied": override_report["applied"],
        "entries": override_report["entries"],
        "metadata_quality": mappings["stats"]["metadata_quality"],
        "backup": str(backup),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir")
    parser.add_argument("--manifest_path")
    parser.add_argument("--overrides_path")
    args = parser.parse_args()
    main(args.processed_dir, args.manifest_path, args.overrides_path)
