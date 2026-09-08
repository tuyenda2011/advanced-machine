"""Resolve metadata with explicit sources and conservative anchored brand matching."""

import csv
import re
from collections import Counter
from pathlib import Path

from src.data.preprocessing import clean_text, metadata_flags


def usable(value):
    return metadata_flags({"brand": value})["has_brand"]


def resolve_metadata(item_metadata, raw_items, overrides_path):
    if raw_items["asin"].duplicated().any():
        raise ValueError("Duplicate ASIN in raw metadata")
    raw = raw_items.set_index("asin").to_dict("index")
    with open(overrides_path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    overrides = {}
    for row in rows:
        asin = row["asin"].strip()
        brand = row.get("brand_override", row.get("brand", ""))
        if not usable(brand):
            continue
        if asin in overrides:
            raise ValueError(f"Duplicate override: {asin}")
        overrides[asin] = clean_text(brand)

    # Learn spellings only from repeated official brands. Match title prefix,
    # never a compatible-device mention or category. Ambiguous short words abstain.
    counts = Counter(clean_text(r.get("brand")) for r in raw.values() if usable(r.get("brand")))
    excluded = {"generic", "unknown", "none", "case", "new", "for", "with", "the",
                "cable", "dual", "ultra", "green", "image", "exact", "cctv", "bass",
                "ipod", "ipad", "iphone", "nexus", "kindle", "galaxy", "replacement",
                "digital", "professional", "wireless", "premium", "universal"}
    brands = sorted((b for b, n in counts.items() if n >= 3 and len(b) >= 4
                     and b.casefold() not in excluded), key=lambda b: (-len(b), b))
    pattern = re.compile(r"^(" + "|".join(re.escape(b) for b in brands) + r")(?=$|\s|[®™:])", re.IGNORECASE) if brands else None
    before = sum(metadata_flags(m)["has_brand"] for m in item_metadata.values())
    matched = official = 0
    for meta in item_metadata.values():
        asin = meta["original_id"]
        source = raw.get(asin, {})
        matched += asin in raw
        old_title = meta.get("title")
        if metadata_flags({"title": source.get("title"), "original_id": asin})["has_title"]:
            meta["title"], meta["title_source"] = clean_text(source["title"]), "metadata"
        elif metadata_flags(meta)["has_title"]:
            meta["title"], meta["title_source"] = old_title, "existing"
        else:
            meta["title"], meta["title_source"] = "Unknown Item", "missing"
        old_brand = meta.get("brand")
        brand = source.get("brand")
        if usable(brand):
            meta["brand"], meta["brand_source"] = clean_text(brand), "metadata"
            official += 1
        elif asin in overrides:
            meta["brand"], meta["brand_source"] = overrides[asin], "override"
        else:
            match = pattern.match(meta["title"]) if pattern and meta["title_source"] != "missing" else None
            meta["brand"] = match.group(1) if match else "Unknown"
            meta["brand_source"] = "title_fallback" if match else "missing"
        if meta["brand"] != old_brand:
            meta.setdefault("brand_original", old_brand)
        meta.update(metadata_flags(meta))
    return {
        "raw_meta_records": len(raw_items), "raw_meta_records_with_brand": sum(counts.values()),
        "filtered_asin_count": len(item_metadata), "filtered_asin_matched": matched,
        "matched_items_with_brand": official, "processed_brands_before": before,
        "processed_brands_after": sum(m["has_brand"] for m in item_metadata.values()),
        "brand_sources": dict(Counter(m["brand_source"] for m in item_metadata.values())),
        "title_sources": dict(Counter(m["title_source"] for m in item_metadata.values())),
        "fallback_policy": "official_brand_frequency_ge3_length_ge4_title_prefix_v1",
    }


def write_brand_template(item_metadata, path):
    entered = {}
    if Path(path).exists():
        with open(path, encoding="utf-8-sig", newline="") as stream:
            entered = {r["asin"]: r.get("brand_override", r.get("brand", "")) for r in csv.DictReader(stream)}
    with open(path, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["asin", "title", "current_brand", "brand_override", "brand_source", "category"])
        writer.writeheader()
        for meta in item_metadata.values():
            writer.writerow({"asin": meta["original_id"], "title": meta["title"], "current_brand": meta["brand"],
                             "brand_override": entered.get(meta["original_id"], ""), "brand_source": meta["brand_source"], "category": meta["categories"]})
