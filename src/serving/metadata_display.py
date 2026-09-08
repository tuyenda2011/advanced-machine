"""Presentation helpers for incomplete item metadata.

These functions are deliberately separate from the metadata stored in
``mappings.pkl``.  A UI fallback must make an item identifiable without
turning a display label into training text or a verified metadata value.
"""

from collections.abc import Callable, Iterable

from src.data.preprocessing import clean_text, is_missing_text, metadata_flags

MISSING_BRAND_FILTER = "__missing_brand__"
MISSING_CATEGORY_FILTER = "__missing_category__"
ALL_BRANDS_LABEL = "All Brands"
ALL_CATEGORIES_LABEL = "All Categories"
MISSING_BRAND_LABEL = "Chưa rõ hãng"
MISSING_CATEGORY_LABEL = "Chưa rõ danh mục"
MISSING_TITLE_LABEL = "Chưa có tên sản phẩm"


def _asin(meta: dict) -> str:
    value = str(meta.get("original_id", "")).strip()
    return value or "Unknown ASIN"


def truncate_label(value: object, max_length: int = 45) -> str:
    """Return a readable label without adding ellipsis to short values."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    text = str(value).strip()
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    return text[: max_length - 3].rstrip() + "..."


def display_title(meta: dict, max_length: int | None = None) -> str:
    """Show a real title or an ASIN-based placeholder for missing titles."""
    flags = metadata_flags(meta)
    if flags["has_title"]:
        title = clean_text(meta.get("title"))
    else:
        title = f"{MISSING_TITLE_LABEL} · {_asin(meta)}"
    return truncate_label(title, max_length) if max_length is not None else title


def display_brand(meta: dict) -> str:
    """Show a brand while making an absent source value explicit."""
    return clean_text(meta.get("brand")) if metadata_flags(meta)["has_brand"] else MISSING_BRAND_LABEL


def display_brand_source(meta: dict) -> str:
    """Expose the provenance of a displayed brand without calling it verified."""
    if not metadata_flags(meta)["has_brand"]:
        return "missing"
    source = clean_text(meta.get("brand_source"))
    return source if not is_missing_text(source) else "metadata"


def display_category(meta: dict, max_length: int | None = None) -> str:
    """Show the original category path, or an explicit missing label."""
    if metadata_flags(meta)["has_category"]:
        category = clean_text(meta.get("categories"))
    else:
        category = MISSING_CATEGORY_LABEL
    return truncate_label(category, max_length) if max_length is not None else category


def category_leaf(meta: dict) -> str:
    """Return the exact terminal category used by the category filter."""
    if not metadata_flags(meta)["has_category"]:
        return ""
    parts = [part.strip() for part in clean_text(meta.get("categories")).split(">")]
    parts = [part for part in parts if not is_missing_text(part)]
    return parts[-1] if parts else ""


def brand_filter_value(meta: dict) -> str:
    """Return a stable value for the brand selectbox."""
    return clean_text(meta.get("brand")) if metadata_flags(meta)["has_brand"] else MISSING_BRAND_FILTER


def category_filter_value(meta: dict) -> str:
    """Return a stable terminal category value for exact filtering."""
    return category_leaf(meta) or MISSING_CATEGORY_FILTER


def unique_filter_options(
    item_metadata: dict[int, dict],
    extractor: Callable[[dict], str],
) -> list[str]:
    """Build deterministic, deduplicated filter values without an arbitrary cap.

    Missing-value sentinels are placed first so they remain easy to find in a
    long selectbox; real values follow in case-insensitive order.
    """
    values = {value for meta in item_metadata.values() if (value := extractor(meta))}
    return sorted(values, key=lambda value: (not value.startswith("__missing_"), value.casefold()))


def format_brand_filter(value: str) -> str:
    return MISSING_BRAND_LABEL if value == MISSING_BRAND_FILTER else value


def format_category_filter(value: str) -> str:
    return MISSING_CATEGORY_LABEL if value == MISSING_CATEGORY_FILTER else value


def metadata_counts(item_metadata: Iterable[dict]) -> dict[str, int]:
    """Return display-only coverage counts for the dashboard caption."""
    values = list(item_metadata)
    flags = [metadata_flags(meta) for meta in values]
    return {
        "items": len(values),
        "missing_titles": sum(not flag["has_title"] for flag in flags),
        "missing_brands": sum(not flag["has_brand"] for flag in flags),
    }
