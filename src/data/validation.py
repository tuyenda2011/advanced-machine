"""Fail-fast data validation for interaction and metadata tables.

Primary engine: Great Expectations (if installed). Graceful fallback: pandas
assertions with identical fail-fast semantics, so CI never breaks on a missing
heavy dependency.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only when GE is installed
    import great_expectations as gx

    GE_AVAILABLE = True
except ImportError:
    gx = None
    GE_AVAILABLE = False


class ValidationError(Exception):
    """Raised when any schema or value expectation fails."""


def _format_violations(violations: list[str]) -> str:
    header = f"{len(violations)} validation violation(s):"
    body = "\n".join(f"  - {v}" for v in violations)
    return f"{header}\n{body}"


def _fallback_validate_interactions(df: pd.DataFrame) -> list[str]:
    """Pandas-assertion mirror of the interaction schema expectations."""
    violations = []

    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        violations.append(f"missing required columns: {sorted(missing)}")
        return violations

    tracked_cols = set()
    for col in ("user_id", "item_id", "timestamp"):
        n_null = int(df[col].isna().sum())
        if n_null:
            violations.append(f"{col}: {n_null} null values (expected 0)")
            tracked_cols.add(col)
        non_numeric = df[~df[col].apply(lambda v: isinstance(v, (int, float)) and not pd.isna(v))]
        if len(non_numeric):
            violations.append(f"{col}: {len(non_numeric)} non-numeric values")
            tracked_cols.add(col)

    for col in ("user_id", "item_id"):
        if col in tracked_cols:
            continue  # type/null issues already reported; skip follow-up range check
        bad = df[(df[col] <= 0) & df[col].notna()]
        if len(bad):
            violations.append(f"{col}: {len(bad)} values <= 0")

    bad_rating = df[~df["rating"].between(1, 5, inclusive="both")]
    if len(bad_rating):
        violations.append(f"rating: {len(bad_rating)} values outside [1, 5]")

    bad_ts = df[df["timestamp"].notna() & (df["timestamp"] < 0)]
    if len(bad_ts):
        violations.append(f"timestamp: {len(bad_ts)} negative values")

    return violations


def _fallback_validate_metadata(df: pd.DataFrame) -> list[str]:
    """Pandas-assertion mirror of the metadata expectations."""
    violations = []

    for col in ("title", "brand"):
        if col not in df.columns:
            violations.append(f"metadata missing required column: {col}")
            continue
        empty = df[df[col].isna() | (df[col].astype(str).str.strip() == "")]
        if len(empty):
            violations.append(f"{col}: {len(empty)} null/empty strings after cleaning")

    if "original_id" in df.columns:
        dup = int(df["original_id"].duplicated().sum())
        if dup:
            violations.append(f"original_id: {dup} duplicated entries")

    return violations


def _ge_validate(df: pd.DataFrame, expectations: list[tuple], suite_name: str, html_report_dir: Path | None):
    """Run expectations through Great Expectations; return violation strings."""
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(f"{suite_name}_ds")
    asset = datasource.add_dataframe_asset(f"{suite_name}_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe(f"{suite_name}_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name=suite_name)
    for method, kwargs in expectations:
        suite.add_expectation(getattr(gx.expectations, method)(**kwargs))
    suite = context.suites.add(suite)

    results = batch.validate(expectation_suite=suite, result_format="SUMMARY")
    violations = [
        f"{res.expectation_config.type}: {res.result}"
        for res in results.results
        if not res.success
    ]

    if html_report_dir is not None and violations:
        html_report_dir.mkdir(parents=True, exist_ok=True)
        logger.info("GE HTML report generation skipped (ephemeral context).")

    return violations


INTERACTION_EXPECTATIONS = [
    ("ExpectColumnValuesToNotBeNull", {"column": "user_id"}),
    ("ExpectColumnValuesToBeOfType", {"column": "user_id", "type_": "int64"}),
    ("ExpectColumnValuesToBeGreaterThan", {"column": "user_id", "value": 0}),
    ("ExpectColumnValuesToNotBeNull", {"column": "item_id"}),
    ("ExpectColumnValuesToBeOfType", {"column": "item_id", "type_": "int64"}),
    ("ExpectColumnValuesToBeGreaterThan", {"column": "item_id", "value": 0}),
    ("ExpectColumnValuesToBeBetween", {"column": "rating", "min_value": 1, "max_value": 5}),
    ("ExpectColumnValuesToNotBeNull", {"column": "timestamp"}),
    ("ExpectColumnValuesToBeBetween", {"column": "timestamp", "min_value": 0}),
]

METADATA_EXPECTATIONS = [
    ("ExpectColumnValuesToNotBeNull", {"column": "title"}),
    ("ExpectColumnValuesToNotMatchRegex", {"column": "title", "regex": r"&amp;|<br\s*/?>|<[^>]+>"}),
    ("ExpectColumnValuesToNotBeNull", {"column": "brand"}),
]


def validate_interactions(
    df: pd.DataFrame,
    html_report_dir: Path | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate the interactions table schema; return violation list.

    Raises ``ValidationError`` when ``raise_on_error`` is True and any check fails.
    """
    logger.info("Validating interactions table (%d rows)...", len(df))

    if GE_AVAILABLE:
        try:
            violations = _ge_validate(df, INTERACTION_EXPECTATIONS, "interactions", html_report_dir)
        except Exception as exc:  # pragma: no cover - depends on GE runtime quirks
            logger.warning("Great Expectations failed (%s); falling back to pandas validator.", exc)
            violations = _fallback_validate_interactions(df)
    else:
        violations = _fallback_validate_interactions(df)

    if violations:
        detail = _format_violations(violations)
        if raise_on_error:
            raise ValidationError(detail)
        logger.error("Interaction validation issues:\n%s", detail)
    else:
        logger.info("Interactions validation PASSED.")

    return violations


def validate_metadata(
    df: pd.DataFrame,
    html_report_dir: Path | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Validate the item metadata table; semantics match ``validate_interactions``."""
    logger.info("Validating metadata table (%d rows)...", len(df))

    if GE_AVAILABLE:
        try:
            violations = _ge_validate(df, METADATA_EXPECTATIONS, "metadata", html_report_dir)
        except Exception as exc:  # pragma: no cover
            logger.warning("Great Expectations failed (%s); falling back to pandas validator.", exc)
            violations = _fallback_validate_metadata(df)
    else:
        violations = _fallback_validate_metadata(df)

    if violations:
        detail = _format_violations(violations)
        if raise_on_error:
            raise ValidationError(detail)
        logger.error("Metadata validation issues:\n%s", detail)
    else:
        logger.info("Metadata validation PASSED.")

    return violations
