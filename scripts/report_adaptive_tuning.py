"""Refresh a validation decision from completed artifacts without training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablate_adaptive import _write_experiment_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if not (root / "run_manifest.json").is_file():
        parser.error("A run_manifest.json is required")
    results = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(root.glob("*/s*/result.json"))]
    if any(not row.get("validation_only") for row in results):
        parser.error("Only validation-only results are supported")
    _write_experiment_tables(root, results)
    print((root / "decision.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
