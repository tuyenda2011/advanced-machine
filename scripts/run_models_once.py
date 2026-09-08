"""Canonical entry point for one real run per model at one density.

The historical ``smoke_test_models.py`` name remains supported for existing
commands; output naming now distinguishes smoke checks from evaluations.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from scripts.smoke_test_models import main as runner_main

    return runner_main()


if __name__ == "__main__":
    raise SystemExit(main())
