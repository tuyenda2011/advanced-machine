"""Build the optional quality-controlled AdaptiveGCL text feature view."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.adaptive_metadata import ADAPTIVE_TEXT_POLICY
from src.data.bundle import BundleError, resolve_bundle
from src.data.text_encoder import (
    DEFAULT_ENCODER,
    PINNED_REVISION,
    build_adaptivegcl_text_view,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Bundle path or data/current.json")
    parser.add_argument(
        "--output_dir",
        help="Directory for the derived files (default: active data/processed)",
    )
    parser.add_argument("--model_name", default=DEFAULT_ENCODER)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args(argv)
    try:
        bundle = resolve_bundle(args.bundle)
    except BundleError as exc:
        parser.error(str(exc))
    if args.output_dir:
        output = Path(args.output_dir).resolve()
    elif bundle.legacy:
        output = bundle.root
    else:
        output = ROOT / ".tmp" / "adaptivegcl_text" / bundle.build_id
    output.mkdir(parents=True, exist_ok=True)
    with bundle.artifact("mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    save_path = output / "adaptivegcl_text_embeddings.pt"
    tensor = build_adaptivegcl_text_view(
        mappings["item_metadata"],
        len(mappings["item2id"]),
        save_path=str(save_path),
        model_name=args.model_name,
        revision=args.revision,
    )
    sidecar = json.loads(Path(str(save_path) + ".json").read_text(encoding="utf-8"))
    sidecar.update(
        {
            "feature_view": "adaptivegcl_quality",
            "metadata_policy": ADAPTIVE_TEXT_POLICY,
            "ssl_eligible_items": int(sum(sidecar["ssl_item_mask"])),
        }
    )
    Path(str(save_path) + ".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(save_path), "shape": list(tensor.shape), "ssl_eligible_items": sidecar["ssl_eligible_items"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
