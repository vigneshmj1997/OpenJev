#!/usr/bin/env python3
"""CLI entry point.

    python scripts/train.py --config configs/modernbert_large.json
    python scripts/train.py --config configs/modernbert_base.json --epochs 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openjev.train import TrainConfig, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenJev")
    parser.add_argument("--config", type=str, help="JSON config file")

    fields = {f.name: f for f in dataclasses.fields(TrainConfig)}
    for name, f in fields.items():
        if f.type is bool or isinstance(f.default, bool):
            parser.add_argument(f"--{name}", type=lambda s: s.lower() == "true", default=None)
        else:
            parser.add_argument(f"--{name}", type=type(f.default), default=None)

    args = parser.parse_args()

    values = {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text()))
    # Explicit CLI flags override the file.
    values.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})

    unknown = set(values) - set(fields)
    if unknown:
        parser.error(f"unknown config keys: {sorted(unknown)}")

    train(TrainConfig(**values))


if __name__ == "__main__":
    main()
