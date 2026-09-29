#!/usr/bin/env python3
"""Train an OpenJev model.

    python scripts/train.py --train-file data/train.jsonl --eval-file data/eval.jsonl
    python scripts/train.py --config configs/openjev.json
    python scripts/train.py --config configs/openjev.json --model openjev-mini
    python scripts/train.py --train-file data/train.jsonl --model microsoft/deberta-v3-base

Values in --config are defaults; flags on the command line override them.
"""

import argparse
import dataclasses
import json
import os
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openjev.train import TrainArgs, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON file of TrainArgs fields")
    hints = typing.get_type_hints(TrainArgs)
    for field in dataclasses.fields(TrainArgs):
        flag = "--" + field.name.replace("_", "-")
        kind = hints[field.name]
        if kind is bool:
            parser.add_argument(flag, action="store_true", default=None)
            continue
        base = next((t for t in typing.get_args(kind) if t is not type(None)), kind)
        parser.add_argument(flag, type=base, default=None)
    return parser


def main() -> None:
    parsed = vars(build_parser().parse_args())
    values = {}
    config_path = parsed.pop("config")
    if config_path:
        with open(config_path) as f:
            values.update(json.load(f))
        print(f"Loaded config {config_path}")
    values.update({k: v for k, v in parsed.items() if v is not None})
    if "train_file" not in values:
        raise SystemExit("--train-file is required (or set train_file in --config)")
    summary = train(TrainArgs(**values))
    print(json.dumps({"best": summary["best"], "output_dir": summary["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
