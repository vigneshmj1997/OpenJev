#!/usr/bin/env python3
"""Answer questions with any OpenJev model. The default model is "openjev".

    python scripts/predict.py --state "Customer wants a refund" --options approve deny escalate
    python scripts/predict.py --type noul --state "The item arrived broken."
    python scripts/predict.py --type multi --state "Late and crushed" --options late damaged lost
    python scripts/predict.py --type score --state "Help! Payouts failing for 3 days." \
        --instructions "How frustrated is the customer?" --options Calm Frustrated "Very angry"
    python scripts/predict.py --model checkpoints/my-run --input data/eval.jsonl
    python scripts/predict.py --request request.json       # a Jev request: state + named questions

--model takes a preset (openjev, openjev-mini, openjev-large), a trained
checkpoint directory, or any Hugging Face encoder id. In --input files each
line's own "type" is used (default choice).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openjev import DEFAULT_MODEL, TYPES, OpenJev  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device")
    parser.add_argument("--state")
    parser.add_argument("--options", nargs="+", help="options; for --type score, the levels low to high")
    parser.add_argument("--instructions", default="", help="the question to ask about the state")
    parser.add_argument("--type", choices=TYPES, default="choice")
    parser.add_argument("--threshold", type=float, default=0.5, help="multi: select options with P >= this")
    parser.add_argument("--input", help="JSONL with 'type', 'state' and 'options' per line")
    parser.add_argument("--request", help="JSON file holding a Jev request (see docs/examples.md)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.request:
        with open(args.request) as f:
            request = json.load(f)
        print(json.dumps(OpenJev.load(args.model, device=args.device).run(request, args.threshold), indent=2))
        return
    if args.input:
        with open(args.input) as f:
            records = [json.loads(line) for line in f if line.strip()]
    elif args.state and (args.options or args.type == "noul"):
        key = "criteria" if args.type == "score" else "options"
        records = [{"type": args.type, "state": args.state, "instructions": args.instructions, key: args.options}]
    else:
        parser.error("give --state and --options (not needed for --type noul), --input, or --request")

    jev = OpenJev.load(args.model, device=args.device)
    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        for record, result in zip(chunk, jev.ask_batch(chunk, args.threshold)):
            print(json.dumps({"state": record["state"], **result}))


if __name__ == "__main__":
    main()
