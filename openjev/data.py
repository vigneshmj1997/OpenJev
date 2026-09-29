"""JSONL data loading and ragged-k batching.

One record per line. `type` picks the question type and defaults to "choice".

    choice (pick exactly one; options softmaxed together):
        {"state": "...", "options": ["a", "b", ...], "label": 1}
        {"state": "...", "options": [...], "probs": [0.7, 0.2, 0.1]}
    noul (true/false; the statement to judge lives in the state):
        {"type": "noul", "state": "...", "label": true}
        {"type": "noul", "state": "...", "prob": 0.8}
    multi (pick any number; each option gets its own independent probability):
        {"type": "multi", "state": "...", "options": [...], "labels": ["a", 2]}
        {"type": "multi", "state": "...", "options": [...], "probs": [0.9, 0.1, 0.6]}
    score (one level on an ordered low-to-high scale; levels softmaxed together):
        {"type": "score", "state": "...", "criteria": ["Calm", "Frustrated", "Very angry"], "label": 1}

`label`/`labels` may be indices or option text.

Any record may also carry the fields of the Jev question format:
    instructions: the question to ask (string, object or array).
    criteria: noul {"true": ..., "false": ...}; choice/multi {option: description},
        used instead of `options`; score a list of 2-10 levels, low to high.
`state` may be a string, object or array. Non-strings are encoded as JSON.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch

TYPES = ("noul", "choice", "multi", "score")
NOUL, CHOICE, MULTI, SCORE = range(len(TYPES))

# A noul question is a two-option choice between these fixed descriptions.
# Index 0 is always "true".
NOUL_OPTIONS = ["Yes, this is true.", "No, this is not true."]

MAX_OPTIONS = 255  # choice and multi
SCORE_LEVELS = (2, 10)


@dataclass
class Example:
    state: str
    options: list[str]  # answer names: what `answer` reports and labels match against
    # noul/choice/score: a distribution over options, sums to 1.
    # multi: P(selected) for each option independently.
    target: list[float]
    type: str = "choice"
    instructions: str = ""
    # What the encoder reads for each option; defaults to `options`.
    texts: list[str] | None = None
    # score: the levels exactly as the request gave them, for the answer's legend.
    levels: list | None = None

    def pair_texts(self) -> list[str]:
        """Second segment of each (state, option) pair: the question, then the option."""
        texts = self.texts or self.options
        return [f"{self.instructions}\n{t}" if self.instructions else t for t in texts]


def as_text(value) -> str:
    """Strings pass through; objects and arrays become compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _option_index(label, options: list[str], where: str) -> int:
    if isinstance(label, str):
        if label not in options:
            raise ValueError(f"{where}: label {label!r} is not one of the options")
        return options.index(label)
    if not 0 <= int(label) < len(options):
        raise ValueError(f"{where}: label index {label} out of range")
    return int(label)


def _noul_target(record: dict, where: str) -> list[float]:
    if "prob" in record:
        p = float(record["prob"])
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{where}: 'prob' must be in [0, 1]")
        return [p, 1.0 - p]
    if "label" in record:
        label = record["label"]
        if isinstance(label, str):
            if label.lower() not in ("true", "false"):
                raise ValueError(f"{where}: noul label must be true or false, got {label!r}")
            label = label.lower() == "true"
        return [1.0, 0.0] if bool(label) else [0.0, 1.0]
    raise ValueError(f"{where}: noul needs 'label' (true/false) or 'prob'")


def _choice_target(record: dict, options: list[str], where: str) -> list[float]:
    if "probs" in record:
        probs = [float(p) for p in record["probs"]]
        if len(probs) != len(options) or any(p < 0 for p in probs) or sum(probs) <= 0:
            raise ValueError(f"{where}: 'probs' must be non-negative, one per option")
        total = sum(probs)
        return [p / total for p in probs]
    if "label" in record:
        target = [0.0] * len(options)
        target[_option_index(record["label"], options, where)] = 1.0
        return target
    raise ValueError(f"{where}: need 'label' or 'probs'")


def _multi_target(record: dict, options: list[str], where: str) -> list[float]:
    if "probs" in record:
        probs = [float(p) for p in record["probs"]]
        if len(probs) != len(options) or any(not 0.0 <= p <= 1.0 for p in probs):
            raise ValueError(f"{where}: multi 'probs' must be in [0, 1], one per option")
        return probs
    if "labels" in record:
        if not isinstance(record["labels"], list):
            raise ValueError(f"{where}: multi 'labels' must be a list (it may be empty)")
        target = [0.0] * len(options)
        for label in record["labels"]:
            target[_option_index(label, options, where)] = 1.0
        return target
    raise ValueError(f"{where}: multi needs 'labels' or 'probs'")


def _options(record: dict, kind: str, where: str) -> tuple[list[str], list[str], list | None]:
    """(names, texts, levels) for a question. `texts` is what the encoder reads."""
    criteria = record.get("criteria")
    if kind == "noul":
        if criteria is None:
            return list(NOUL_OPTIONS), list(NOUL_OPTIONS), None
        if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
            raise ValueError(f"{where}: noul 'criteria' must have exactly 'true' and 'false'")
        return list(NOUL_OPTIONS), [f"Yes: {as_text(criteria['true'])}", f"No: {as_text(criteria['false'])}"], None

    if kind == "score":
        low, high = SCORE_LEVELS
        if not isinstance(criteria, list) or not low <= len(criteria) <= high:
            raise ValueError(f"{where}: score needs 'criteria' (list of {low} to {high} levels, low to high)")
        names = [as_text(level) for level in criteria]
        n = len(names)
        return names, [f"Level {i + 1} of {n}: {name}" for i, name in enumerate(names)], list(criteria)

    minimum = 1 if kind == "multi" else 2
    if criteria is not None:
        if not isinstance(criteria, dict):
            raise ValueError(f"{where}: {kind} 'criteria' must map each option to its description")
        names = [str(name) for name in criteria]
        texts = [f"{name}: {as_text(desc)}" for name, desc in criteria.items()]
    else:
        options = record.get("options")
        if not isinstance(options, list):
            raise ValueError(f"{where}: {kind} needs 'criteria' or 'options'")
        names = texts = [str(o) for o in options]
    if not minimum <= len(names) <= MAX_OPTIONS:
        raise ValueError(f"{where}: {kind} needs {minimum} to {MAX_OPTIONS} options, got {len(names)}")
    return names, list(texts), None


def question_from(record: dict, where: str = "") -> Example:
    """Validate type, state, instructions and options. The target is left as zeros."""
    kind = record.get("type", "choice")
    if kind not in TYPES:
        raise ValueError(f"{where}: type must be one of {TYPES}, got {kind!r}")
    state = record.get("state")
    if not isinstance(state, (str, dict, list)):
        raise ValueError(f"{where}: need 'state' (string, object or array)")
    instructions = record.get("instructions", "")
    if not isinstance(instructions, (str, dict, list)):
        raise ValueError(f"{where}: 'instructions' must be a string, object or array")
    names, texts, levels = _options(record, kind, where)
    return Example(as_text(state), names, [0.0] * len(names), kind,
                   instructions=as_text(instructions), texts=texts, levels=levels)


def questions_from_request(request: dict, where: str = "request") -> dict[str, Example]:
    """Validate a Jev request and return its questions by name, each carrying the shared state.

        {"state": ..., "model": "...", "questions": {"name": {"type": ..., "instructions": ..., "criteria": ...}}}

    `model` is not checked: the loaded model answers every request.
    """
    if not isinstance(request, dict):
        raise ValueError(f"{where}: must be an object")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{where}: need 'questions' (a non-empty map of name to question)")
    examples = {}
    for name, question in questions.items():
        at = f"{where}: question {name!r}"
        if not isinstance(question, dict):
            raise ValueError(f"{at}: must be an object")
        if "type" not in question or "instructions" not in question:
            raise ValueError(f"{at}: need 'type' and 'instructions'")
        if question["type"] in ("choice", "score") and "criteria" not in question:
            raise ValueError(f"{at}: {question['type']} needs 'criteria'")
        examples[name] = question_from({**question, "state": request.get("state")}, at)
    return examples


def parse_record(record: dict, where: str = "") -> Example:
    example = question_from(record, where)
    if example.type == "noul":
        example.target = _noul_target(record, where)
    elif example.type == "multi":
        example.target = _multi_target(record, example.options, where)
    else:  # choice and score: one distribution over the options
        example.target = _choice_target(record, example.options, where)
    return example


def load_jsonl(path: str) -> list[Example]:
    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                examples.append(parse_record(json.loads(line), f"{path}:{i}"))
    if not examples:
        raise ValueError(f"{path}: no examples")
    return examples


def split(examples: list[Example], fraction: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Carve off `fraction` of examples (at least one). Returns (rest, carved)."""
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    n = max(1, int(len(shuffled) * fraction))
    if n >= len(shuffled):
        raise ValueError("not enough examples to carve a held-out split")
    return shuffled[n:], shuffled[:n]


class Collator:
    """Flattens a batch of examples into (state, option) pairs."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[Example]) -> dict:
        states, options, group_index, option_index = [], [], [], []
        max_options = max(len(ex.options) for ex in batch)
        targets = torch.zeros(len(batch), max_options)
        for g, ex in enumerate(batch):
            for o, option in enumerate(ex.pair_texts()):
                states.append(ex.state)
                options.append(option)
                group_index.append(g)
                option_index.append(o)
            targets[g, : len(ex.options)] = torch.tensor(ex.target)

        enc = self.tokenizer(
            states, options, truncation="longest_first", max_length=self.max_length,
            padding=True, return_tensors="pt",
        )
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "group_index": torch.tensor(group_index),
            "option_index": torch.tensor(option_index),
            "kinds": torch.tensor([TYPES.index(ex.type) for ex in batch]),
            "num_groups": len(batch),
            "max_options": max_options,
            "targets": targets,
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        return out
