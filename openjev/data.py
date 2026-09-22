"""Dataset and collator.

One training example is (state, question, target distribution). The collator
flattens every option of every question in the batch into a single sequence batch,
then records which question each row belongs to so the model can scatter scores
back into a ragged (num_questions, max_k) grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

from .schema import Question, QuestionType


@dataclass
class Example:
    state: str
    question: Question
    target: list[float]     # distribution over question.keys, sums to 1


def _as_distribution(raw: dict[str, Any], question: Question) -> list[float]:
    """Accept either a hard label or an explicit soft distribution."""
    keys = question.keys

    if "target" in raw:
        dist = raw["target"]
        if isinstance(dist, dict):
            vec = [float(dist.get(k, 0.0)) for k in keys]
        else:
            vec = [float(x) for x in dist]
        total = sum(vec)
        if total <= 0:
            raise ValueError(f"target distribution sums to {total}")
        return [v / total for v in vec]

    label = raw["label"]
    if question.type is QuestionType.NOUL and isinstance(label, bool):
        label = "true" if label else "false"
    if isinstance(label, int) and question.type is QuestionType.SCORE:
        label = str(label)
    if label not in keys:
        raise ValueError(f"label {label!r} not in option keys {keys}")
    return [1.0 if k == label else 0.0 for k in keys]


class JevDataset(Dataset):
    """Reads JSONL where each line is one (state, question, label) triple.

    {"state": "...", "question": {"type": "choice", "instructions": "...",
                                  "criteria": {...}},
     "label": "billing"}
    """

    def __init__(self, path: str | Path) -> None:
        self.examples: list[Example] = []
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    question = Question.from_dict(raw["question"])
                    self.examples.append(
                        Example(
                            state=raw["state"],
                            question=question,
                            target=_as_distribution(raw, question),
                        )
                    )
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def build_pair_text(state: str, question: Question, option_desc: str) -> tuple[str, str]:
    """The (segment A, segment B) pair handed to the tokenizer.

    Segment A is the state; segment B is the instruction plus the one option
    description under consideration. Keeping the state in its own segment means
    the tokenizer truncates the state first when a sequence overflows, rather
    than silently eating the option text that carries the question.
    """
    return state, f"{question.instructions} [OPT] {option_desc}"


@dataclass
class JevCollator:
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, batch: list[Example]) -> dict[str, Any]:
        pair_a: list[str] = []
        pair_b: list[str] = []
        group_index: list[int] = []
        option_index: list[int] = []

        for qi, ex in enumerate(batch):
            for oi, desc in enumerate(ex.question.descriptions):
                a, b = build_pair_text(ex.state, ex.question, desc)
                pair_a.append(a)
                pair_b.append(b)
                group_index.append(qi)
                option_index.append(oi)

        encoded = self.tokenizer(
            pair_a,
            pair_b,
            padding=True,
            truncation="only_first",   # drop state tokens, never the option
            max_length=self.max_length,
            return_tensors="pt",
        )

        max_k = max(ex.question.k for ex in batch)
        target = torch.zeros(len(batch), max_k)
        for qi, ex in enumerate(batch):
            target[qi, : ex.question.k] = torch.tensor(ex.target)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "group_index": torch.tensor(group_index, dtype=torch.long),
            "option_index": torch.tensor(option_index, dtype=torch.long),
            "num_questions": len(batch),
            "max_k": max_k,
            "target_dist": target,
            "labels": target.argmax(dim=-1),
        }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
