"""Inference: load any OpenJev model and get calibrated answers.

    from openjev import OpenJev
    jev = OpenJev.load()                       # default: "openjev"

    jev.noul("The customer says the item arrived broken.")
    # {"type": "noul", "noul": 0.93}
    jev.choice("Refund request for a damaged item", ["approve", "deny", "escalate"])
    # {"type": "choice", "choice": "approve",
    #  "probabilities": {"approve": 0.81, "deny": 0.04, "escalate": 0.15}, "confidence": 0.52}
    jev.multi("Arrived late and the box was crushed", ["late", "damaged", "wrong item"])
    # {"type": "multi", "selected": ["late", "damaged"],
    #  "probabilities": {"late": 0.94, "damaged": 0.88, "wrong item": 0.03}}
    jev.score("Help! My payouts have been failing for 3 days.", ["Calm", "Frustrated", "Very angry"],
              instructions="How frustrated is the customer?")
    # {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
    #  "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.82}

    # The Jev request format: one state, named questions, answered together.
    jev.run({"state": "...", "questions": {"department": {"type": "choice", "instructions": "...",
                                                          "criteria": {"billing": "...", "technical": "..."}}}})
    # {"model": "openjev", "answers": {"department": {...}},
    #  "usage": {"input_tokens": 120, "output_tokens": 0}, "elapsed": 35}

See docs/examples.md for every question type in the request format.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Sequence

import torch

from .config import DEFAULT_MODEL, base_config, find_checkpoint
from .data import Collator, Example, question_from, questions_from_request
from .model import OpenJevModel, load_tokenizer, probabilities
from .train import autocast, model_inputs, pick_device, to_device


def confidence_from(probs: Sequence[float]) -> float:
    """Peakedness of a distribution: uniform -> 0, one-hot -> 1.

    Normalized entropy rather than max(p): [0.5, 0.5, 0.0] and
    [0.5, 0.25, 0.25] share max(p) = 0.5, but the first has ruled an option
    out and the second has not.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(k)))


def answer(example: Example, probs: Sequence[float], threshold: float = 0.5) -> dict:
    """The response for one question. Every key comes from the question itself."""
    probs = [float(p) for p in probs]
    if example.type == "noul":
        return {"type": "noul", "noul": probs[0]}  # index 0 is "true"
    table = dict(zip(example.options, probs))
    if example.type == "multi":
        return {"type": "multi",
                "selected": [o for o, p in table.items() if p >= threshold],
                "probabilities": table}
    if example.type == "score":
        # Levels are keyed by index from "0" (low). The score is the expected
        # index, so it can land between levels.
        return {"type": "score",
                "score": sum(i * p for i, p in enumerate(probs)),
                "legend": {str(i): level for i, level in enumerate(example.levels or example.options)},
                "probabilities": {str(i): p for i, p in enumerate(probs)},
                "confidence": confidence_from(probs)}
    return {"type": "choice",
            "choice": example.options[max(range(len(probs)), key=probs.__getitem__)],
            "probabilities": table,
            "confidence": confidence_from(probs)}


class OpenJev:
    def __init__(self, model: OpenJevModel, tokenizer, device: torch.device, name: str = DEFAULT_MODEL):
        self.name = name  # reported as "model" in `run` responses
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.collate = Collator(tokenizer, model.config.max_length)

    @classmethod
    def load(cls, name_or_path: str = DEFAULT_MODEL, device: str | None = None) -> "OpenJev":
        """Load a trained checkpoint, preset name, or any Hugging Face encoder.

        A preset resolves to its trained checkpoint under checkpoints/ when one
        exists. Otherwise the encoder loads with untrained heads, which is only
        useful as a starting point for training.
        """
        checkpoint = find_checkpoint(name_or_path)
        if checkpoint:
            model = OpenJevModel.from_pretrained(checkpoint)
            tokenizer = load_tokenizer(model.config, checkpoint)
        else:
            warnings.warn(
                f"No trained OpenJev checkpoint for {name_or_path!r}; the scoring heads are "
                "untrained and their probabilities are meaningless. Train first with scripts/train.py."
            )
            config = base_config(name_or_path)
            model = OpenJevModel(config)
            tokenizer = load_tokenizer(config)
        return cls(model, tokenizer, pick_device(device), name=name_or_path)

    @torch.no_grad()
    def _score(self, examples: list[Example]) -> tuple[list[list[float]], list[int]]:
        """Calibrated probabilities per example (padding dropped) and input tokens per example."""
        batch = to_device(self.collate(examples), self.device)
        with autocast(self.device):
            logits = self.model(**model_inputs(batch))
        probs = probabilities(logits.float(), batch["kinds"], self.model.temperature).cpu()
        pair_tokens = batch["attention_mask"].sum(1).cpu()
        tokens = torch.zeros(len(examples), dtype=torch.long).index_add_(0, batch["group_index"].cpu(), pair_tokens)
        return [row[: len(ex.options)].tolist() for ex, row in zip(examples, probs)], tokens.tolist()

    def probabilities(self, examples: list[Example]) -> list[list[float]]:
        """Calibrated probabilities per example, one per option, padding dropped."""
        return self._score(examples)[0]

    def run(self, request: dict, threshold: float = 0.5) -> dict:
        """Answer a Jev request: one `state` and a map of named `questions`.

        All questions go through the encoder in one batch. `usage.output_tokens`
        is always 0: nothing is generated. `elapsed` is in milliseconds.
        """
        start = time.perf_counter()
        questions = questions_from_request(request)
        probs, tokens = self._score(list(questions.values()))
        answers = {name: answer(ex, p, threshold) for (name, ex), p in zip(questions.items(), probs)}
        return {"model": self.name,
                "answers": answers,
                "usage": {"input_tokens": sum(tokens), "output_tokens": 0},
                "elapsed": round((time.perf_counter() - start) * 1000)}

    def ask_batch(self, questions: list[dict], threshold: float = 0.5) -> list[dict]:
        """Answer mixed questions, each a dict with its own "state" (see data.question_from)."""
        examples = [question_from(q, f"question {i}") for i, q in enumerate(questions)]
        return [answer(ex, p, threshold) for ex, p in zip(examples, self.probabilities(examples))]

    def noul(self, state, instructions="", criteria: dict | None = None) -> dict:
        """Is the statement true (or the answer to `instructions` yes)? Returns P(true)."""
        q = {"type": "noul", "state": state, "instructions": instructions}
        if criteria is not None:
            q["criteria"] = criteria
        return self.ask_batch([q])[0]

    def choice(self, state, options: list[str] | dict, instructions="") -> dict:
        """Pick exactly one option. `options` may be a list or {option: description}."""
        return self.ask_batch([_with_options({"type": "choice", "state": state, "instructions": instructions},
                                             options)])[0]

    def multi(self, state, options: list[str] | dict, threshold: float = 0.5, instructions="") -> dict:
        """Pick any number of options. Each probability is independent."""
        return self.ask_batch([_with_options({"type": "multi", "state": state, "instructions": instructions},
                                             options)], threshold)[0]

    def score(self, state, levels: list, instructions="") -> dict:
        """Place the state on an ordered scale of 2-10 levels, low to high."""
        return self.ask_batch([{"type": "score", "state": state, "instructions": instructions,
                                "criteria": levels}])[0]

    # ---- choice shorthands, kept for existing callers -------------------------

    def predict_batch(self, items: list[tuple[str, list[str]]]) -> list[dict[str, float]]:
        results = self.ask_batch([{"state": c, "options": list(o)} for c, o in items])
        return [r["probabilities"] for r in results]

    def predict(self, state: str, options: list[str]) -> dict[str, float]:
        return self.predict_batch([(state, options)])[0]


def _with_options(question: dict, options: list[str] | dict) -> dict:
    key = "criteria" if isinstance(options, dict) else "options"
    return {**question, key: options if isinstance(options, dict) else list(options)}
