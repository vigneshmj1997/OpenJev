"""Inference: state + questions -> the Jev response envelope.

Every question in a call is scored against the same state in one batched pass and
in isolation from its siblings, so adding a question costs tokens rather than a
round trip. The response is assembled from the caller's own keys; the model
contributes only the leaf floats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .data import build_pair_text
from .model import OpenJevModel
from .schema import Question, serialize_answer


class OpenJev:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        checkpoint = Path(checkpoint)
        blob = torch.load(checkpoint / "openjev.pt", map_location="cpu", weights_only=False)
        config = blob["config"]

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = config["max_length"]

        self.model = OpenJevModel(
            base_model=config["base_model"],
            pooling=config["pooling"],
            head_dropout=0.0,
        )
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device).eval()

        self.model_name = f"openjev-{checkpoint.name}"

    @torch.no_grad()
    def decide(
        self,
        state: str | dict | list,
        questions: dict[str, dict[str, Any] | Question],
        batch_size: int = 64,
    ) -> dict[str, Any]:
        if not isinstance(state, str):
            import json

            state = json.dumps(state, ensure_ascii=False)

        parsed = {
            name: (q if isinstance(q, Question) else Question.from_dict(q))
            for name, q in questions.items()
        }

        pair_a: list[str] = []
        pair_b: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        for name, question in parsed.items():
            start = len(pair_a)
            for desc in question.descriptions:
                a, b = build_pair_text(state, question, desc)
                pair_a.append(a)
                pair_b.append(b)
            spans[name] = (start, len(pair_a))

        scores: list[torch.Tensor] = []
        for i in range(0, len(pair_a), batch_size):
            encoded = self.tokenizer(
                pair_a[i : i + batch_size],
                pair_b[i : i + batch_size],
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            scores.append(self.model.score_candidates(**encoded).float().cpu())
        flat = torch.cat(scores) if scores else torch.empty(0)

        answers: dict[str, Any] = {}
        for name, question in parsed.items():
            start, end = spans[name]
            # Softmax within the question only — questions never compete.
            probs = torch.softmax(flat[start:end] / self.model.temperature.cpu(), dim=-1)
            answers[name] = serialize_answer(question, probs.tolist())

        input_tokens = sum(
            len(self.tokenizer(a, b)["input_ids"]) for a, b in zip(pair_a, pair_b)
        )
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
