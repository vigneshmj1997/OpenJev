"""Typed question/answer primitives matching the Jev API envelope.

The response shape is a deterministic function of the request: every key in an
answer is copied from the question that produced it, and the model contributes
only the leaf floats. These dataclasses encode that contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence


class QuestionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


# The scalar head scores (state, option-description) pairs, so a Noul question is
# just a two-option Choice whose options are these fixed descriptions. Overridable
# per question via `criteria`.
NOUL_DEFAULT_CRITERIA = {"true": "Yes, this is true.", "false": "No, this is not true."}


@dataclass
class Question:
    """One typed question. `criteria` carries the option descriptions that are
    actually encoded; the keys are labels the serializer zips back on at the end.

    For CHOICE/NOUL, `criteria` is a mapping of key -> description.
    For SCORE, `criteria` is an ordered sequence of level descriptions.
    """

    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str]

    def __post_init__(self) -> None:
        self.type = QuestionType(self.type)
        if self.type is QuestionType.NOUL and not self.criteria:
            self.criteria = dict(NOUL_DEFAULT_CRITERIA)
        if self.type is QuestionType.SCORE:
            if not isinstance(self.criteria, (list, tuple)):
                raise ValueError("score criteria must be an ordered list of level descriptions")
            if len(self.criteria) < 2:
                raise ValueError("score needs at least 2 levels")
        else:
            if not isinstance(self.criteria, dict):
                raise ValueError(f"{self.type.value} criteria must be a dict of key -> description")
            if self.type is QuestionType.NOUL and set(self.criteria) != {"true", "false"}:
                raise ValueError("noul criteria must have exactly the keys 'true' and 'false'")
            if self.type is QuestionType.CHOICE:
                if not 2 <= len(self.criteria) <= 255:
                    raise ValueError("choice needs between 2 and 255 options")

    @property
    def keys(self) -> list[str]:
        """Option labels, in the canonical order the head scores them."""
        if self.type is QuestionType.SCORE:
            return [str(i) for i in range(len(self.criteria))]
        if self.type is QuestionType.NOUL:
            return ["true", "false"]  # index 0 is always the positive class
        return list(self.criteria.keys())

    @property
    def descriptions(self) -> list[str]:
        """The text actually encoded beside the state, aligned with `keys`."""
        if self.type is QuestionType.SCORE:
            return list(self.criteria)
        return [self.criteria[k] for k in self.keys]

    @property
    def k(self) -> int:
        return len(self.keys)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Question":
        return cls(
            type=QuestionType(raw["type"]),
            instructions=raw["instructions"],
            criteria=raw.get("criteria") or {},
        )


def confidence_from(probs: Sequence[float]) -> float:
    """Peakedness of the distribution, normalized so uniform -> 0 and one-hot -> 1.

    Uses normalized entropy rather than max(p) because max(p) is insensitive to
    how the remaining mass is spread: [0.5, 0.5, 0.0] and [0.5, 0.25, 0.25] both
    have max(p) = 0.5, but the first has definitively eliminated an option while
    the second has ruled out nothing. Entropy ranks the first higher.
    """
    k = len(probs)
    if k < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(k)))


def serialize_answer(question: Question, probs: Sequence[float]) -> dict[str, Any]:
    """Build the response object for one question.

    This is the whole "API layer": it is pure arithmetic plus dict assembly over
    the caller's own keys. Nothing here can emit a value outside the schema,
    which is where the type-safety guarantee actually comes from.
    """
    probs = [float(p) for p in probs]
    keys = question.keys

    if question.type is QuestionType.NOUL:
        # No probabilities map (it would be redundant) and no confidence field:
        # for a binary question the probability *is* the certainty.
        return {"type": "noul", "noul": round(probs[0], 4)}

    table = {key: round(p, 4) for key, p in zip(keys, probs)}

    if question.type is QuestionType.CHOICE:
        return {
            "type": "choice",
            "choice": keys[max(range(len(probs)), key=probs.__getitem__)],
            "probabilities": table,
            "confidence": round(confidence_from(probs), 4),
        }

    # SCORE: the probability-weighted expectation over *ordered* levels, so the
    # result legitimately falls between levels (1.05 sits between "Frustrated"
    # and "Very angry"). This only means anything because the levels are ordered.
    expectation = sum(i * p for i, p in enumerate(probs))
    return {
        "type": "score",
        "score": round(expectation, 4),
        "legend": {str(i): desc for i, desc in enumerate(question.criteria)},
        "probabilities": table,
        "confidence": round(confidence_from(probs), 4),
    }
