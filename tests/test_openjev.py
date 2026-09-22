"""Tests for the pieces that are easy to get subtly wrong.

Run: python -m pytest tests/ -v   (or: python tests/test_openjev.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from openjev.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    fit_temperature,
)
from openjev.data import _as_distribution  # noqa: E402
from openjev.schema import Question, confidence_from, serialize_answer  # noqa: E402

CHOICE = Question.from_dict(
    {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    }
)
NOUL = Question.from_dict(
    {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    }
)
SCORE = Question.from_dict(
    {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    }
)


def test_envelope_matches_documented_shape():
    assert serialize_answer(NOUL, [0.95, 0.05]) == {"type": "noul", "noul": 0.95}

    team = serialize_answer(CHOICE, [0.88, 0.12, 0.0])
    assert team["choice"] == "billing"
    assert team["probabilities"] == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert set(team) == {"type", "choice", "probabilities", "confidence"}

    anger = serialize_answer(SCORE, [0.0, 0.95, 0.05])
    assert anger["score"] == 1.05  # 0*0.0 + 1*0.95 + 2*0.05
    assert anger["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_noul_omits_confidence():
    """For a binary question the probability IS the certainty, so a separate
    confidence field would be redundant."""
    assert "confidence" not in serialize_answer(NOUL, [0.7, 0.3])
    assert "probabilities" not in serialize_answer(NOUL, [0.7, 0.3])


def test_response_keys_come_from_request():
    """Every key in an answer must be traceable to the question that produced it."""
    answer = serialize_answer(CHOICE, [0.5, 0.3, 0.2])
    assert list(answer["probabilities"]) == list(CHOICE.criteria)
    assert answer["choice"] in CHOICE.criteria


def test_score_is_an_expectation_not_an_index():
    """Ordered levels make a between-levels answer meaningful."""
    assert serialize_answer(SCORE, [1.0, 0.0, 0.0])["score"] == 0.0
    assert serialize_answer(SCORE, [0.0, 0.0, 1.0])["score"] == 2.0
    assert serialize_answer(SCORE, [0.5, 0.0, 0.5])["score"] == 1.0
    assert serialize_answer(SCORE, [0.0, 0.5, 0.5])["score"] == 1.5


def test_confidence_bounds():
    assert confidence_from([1.0, 0.0, 0.0]) == 1.0          # one-hot
    assert confidence_from([1 / 3, 1 / 3, 1 / 3]) < 1e-9    # uniform
    assert confidence_from([0.5, 0.25, 0.25]) > confidence_from([0.34, 0.33, 0.33])


def test_confidence_distinguishes_spread():
    """max(p) calls these equally confident (both 0.5). Entropy does not: the
    two-way tie has definitively eliminated one option, while the thin spread
    has ruled out nothing, so the tie is the more informative distribution."""
    tie = confidence_from([0.5, 0.5, 0.0])
    spread = confidence_from([0.5, 0.25, 0.25])
    assert tie > spread


def test_label_forms():
    assert _as_distribution({"label": "technical"}, CHOICE) == [0.0, 1.0, 0.0]
    assert _as_distribution({"label": True}, NOUL) == [1.0, 0.0]
    assert _as_distribution({"label": 2}, SCORE) == [0.0, 0.0, 1.0]
    assert _as_distribution({"target": [0.0, 0.95, 0.05]}, SCORE) == [0.0, 0.95, 0.05]
    assert _as_distribution({"target": {"billing": 0.6, "technical": 0.4}}, CHOICE) == [0.6, 0.4, 0.0]
    assert _as_distribution({"target": [1, 3]}, NOUL) == [0.25, 0.75]  # renormalized


def test_noul_positive_class_is_index_zero():
    """probs[0] is what gets reported as `noul`, so `true` must sort first."""
    assert NOUL.keys[0] == "true"
    assert _as_distribution({"label": True}, NOUL)[0] == 1.0


def test_keys_align_with_descriptions():
    for q in (CHOICE, NOUL, SCORE):
        assert len(q.keys) == len(q.descriptions) == q.k


def test_temperature_reduces_ece_without_moving_argmax():
    torch.manual_seed(0)
    n, k = 2000, 3
    labels = torch.randint(0, k, (n,))
    logits = torch.randn(n, k)
    correct = torch.rand(n) < 0.70
    logits[torch.arange(n), labels] += torch.where(correct, 4.0, -1.0)
    logits *= 2.5  # make it overconfident

    before = logits.softmax(-1)
    temperature = fit_temperature(logits.clone(), labels)
    after = (logits / temperature).softmax(-1)

    assert temperature > 1.0
    assert expected_calibration_error(after, labels) < expected_calibration_error(before, labels)
    assert brier_score(after, labels) < brier_score(before, labels)
    # Temperature scaling is monotone: it cannot change which option wins.
    assert torch.equal(before.argmax(-1), after.argmax(-1))


def test_ragged_k_masking():
    """A k=2 noul and a k=3 choice must coexist in one batch without the noul
    leaking probability mass into its unused slot."""
    logits = torch.full((2, 3), float("-inf"))
    logits[0, :3] = torch.tensor([1.0, 2.0, 0.5])
    logits[1, :2] = torch.tensor([1.0, 1.0])
    probs = logits.log_softmax(-1).exp()

    assert probs[1, 2].item() == 0.0
    assert abs(probs[0].sum().item() - 1.0) < 1e-6
    assert abs(probs[1].sum().item() - 1.0) < 1e-6
    assert abs(probs[1, 0].item() - 0.5) < 1e-6


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
