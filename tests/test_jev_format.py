"""The Jev request format: instructions, criteria, score, and run()."""

import json
import math
import re
from pathlib import Path

import pytest
import torch

from openjev import OpenJev, OpenJevConfig, OpenJevModel
from openjev.data import NOUL_OPTIONS, SCORE, TYPES, parse_record, question_from, questions_from_request
from openjev.infer import answer, confidence_from
from openjev.model import _per_type, load_tokenizer
from test_openjev import tiny_encoder_dir  # noqa: F401

DOCS = Path(__file__).resolve().parent.parent / "docs" / "examples.md"


def test_criteria_become_option_texts():
    q = question_from({"type": "choice", "state": "s", "instructions": "Which team?",
                       "criteria": {"billing": "Payments", "technical": "Bugs"}})
    assert q.options == ["billing", "technical"]
    assert q.pair_texts() == ["Which team?\nbilling: Payments", "Which team?\ntechnical: Bugs"]

    noul = question_from({"type": "noul", "state": "s", "criteria": {"true": "Urgent", "false": "Not urgent"}})
    assert noul.options == NOUL_OPTIONS
    assert noul.pair_texts() == ["Yes: Urgent", "No: Not urgent"]

    plain = question_from({"state": "s", "options": ["a", "b"]})  # the flat format is unchanged
    assert plain.pair_texts() == ["a", "b"] and plain.instructions == ""


def test_structured_state_and_instructions_are_json():
    q = question_from({"type": "noul", "state": {"plan": "enterprise"}, "instructions": {"question": "Q?", "policy": "P"}})
    assert json.loads(q.state) == {"plan": "enterprise"}
    assert json.loads(q.instructions) == {"question": "Q?", "policy": "P"}


def test_score_levels_and_labels():
    levels = ["Calm", {"level": "Angry"}]
    q = parse_record({"type": "score", "state": "s", "criteria": levels, "label": 1})
    assert q.type == "score" and q.levels == levels and q.target == [0.0, 1.0]
    assert q.pair_texts() == ["Level 1 of 2: Calm", 'Level 2 of 2: {"level": "Angry"}']
    assert parse_record({"type": "score", "state": "s", "criteria": ["a", "b", "c"], "label": "c"}).target == [0, 0, 1.0]


@pytest.mark.parametrize("question", [
    {"type": "score", "criteria": ["only one"]},
    {"type": "score", "criteria": [str(i) for i in range(11)]},
    {"type": "choice", "criteria": {str(i): "" for i in range(256)}},
    {"type": "choice", "criteria": ["a", "b"]},
    {"type": "noul", "criteria": {"yes": "a", "no": "b"}},
])
def test_limits_and_bad_criteria(question):
    with pytest.raises(ValueError):
        question_from({"state": "s", **question})


def test_request_validation():
    ok = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Q?"}}}
    assert list(questions_from_request(ok)) == ["q"]
    for bad in ({"state": "s", "questions": {}},
                {"state": "s", "questions": {"q": {"type": "noul"}}},
                {"state": "s", "questions": {"q": {"type": "choice", "instructions": "Q?", "options": ["a", "b"]}}},
                {"questions": {"q": {"type": "noul", "instructions": "Q?"}}}):
        with pytest.raises(ValueError):
            questions_from_request(bad)


def test_score_answer_matches_jev_docs():
    q = question_from({"type": "score", "state": "s", "criteria": ["Calm", "Frustrated", "Very angry"]})
    out = answer(q, [0.0, 0.95, 0.05])
    assert out["score"] == pytest.approx(1.05)
    assert out["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert out["probabilities"] == {"0": 0.0, "1": 0.95, "2": 0.05}
    assert out["confidence"] == pytest.approx(confidence_from([0.0, 0.95, 0.05]))


def test_old_checkpoint_temperatures_stretch_to_every_type():
    assert _per_type(torch.tensor([2.0])).tolist() == [2.0] * len(TYPES)
    stretched = _per_type(torch.tensor([1.5, 2.0, 3.0]))  # saved before score existed
    assert stretched.tolist() == [1.5, 2.0, 3.0, 2.0] and len(stretched) == len(TYPES)
    assert stretched[SCORE] == stretched[1]  # score reuses the choice temperature


def test_run_answers_every_docs_example(tiny_encoder_dir):  # noqa: F811
    config = OpenJevConfig(encoder=tiny_encoder_dir, max_length=64)
    jev = OpenJev(OpenJevModel(config), load_tokenizer(config), torch.device("cpu"), name="tiny")

    blocks = [json.loads(b) for b in re.findall(r"```json\n(.*?)```", DOCS.read_text(), re.S)]
    requests = [b for b in blocks if "questions" in b]
    assert len(requests) >= 9
    for request in requests:
        response = jev.run(request)
        assert response["model"] == "tiny" and response["usage"]["input_tokens"] > 0
        assert set(response["answers"]) == set(request["questions"])
        for name, got in response["answers"].items():
            question = request["questions"][name]
            assert got["type"] == question["type"]
            if got["type"] == "noul":
                assert 0.0 <= got["noul"] <= 1.0
            else:
                assert math.isclose(sum(got["probabilities"].values()), 1.0, rel_tol=1e-5)
            if got["type"] == "choice":
                assert set(got["probabilities"]) == set(question["criteria"])
            if got["type"] == "score":
                assert got["legend"] == {str(i): c for i, c in enumerate(question["criteria"])}
                assert 0.0 <= got["score"] <= len(question["criteria"]) - 1


def test_run_matches_one_question_at_a_time(tiny_encoder_dir):  # noqa: F811
    config = OpenJevConfig(encoder=tiny_encoder_dir, max_length=64)
    jev = OpenJev(OpenJevModel(config), load_tokenizer(config), torch.device("cpu"))
    state = "item broken refund"
    both = jev.run({"state": state, "questions": {
        "a": {"type": "score", "instructions": "how late", "criteria": ["late", "broken", "the"]},
        "b": {"type": "choice", "instructions": "refund", "criteria": {"approve": "item broken", "deny": "late"}},
    }})["answers"]
    alone = jev.score(state, ["late", "broken", "the"], instructions="how late")
    assert both["a"]["score"] == pytest.approx(alone["score"], abs=1e-5)
    single = jev.choice(state, {"approve": "item broken", "deny": "late"}, instructions="refund")
    assert both["b"]["probabilities"] == pytest.approx(single["probabilities"], abs=1e-5)
