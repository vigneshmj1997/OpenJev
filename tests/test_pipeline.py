import os
import math

import torch

from openjev import OpenJev, OpenJevModel, OpenJevPipeline
from openjev.config import OpenJevConfig
from openjev.model import load_tokenizer
from test_openjev import tiny_encoder_dir  # noqa: F401

QUESTIONS = [
    {"type": "noul", "state": "item broken refund"},
    {"state": "item broken refund", "options": ["approve", "deny", "escalate"]},
    {"type": "multi", "state": "the order is late", "options": ["broken", "late"]},
    {"state": "the order is late", "options": ["approve", "deny"]},
]


def test_pipeline_matches_openjev(tiny_encoder_dir, tmp_path):  # noqa: F811
    config = OpenJevConfig(encoder=tiny_encoder_dir, max_length=32)
    model, tokenizer = OpenJevModel(config), load_tokenizer(config)
    expected = OpenJev(model, tokenizer, torch.device("cpu")).ask_batch(QUESTIONS, threshold=0.3)

    pipe = OpenJevPipeline(model=model, tokenizer=tokenizer, device="cpu")
    single = pipe(QUESTIONS[1], threshold=0.3)
    batched = pipe(QUESTIONS, batch_size=3, threshold=0.3)
    unbatched = pipe(QUESTIONS, threshold=0.3)
    streamed = list(pipe((q for q in QUESTIONS), batch_size=2, threshold=0.3))

    for results in (batched, unbatched, streamed):
        assert len(results) == len(QUESTIONS)
        for got, want in zip(results, expected):
            assert got.keys() == want.keys()
            for key in ("noul", "choice", "selected", "probabilities", "confidence"):
                if key in want and isinstance(want[key], dict):
                    for o in want[key]:
                        assert math.isclose(got[key][o], want[key][o], abs_tol=1e-5)
                elif key in want and isinstance(want[key], float):
                    assert math.isclose(got[key], want[key], abs_tol=1e-5)
                elif key in want:
                    assert got[key] == want[key]
    assert single["choice"] == expected[1]["choice"]

    pipe.save_pretrained(str(tmp_path / "ckpt"))
    assert os.path.isfile(tmp_path / "ckpt" / "openjev_config.json")
    OpenJev.load(str(tmp_path / "ckpt"), device="cpu")
    assert "OpenJevPipeline" in repr(pipe)
