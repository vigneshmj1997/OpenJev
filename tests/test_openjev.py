"""Offline tests: a tiny random BERT stands in for the real encoder."""

import json
import math
import os

import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from openjev import OpenJev, OpenJevConfig, OpenJevModel, TrainArgs, train
from openjev.calibration import evaluate_logits, expected_calibration_error, fit_temperature, fit_temperatures
from openjev.config import base_config, find_checkpoint
from openjev.data import CHOICE, MULTI, NOUL, NOUL_OPTIONS, Collator, Example, parse_record
from openjev.model import openjev_loss, probabilities, soft_cross_entropy
from openjev.train import model_inputs

WORDS = "refund approve deny escalate order item broken late yes no maybe the a is to".split()


@pytest.fixture(scope="session")
def tiny_encoder_dir(tmp_path_factory):
    """A tiny random BERT + tokenizer saved like any Hugging Face encoder."""
    directory = tmp_path_factory.mktemp("tiny-bert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocab))
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    torch.manual_seed(0)
    encoder = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, intermediate_size=64,
                                   max_position_embeddings=64))
    encoder.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


def make_model(encoder_dir):
    config = OpenJevConfig(encoder=encoder_dir, max_length=32, dropout=0.0)
    return OpenJevModel(config), BertTokenizerFast.from_pretrained(encoder_dir)


def test_parse_record_label_forms():
    assert parse_record({"state": "c", "options": ["a", "b"], "label": 1}).target == [0.0, 1.0]
    assert parse_record({"state": "c", "options": ["a", "b"], "label": "a"}).target == [1.0, 0.0]
    assert parse_record({"state": "c", "options": ["a", "b"], "probs": [3, 1]}).target == [0.75, 0.25]
    with pytest.raises(ValueError):
        parse_record({"state": "c", "options": ["a", "b"], "label": "z"})
    with pytest.raises(ValueError):
        parse_record({"state": "c", "options": ["a"], "label": 0})


def test_parse_record_noul_and_multi():
    noul = parse_record({"type": "noul", "state": "c", "label": True})
    assert noul.options == NOUL_OPTIONS and noul.target == [1.0, 0.0]
    assert parse_record({"type": "noul", "state": "c", "label": "false"}).target == [0.0, 1.0]
    assert parse_record({"type": "noul", "state": "c", "prob": 0.25}).target == [0.25, 0.75]

    multi = parse_record({"type": "multi", "state": "c", "options": ["a", "b", "c"], "labels": ["a", 2]})
    assert multi.target == [1.0, 0.0, 1.0]  # independent, not normalized
    assert parse_record({"type": "multi", "state": "c", "options": ["a", "b"], "labels": []}).target == [0, 0]
    assert parse_record({"type": "multi", "state": "c", "options": ["a"], "probs": [0.9]}).target == [0.9]
    with pytest.raises(ValueError):
        parse_record({"type": "multi", "state": "c", "options": ["a", "b"], "probs": [1.5, 0]})
    with pytest.raises(ValueError):
        parse_record({"type": "noul", "state": "c", "label": "maybe"})
    with pytest.raises(ValueError):
        parse_record({"type": "rank", "state": "c", "options": ["a", "b"], "label": 0})


def test_ragged_options_padded_with_neg_inf(tiny_encoder_dir):
    model, tokenizer = make_model(tiny_encoder_dir)
    batch = Collator(tokenizer, 32)([
        Example("refund order", ["approve", "deny"], [1.0, 0.0]),
        Example("item broken", ["approve", "deny", "escalate", "maybe"], [0, 0, 1.0, 0]),
    ])
    logits = model(**model_inputs(batch))
    assert logits.shape == (2, 4)
    assert torch.isinf(logits[0, 2:]).all() and torch.isfinite(logits[0, :2]).all()
    assert torch.isfinite(logits[1]).all()
    loss = soft_cross_entropy(logits, batch["targets"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_scores_do_not_depend_on_batch_neighbours(tiny_encoder_dir):
    model, tokenizer = make_model(tiny_encoder_dir)
    model.eval()
    collate = Collator(tokenizer, 32)
    ex = Example("refund order", ["approve", "deny"], [1.0, 0.0])
    alone = model(**model_inputs(collate([ex])))
    paired = model(**model_inputs(collate([ex, Example("late item", ["yes", "no", "maybe"], [1, 0, 0])])))
    assert torch.allclose(alone[0], paired[0, :2], atol=1e-5)


def test_mixed_types_in_one_batch(tiny_encoder_dir):
    model, tokenizer = make_model(tiny_encoder_dir)
    batch = Collator(tokenizer, 32)([
        Example("item broken", NOUL_OPTIONS, [1.0, 0.0], "noul"),
        Example("refund order", ["approve", "deny", "escalate"], [0, 1.0, 0], "choice"),
        Example("late item broken", ["late", "broken", "the", "a"], [1.0, 1.0, 0, 0], "multi"),
    ])
    assert batch["kinds"].tolist() == [NOUL, CHOICE, MULTI]
    logits = model(**model_inputs(batch))
    assert logits.shape == (3, 4)

    probs = probabilities(logits.detach(), batch["kinds"], model.temperature)
    assert probs[0].sum() == pytest.approx(1.0) and (probs[0, 2:] == 0).all()
    assert probs[1].sum() == pytest.approx(1.0) and probs[1, 3] == 0
    assert torch.allclose(probs[2], torch.sigmoid(logits[2].detach()))  # independent, need not sum to 1

    loss = openjev_loss(logits, batch["targets"], batch["kinds"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert model.multi_head.weight.grad.abs().sum() > 0


def test_multi_head_scores_only_multi_rows(tiny_encoder_dir):
    model, tokenizer = make_model(tiny_encoder_dir)
    model.eval()
    collate = Collator(tokenizer, 32)
    as_choice = model(**model_inputs(collate([Example("refund", ["approve", "deny"], [1, 0], "choice")])))
    as_multi = model(**model_inputs(collate([Example("refund", ["approve", "deny"], [1, 0], "multi")])))
    with torch.no_grad():
        model.multi_head.weight.zero_()
        model.multi_head.bias.fill_(3.0)
    assert torch.allclose(model(**model_inputs(collate([Example("refund", ["approve", "deny"], [1, 0], "multi")]))),
                          torch.full((1, 2), 3.0))
    assert not torch.allclose(as_choice, as_multi)


def test_ece_and_temperature():
    labels = torch.tensor([0, 1, 0, 1])
    perfect = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, labels) == pytest.approx(0.0)

    # Overconfident logits: true temperature is 3, fitting should recover it.
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 3)
    targets = torch.nn.functional.one_hot(torch.distributions.Categorical(logits=true_logits).sample(), 3).float()
    t = fit_temperature(true_logits * 3, targets)
    assert t == pytest.approx(3.0, rel=0.15)
    fitted = evaluate_logits(true_logits * 3, targets, temperatures=(1.0, t, 1.0))
    assert fitted["ece"] < evaluate_logits(true_logits * 3, targets)["ece"]
    assert fitted["choice_temperature"] == t


def test_multi_temperature_is_fitted_separately():
    # Overconfident multi logits (true T = 2.5) next to already-calibrated choice logits.
    torch.manual_seed(0)
    multi_logits = torch.randn(3000, 3)
    multi_targets = torch.bernoulli(torch.sigmoid(multi_logits))
    choice_logits = torch.randn(3000, 3)
    choice_targets = torch.nn.functional.one_hot(
        torch.distributions.Categorical(logits=choice_logits).sample(), 3).float()
    logits = torch.cat([multi_logits * 2.5, choice_logits])
    targets = torch.cat([multi_targets, choice_targets])
    kinds = torch.tensor([MULTI] * 3000 + [CHOICE] * 3000)

    temps = fit_temperatures(logits, targets, kinds)
    assert temps[MULTI] == pytest.approx(2.5, rel=0.15)
    assert temps[CHOICE] == pytest.approx(1.0, rel=0.15)
    assert temps[NOUL] == 1.0  # absent type is left alone
    metrics = evaluate_logits(logits, targets, kinds, temps)
    assert metrics["multi_ece"] < evaluate_logits(logits, targets, kinds)["multi_ece"]
    assert "noul_ece" not in metrics


def test_presets_and_default():
    assert base_config("openjev").encoder == "answerdotai/ModernBERT-base"
    assert base_config("openjev-mini").encoder == "google/bert_uncased_L-4_H-256_A-4"
    assert base_config("some/encoder", max_length=128).max_length == 128
    assert find_checkpoint("definitely/not-a-checkpoint") is None


def test_end_to_end_train_save_load_predict(tiny_encoder_dir, tmp_path):
    rows = []
    for i in range(30):
        yes = i % 2 == 0
        state = "item broken refund" if yes else "the order is late"
        rows.append({"state": state, "options": ["approve", "deny"], "label": 0 if yes else 1})
        rows.append({"type": "noul", "state": state, "label": yes})
        rows.append({"type": "multi", "state": state, "options": ["broken", "late", "maybe"],
                     "labels": ["broken"] if yes else ["late"]})
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "ckpt"

    summary = train(TrainArgs(train_file=str(train_file), model=tiny_encoder_dir, output_dir=str(out),
                              epochs=2, batch_size=4, learning_rate=1e-3, max_length=32,
                              device="cpu", log_every=1000))
    assert len(summary["history"]) == 2
    assert os.path.isfile(out / "openjev_config.json")
    assert (out / "training_summary.json").exists()

    jev = OpenJev.load(str(out), device="cpu")
    for i, kind in enumerate(("noul", "choice", "multi")):
        assert jev.model.temperature[i].item() == pytest.approx(summary["best"][f"{kind}_temperature"])

    probs = jev.predict("item broken refund", ["approve", "deny", "escalate"])
    assert set(probs) == {"approve", "deny", "escalate"}
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-5)

    noul = jev.noul("item broken refund")
    assert noul["type"] == "noul" and 0.0 <= noul["noul"] <= 1.0

    choice = jev.choice("item broken refund", ["approve", "deny"])
    assert choice["choice"] in ("approve", "deny") and 0.0 <= choice["confidence"] <= 1.0

    multi = jev.multi("item broken refund", ["broken", "late", "maybe"], threshold=0.0)
    assert multi["selected"] == ["broken", "late", "maybe"]  # threshold 0 selects everything
    assert set(multi["probabilities"]) == {"broken", "late", "maybe"}
