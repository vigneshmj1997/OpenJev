# OpenJev

Open source version of Jev — an encoder-only, non-autoregressive decision model
that returns typed, calibrated probabilities instead of text.

Base model: [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large).

## Why this base model

Most public Jev reproductions bolt a scalar head onto a decoder (Qwen) and strip
the autoregressive loop. That inherits the causal mask, so state tokens cannot
attend to the question that comes after them — the model has to integrate at the
very end. A decision task wants the opposite.

ModernBERT is the encoder that makes the bidirectional version practical:

| | context | bidirectional | positions | attention |
|---|---|---|---|---|
| BERT-large | 512 | yes | learned absolute | vanilla O(n²) |
| DeBERTa-v3-large | 512 | yes | relative | disentangled |
| Qwen + head | long | **no** (causal) | RoPE | Flash |
| **ModernBERT-large** | **8192** | **yes** | **RoPE** | **Flash + local/global** |

DeBERTa-v3 edges it on GLUE but caps at 512 tokens, which rules it out for a
long-state decision model. ModernBERT is the only option with long context *and*
a modern attention stack *and* no causal mask.

## Architecture

The head is inverted relative to a standard BERT classifier. Rather than a
`Linear(hidden, k)` with `k` frozen at training time, options arrive as text and
a single `Linear(hidden, 1)` scores each one:

```
[CLS] state [SEP] instructions [OPT] option_i  ->  encoder  ->  pool  ->  s_i
```

The `k` scores are stacked and softmaxed. `k` appears only in how many rows get
stacked, never in a weight shape — so the label set can change per request, which
a fixed classification head cannot do.

Questions in one call are scored against the same state and in isolation from
each other; a batch can mix a `k=2` noul with a `k=17` choice, because unused
slots in the scatter grid are held at `-inf` and softmax ignores them.

## Install

```bash
pip install -r requirements.txt
```

## Data format

One JSONL line per (state, question, label):

```json
{"state": "Help! My payouts have been failing for 3 days.",
 "question": {"type": "choice",
              "instructions": "Which team should handle this?",
              "criteria": {"billing": "Payments, invoicing, refunds",
                           "technical": "Bugs, outages, integrations",
                           "sales": "Pricing, upgrades, new accounts"}},
 "label": "billing"}
```

`criteria` **descriptions** are the text encoded beside the state; `criteria`
**keys** are labels zipped back on at serialization. Renaming a key cannot change
the numbers; rewriting its description will.

Labels accept several forms: a key string, `true`/`false` for noul, a level index
for score, or `"target"` with an explicit soft distribution
(`{"billing": 0.6, "technical": 0.4}`). Soft targets let a label carry real
uncertainty instead of forcing a one-hot the model must be overconfident to fit.

## Train

```bash
python scripts/train.py --config configs/modernbert_large.json
python scripts/train.py --config configs/modernbert_base.json --epochs 1
```

Two stages, in order:

1. **Fit** the encoder + scalar head on a soft-target cross-entropy. Two
   parameter groups — the pretrained encoder moves at `2e-5`, the fresh head at
   `1e-4`.
2. **Calibrate** a single scalar temperature on a held-out split, everything else
   frozen.

Stage 2 is not optional. What comes out of stage 1 ranks well and is
systematically overconfident; the temperature is what makes the reported
probability usable as a routing signal. It has one parameter, so a few hundred
held-out rows suffice — and because it is monotone it **cannot change the
argmax**, only the confidence. Fit it on data the model never trained on, or you
will calibrate against memorized answers.

Checkpoints land in `output_dir/{best,final}`, alongside `history.json`,
`summary.json`, and `reliability.json` (per-bin confidence vs. accuracy).

## Infer

```python
from openjev import OpenJev

jev = OpenJev("checkpoints/openjev-large/final")
jev.decide(
    state="Help! My payouts have been failing for 3 days.",
    questions={
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                      "criteria": {"true": "Explicitly time-sensitive",
                                   "false": "No urgency expressed"}},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments, invoicing, refunds",
                              "technical": "Bugs, outages, integrations",
                              "sales": "Pricing, upgrades, new accounts"}},
        "anger": {"type": "score", "instructions": "How frustrated is the customer?",
                  "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
)
```

Returns the Jev envelope:

```json
{
  "model": "openjev-final",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.95},
    "team": {"type": "choice", "choice": "billing",
             "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
             "confidence": 0.81},
    "anger": {"type": "score", "score": 1.05,
              "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
              "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
              "confidence": 0.92}
  },
  "usage": {"input_tokens": 296, "output_tokens": 0}
}
```

Every key in the response is copied from the request; the model supplies only the
leaf floats. That is where "cannot produce type errors" comes from — it is
structural, not a training achievement. Note it is *not* a correctness guarantee:
the model can be confidently wrong, it just cannot be malformed.

### Answer shapes

- **noul** — bare probability, no `probabilities` map and no `confidence`: for a
  binary question the probability *is* the certainty (0.5 is maximal uncertainty).
- **choice** — `probabilities` over your criteria keys; `choice` is `argmax` used
  to index your own key list.
- **score** — `score` is the probability-weighted expectation `Σ i·pᵢ`, so `1.05`
  legitimately falls between levels 1 and 2. This is meaningful only because
  levels are *ordered*; there is no sensible average of `billing` and `sales`,
  which is why choice has no equivalent field.

## Metrics

Accuracy alone will hide the failure that matters here, since the product is the
probability rather than the argmax. Training reports:

- **ECE** — average gap between stated confidence and observed accuracy.
- **Brier** — proper scoring rule; minimized only by reporting true beliefs, so
  unlike accuracy it cannot be gamed by confident guessing.
- **reliability.json** — per-bin confidence vs. accuracy.

Checkpoint selection uses ECE, not accuracy.

## Test

```bash
python tests/test_openjev.py      # or: python -m pytest tests/ -v
```

## Notes on fidelity

TypeSafe has not disclosed Jev's architecture — no parameter count, no context
window, no confirmation it is encoder-only. The encoder reading here follows the
credible outside assessment rather than a published spec. `RLCD` in particular is
a named-but-undocumented method; the temperature-scaling stage here is a
well-understood stand-in for its calibration goal, not a reproduction of it.
