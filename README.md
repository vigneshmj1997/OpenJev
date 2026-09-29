# OpenJev

An encoder-only decision model. You give it some state and a question. It
returns calibrated probabilities. There is no text generation, so it cannot
hallucinate an answer that isn't one of your options.

There are four question types:

| Type | Asks | Returns |
|---|---|---|
| `noul` | Is the statement in the state true? | P(true) |
| `choice` | Which **one** of these options? | a distribution over the options (sums to 1) |
| `multi` | Which of these options apply (any number)? | an independent P for each option |
| `score` | Where does it fall on this low-to-high scale (2 to 10 levels)? | a distribution over the levels and the expected level |

```python
from openjev import OpenJev

jev = OpenJev.load()  # default model: "openjev"

jev.noul("Tracking says the package was delivered, but the customer has a photo of an empty porch.")
# {"type": "noul", "noul": 0.71}

jev.choice("Package never arrived, tracking says lost",
           ["approve refund", "deny refund", "escalate to human"])
# {"type": "choice", "choice": "approve refund",
#  "probabilities": {"approve refund": 0.79, "deny refund": 0.15, "escalate to human": 0.06},
#  "confidence": 0.37}

jev.multi("Arrived two weeks late and the box was crushed",
          ["late", "damaged", "wrong item"])
# {"type": "multi", "selected": ["late", "damaged"],
#  "probabilities": {"late": 0.94, "damaged": 0.88, "wrong item": 0.03}}

jev.score("Help! My payouts have been failing for 3 days.",
          ["Calm", "Frustrated", "Very angry"], instructions="How frustrated is the customer?")
# {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
#  "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05}, "confidence": 0.82}
```

Every shorthand also takes `instructions` (the question to ask), and `choice`
and `multi` accept `{option: description}` in place of a list.
`jev.predict(state, options)` is kept as a shorthand for the choice
probabilities.

### Jev request format

`jev.run(request)` answers one `state` against several named questions in one
batch, in the same shape as the Jev API:

```python
jev.run({
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations"}},
    },
})
# {"model": "openjev", "answers": {"is_urgent": {...}, "department": {...}},
#  "usage": {"input_tokens": 118, "output_tokens": 0}, "elapsed": 31}
```

[docs/examples.md](docs/examples.md) has an example of every question type with
its output.

### Hugging Face pipeline

`OpenJevPipeline` is a `transformers.Pipeline`, so it works like any other
Hugging Face pipeline: pass one input or many, and it handles batching and the
device for you. Each input is one question with its own `state`, in the same
format as a line of training data (see [Data](#data)).

```python
from openjev import OpenJev, OpenJevPipeline

jev = OpenJev.load()  # or a trained checkpoint: OpenJev.load("checkpoints/my-run")
pipe = OpenJevPipeline(model=jev.model, tokenizer=jev.tokenizer, device=jev.device)

# One question in, one answer out.
pipe({"type": "noul", "state": "Help! My payouts have been failing for 3 days.",
      "instructions": "Does this convey urgency?"})
# {"type": "noul", "noul": 0.95}

# A list in, a list out, batched.
pipe([
    {"type": "choice", "state": "Card was charged twice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations"}},
    {"type": "score", "state": "Card was charged twice", "instructions": "How frustrated is the customer?",
     "criteria": ["Calm", "Frustrated", "Very angry"]},
], batch_size=16)
```

A generator is streamed: answers come back one at a time as each batch
finishes, and the input is never loaded into memory all at once. This suits
large files:

```python
import json

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

for result in pipe(read_jsonl("data/eval.jsonl"), batch_size=16):
    print(result)
```

| Argument | Where | What it does |
|---|---|---|
| `device` | constructor | `"cpu"`, `"cuda"`, `"mps"` or a `torch.device` |
| `batch_size` | constructor or call | questions per forward pass (default 1) |
| `threshold` | constructor or call | `multi`: select options with P >= this (default 0.5) |

Values given in the call override the constructor's. `pipe.save_pretrained(dir)`
writes a normal OpenJev checkpoint, which `OpenJev.load(dir)` or
`OpenJevModel.from_pretrained(dir)` can load again.

Use `jev.run` for a full request with several named questions; the pipeline
answers questions one by one.

`transformers.pipeline("openjev", ...)` does **not** work yet. That factory only
loads models stored in Hugging Face's own model and config format, and
`OpenJevModel` is a plain PyTorch module. Create `OpenJevPipeline` directly as
shown above.

## How it works

- Each `(state, option)` pair goes through a bidirectional encoder. A
  `Linear(hidden, 1)` head turns each pair into one score. The heads never see
  the number of options, so options can differ on every request.
- **choice:** the scores for one example are softmaxed together.
- **noul:** a two-option choice between the fixed descriptions
  "Yes, this is true." and "No, this is not true.", or between the `true` and
  `false` criteria when a question gives them.
- **score:** a choice over the levels, each read as "Level i of n: ...". The
  answer's `score` is the expected level index (0 = lowest), so it can fall
  between levels.
- `instructions` go in front of every option, and `criteria` descriptions after
  its name, so the encoder reads `state` against `instructions + option: description`.
  Objects and arrays in `state` or `instructions` are encoded as JSON.
- **multi:** a separate `Linear(hidden, 1)` head, with a sigmoid on each score.
  It needs its own head because a softmax ignores a constant shift, so the
  choice head never learns an absolute level, and a sigmoid needs one.
- **Stage 1:** train with soft cross-entropy (noul, choice, score) and binary
  cross-entropy per option (multi). The types can be mixed in one file and one
  batch.
- **Stage 2:** after every epoch, fit one temperature per type on a
  calibration split. Then measure ECE on the eval split. The checkpoint with
  the **lowest ECE** is kept, because the product is the probability, not the
  argmax.

## Models

`--model` accepts any of the following:

| Value | What it loads |
|---|---|
| `openjev` *(default)* | `answerdotai/ModernBERT-base`: 8k context, RoPE |
| `openjev-mini` | Google BERT-mini (`google/bert_uncased_L-4_H-256_A-4`, ~11M params) |
| `openjev-large` | `answerdotai/ModernBERT-large` |
| any HF encoder id | e.g. `microsoft/deberta-v3-base`, `bert-base-uncased` |
| a directory | a trained OpenJev checkpoint |

A preset name automatically uses `checkpoints/<name>` once you have trained it.

## Data

JSONL, one question per line. `type` is `noul`, `choice`, `multi` or `score`
and defaults to `choice`. Labels can be indices or option text. Any line can
also carry `instructions` and `criteria` as in the Jev request format.

```json
{"state": "Item arrived broken", "options": ["approve refund", "deny refund"], "label": "approve refund"}
{"state": "Arrived late but intact", "options": ["approve", "deny", "escalate"], "probs": [0.5, 0.2, 0.3]}
{"type": "noul", "state": "Claim: the item arrived broken. Photo shows a cracked screen.", "label": true}
{"type": "noul", "state": "...", "prob": 0.8}
{"type": "multi", "state": "Late and crushed", "options": ["late", "damaged", "wrong item"], "labels": ["late", "damaged"]}
{"type": "multi", "state": "...", "options": ["late", "damaged"], "probs": [0.9, 0.4]}
{"type": "score", "state": "Help! Payouts failing for 3 days.", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"], "label": "Frustrated"}
{"type": "choice", "state": "...", "instructions": "Which team?", "criteria": {"billing": "Payments, refunds", "technical": "Bugs, outages"}, "label": "billing"}
```

- **choice** `probs` are normalized to sum to 1.
- **noul** has no `options`. Put the statement to judge in the state.
- **multi** `labels` may be empty (nothing applies), and its `probs` are
  independent, each in [0, 1].
- **score** `criteria` lists 2 to 10 levels, low to high. Its `label` is a level
  index or level text, or give `probs` over the levels.

`data/` holds a toy dataset with 11 decision categories: refunds, loans, content
moderation, IT ticket routing, insurance claims, nurse-line triage, hiring,
email triage, flight disruption, card fraud and review sentiment. Each category
has its own rules, its own options (2 to 5) and its own facts. The state holds
only the facts of the situation, with no instructions, rules or question. The
model learns each category's rules from the labelled examples. Labels are
balanced within each category.
Regenerate or resize it with:

```bash
python scripts/make_toy_data.py --train 5000 --eval 500
python scripts/make_toy_data.py --domains loan fraud insurance   # only some categories
python scripts/make_toy_data.py --include-rules                   # also put the rules in the state
```

## Train

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/openjev.json           # default model
python scripts/train.py --config configs/openjev_mini.json      # small model, fine on CPU
python scripts/train.py --train-file my.jsonl --model microsoft/deberta-v3-base
```

- Command-line flags override values from `--config`. Run
  `python scripts/train.py -h` to see every option.
- If `--eval-file` or `--calibration-file` is missing, that split is carved
  from the training data (`--heldout-fraction`, default 0.1).
- The best checkpoint and `training_summary.json` are written to
  `checkpoints/<model>`.

## Predict

```bash
python scripts/predict.py --state "Customer wants a refund" --options approve deny escalate
python scripts/predict.py --type noul --state "The item arrived broken."
python scripts/predict.py --type multi --state "Late and crushed" --options late damaged "wrong item"
python scripts/predict.py --type score --state "Help! Payouts failing for 3 days." \
    --instructions "How frustrated is the customer?" --options Calm Frustrated "Very angry"
python scripts/predict.py --model openjev-mini --input data/eval.jsonl   # uses each line's "type"
python scripts/predict.py --request request.json                         # a full Jev request
```

## Test

```bash
python -m pytest tests
```

The tests use a tiny random BERT, so they run offline in a few seconds.
