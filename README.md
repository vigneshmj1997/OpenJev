# OpenJev

An encoder-only decision model. You give it a state and a question; it returns
calibrated probabilities over your options. It generates no text, so it can't
answer with something that isn't one of your options.

| Type | Asks | Returns |
|---|---|---|
| `noul` | Is it true? | P(yes) |
| `choice` | Which **one** option? | probabilities that sum to 1 |
| `multi` | Which options apply? | an independent P per option |
| `score` | Where on a low-to-high scale (2 to 10 levels)? | probabilities per level and the expected level |

## Hugging Face pipeline

```python
from openjev import OpenJev, OpenJevPipeline

jev = OpenJev.load()
pipe = OpenJevPipeline(model=jev.model, tokenizer=jev.tokenizer, device=jev.device)

pipe({"type": "noul", "state": "Payouts failing for 3 days.", "instructions": "Is this urgent?"})
pipe([question_1, question_2], batch_size=16)  # list in, list out
pipe(read_jsonl("questions.jsonl"))             # generator in, answers streamed
```

- Each input is one question: `type`, `state`, `instructions`, and `options` or `criteria`.
- Options: `device`, `batch_size`, `threshold` (for `multi`). Values passed in the call override the constructor's.
- `pipe.save_pretrained(dir)` writes a normal OpenJev checkpoint.
- For several named questions about one state, use `jev.run(request)`; see [docs/examples.md](docs/examples.md).
- `transformers.pipeline("openjev")` doesn't work yet; create `OpenJevPipeline` directly.

## Notes

- **How it works:** each (state, option) pair goes through ModernBERT, and a linear head gives it one score. Scores are softmaxed (`noul`, `choice`, `score`) or passed through a sigmoid (`multi`), then calibrated.
- **Models:** `openjev` (ModernBERT-base, default), `openjev-large`, any Hugging Face encoder id, or a checkpoint directory.
- **Predict:** `python scripts/predict.py --help`.
- **Test:** `python -m pytest tests` (runs offline in a few seconds).
