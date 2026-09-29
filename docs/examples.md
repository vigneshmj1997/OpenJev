# Request examples

Send any of these requests to a loaded model with `jev.run(request)`, or save
it to a file and run `python scripts/predict.py --request request.json`.
`notebook/notebook.ipynb` runs every example below.

```python
from openjev import OpenJev

jev = OpenJev.load()
jev.run({"state": "...", "model": "jev-latest", "questions": {...}})
```

`model` in the request is accepted but not checked: the model you loaded
answers. The response also carries `elapsed` (milliseconds), and
`usage.output_tokens` is always 0 because OpenJev generates no text.

A request has one `state` (the text being judged) and any number of named
`questions` about it. Each question has:

| Field | Meaning |
|---|---|
| `type` | `noul`, `choice` or `score` |
| `instructions` | The question to ask about the state. A string, object or array. |
| `criteria` | What each answer means (see each type below). Optional for `noul`, required for `choice` and `score`. |

`state` can also be a string, an object or an array. `questions` are answered
independently of each other.

| Type | `criteria` | Limits | Answer fields |
|---|---|---|---|
| `noul` | optional `{"true": ..., "false": ...}` | | `noul`: P(yes), 0 to 1 |
| `choice` | `{option: description}` | up to 255 options | `choice`, `probabilities`, `confidence` |
| `score` | list of levels, low to high | 2 to 10 levels | `score`, `legend`, `probabilities`, `confidence` |

Every example below is followed by its output. The numbers are illustrative,
not from a trained model, but each output follows the rules of the response
format:

- `answers` is keyed by the question names from the request.
- `choice` and `score` probabilities sum to 1. `choice` keys them by option name;
  `score` keys them by level index, starting at `"0"`, and `legend` maps each index
  back to its level.
- `score` is the probability-weighted average level index, so it can fall between
  levels: `0.95` on level 1 and `0.05` on level 2 gives `1.05`.
- `confidence` is OpenJev's normalized entropy: 0 when every option is equally
  likely, 1 when one option has all the probability.

## `noul`: true or false

Answers a yes/no question about the state. `criteria` is optional; when given,
it describes what `true` and `false` mean.

```json
{
  "type": "noul",
  "instructions": "Does this message convey urgency?",
  "criteria": {
    "true": "Explicitly needs immediate attention",
    "false": "No urgency expressed"
  }
}
```

Answer:

```json
{"type": "noul", "noul": 0.95}
```

## `choice`: pick exactly one

`criteria` maps each option name to a description of when it applies. The
option names are the possible answers.

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
      "confidence": 0.67
    }
  },
  "usage": {"input_tokens": 318, "output_tokens": 34}
}
```

## `score`: a point on an ordered scale

`criteria` is a list of levels, ordered from lowest to highest.

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
      "confidence": 0.82
    }
  },
  "usage": {"input_tokens": 304, "output_tokens": 18}
}
```

## All three in one request

Questions are keyed by name, so one request can ask several things about the
same state.

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Does this message convey urgency?",
      "criteria": {
        "true": "Explicitly needs immediate attention",
        "false": "No urgency expressed"
      }
    },
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "Payments, invoicing, refunds",
        "technical": "Bugs, outages, integrations",
        "sales": "Pricing, upgrades, new accounts"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "urgent": {"type": "noul", "noul": 0.95},
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
      "confidence": 0.67
    },
    "frustration": {
      "type": "score",
      "score": 1.05,
      "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
      "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
      "confidence": 0.82
    }
  },
  "usage": {"input_tokens": 402, "output_tokens": 60}
}
```

## More examples

### `noul` without `criteria`

`criteria` can be left out when the question is clear on its own.

```json
{
  "state": "I was charged twice for my March invoice.",
  "model": "jev-latest",
  "questions": {
    "mentions_refund": {
      "type": "noul",
      "instructions": "Is the customer asking for money back?"
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "mentions_refund": {"type": "noul", "noul": 0.87}
  },
  "usage": {"input_tokens": 188, "output_tokens": 12}
}
```

### Structured `state`

`state` can be an object instead of plain text, such as a support ticket with
its metadata.

```json
{
  "state": {
    "subject": "Cannot log in after password reset",
    "body": "I reset my password an hour ago and the new one is rejected. I have a demo with a client at 3pm.",
    "plan": "enterprise",
    "previous_tickets": 4
  },
  "model": "jev-latest",
  "questions": {
    "escalate": {
      "type": "noul",
      "instructions": "Should this ticket skip the queue?",
      "criteria": {
        "true": "Enterprise customer blocked with a deadline today",
        "false": "Can wait for normal handling"
      }
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "escalate": {"type": "noul", "noul": 0.93}
  },
  "usage": {"input_tokens": 276, "output_tokens": 12}
}
```

### Structured `instructions`

When a question needs extra data, put the question and the data in an object
and refer to the data by field name.

```json
{
  "state": "Hi, the blender I ordered 40 days ago stopped working. Can I return it?",
  "model": "jev-latest",
  "questions": {
    "eligible": {
      "type": "noul",
      "instructions": {
        "question": "Is this return allowed under the policy?",
        "policy": "Items can be returned within 30 days of delivery. Defective items can be returned within 1 year."
      },
      "criteria": {
        "true": "The request fits the policy",
        "false": "The request is outside the policy"
      }
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "eligible": {"type": "noul", "noul": 0.84}
  },
  "usage": {"input_tokens": 254, "output_tokens": 12}
}
```

### `choice` with more options

A `choice` can have up to 255 options. Each description should say when that
option applies, not just repeat its name.

```json
{
  "state": "The app crashes every time I open the camera on my Pixel 8.",
  "model": "jev-latest",
  "questions": {
    "component": {
      "type": "choice",
      "instructions": "Which part of the product is affected?",
      "criteria": {
        "auth": "Login, signup, passwords, two-factor codes",
        "payments": "Cards, charges, payouts, invoices",
        "media": "Camera, photo upload, video playback",
        "notifications": "Push notifications, emails, SMS",
        "sync": "Data missing or out of date across devices",
        "other": "None of the above"
      }
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "component": {
      "type": "choice",
      "choice": "media",
      "probabilities": {"auth": 0.01, "payments": 0.0, "media": 0.93, "notifications": 0.01, "sync": 0.02, "other": 0.03},
      "confidence": 0.81
    }
  },
  "usage": {"input_tokens": 331, "output_tokens": 48}
}
```

### `score` with object levels

`score` levels can be objects as well as strings, which lets each level carry
a name and a description. List them from low to high.

```json
{
  "state": "Checkout has been down for all EU customers for 20 minutes.",
  "model": "jev-latest",
  "questions": {
    "severity": {
      "type": "score",
      "instructions": "How severe is this incident?",
      "criteria": [
        {"level": "SEV4", "description": "Cosmetic issue, no user impact"},
        {"level": "SEV3", "description": "Minor feature broken, workaround exists"},
        {"level": "SEV2", "description": "Major feature broken for some users"},
        {"level": "SEV1", "description": "Core flow down for many users"}
      ]
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "severity": {
      "type": "score",
      "score": 2.76,
      "legend": {
        "0": {"level": "SEV4", "description": "Cosmetic issue, no user impact"},
        "1": {"level": "SEV3", "description": "Minor feature broken, workaround exists"},
        "2": {"level": "SEV2", "description": "Major feature broken for some users"},
        "3": {"level": "SEV1", "description": "Core flow down for many users"}
      },
      "probabilities": {"0": 0.0, "1": 0.03, "2": 0.18, "3": 0.79},
      "confidence": 0.57
    }
  },
  "usage": {"input_tokens": 297, "output_tokens": 30}
}
```

### Several questions over a review

One request can run a mix of question types against the same state.

```json
{
  "state": "Delivery was quick but the headphones stopped charging after two days. Support never replied.",
  "model": "jev-latest",
  "questions": {
    "satisfaction": {
      "type": "score",
      "instructions": "How satisfied is the customer overall?",
      "criteria": ["Very unhappy", "Unhappy", "Neutral", "Happy", "Very happy"]
    },
    "topic": {
      "type": "choice",
      "instructions": "What is the main complaint about?",
      "criteria": {
        "shipping": "Speed or condition of delivery",
        "quality": "The product broke or does not work",
        "support": "Slow or unhelpful customer service",
        "price": "Cost or value for money"
      }
    },
    "needs_reply": {
      "type": "noul",
      "instructions": "Should someone from support contact this customer?"
    }
  }
}
```

Output:

```json
{
  "model": "jev-latest",
  "answers": {
    "satisfaction": {
      "type": "score",
      "score": 0.83,
      "legend": {"0": "Very unhappy", "1": "Unhappy", "2": "Neutral", "3": "Happy", "4": "Very happy"},
      "probabilities": {"0": 0.34, "1": 0.52, "2": 0.11, "3": 0.03, "4": 0.0},
      "confidence": 0.34
    },
    "topic": {
      "type": "choice",
      "choice": "quality",
      "probabilities": {"shipping": 0.02, "quality": 0.71, "support": 0.25, "price": 0.02},
      "confidence": 0.46
    },
    "needs_reply": {"type": "noul", "noul": 0.91}
  },
  "usage": {"input_tokens": 389, "output_tokens": 72}
}
```

