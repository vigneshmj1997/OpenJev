"""OpenJev: an encoder that scores each (state, option) pair.

The head is `Linear(hidden, 1)` applied per option, not `Linear(hidden, k)`,
so the number and wording of options can change on every request. Scores for
one example are gathered into a row padded with -inf. What happens to a row
depends on its question type:

  * noul, choice, score: the row is softmaxed, so exactly one option wins.
  * multi: each score goes through its own sigmoid, so any number can be true.
"""

from __future__ import annotations

import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertConfig, BertModel

from .config import OpenJevConfig
from .data import CHOICE, MULTI, TYPES

HEAD_NAME = "head.pt"
ENCODER_DIR = "encoder"


class OpenJevModel(nn.Module):
    def __init__(self, config: OpenJevConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(config.encoder)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        # noul and choice. A softmax ignores a constant shift, so this head
        # never learns an absolute level; that is fine for picking one option.
        self.head = nn.Linear(hidden, 1)
        # multi. A sigmoid needs an absolute level (0.5 must mean something),
        # so it gets its own head instead of reusing the shift-free one.
        self.multi_head = nn.Linear(hidden, 1)
        # One temperature per question type, indexed like TYPES. Fitted after
        # training on held-out data; 1.0 means uncalibrated.
        self.register_buffer("temperature", torch.ones(len(TYPES)))

    @property
    def device(self) -> torch.device:
        # transformers.Pipeline reads model.device, which plain nn.Modules lack.
        return self.temperature.device

    def score_candidates(self, input_ids, attention_mask, token_type_ids=None,
                         multi: torch.Tensor | None = None) -> torch.Tensor:
        """One scalar score per (state, option) pair. Shape (N,).

        `multi` is a bool per pair; those pairs are scored by `multi_head`.
        """
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        if self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = hidden[:, 0]
        pooled = self.dropout(pooled)
        scores = self.head(pooled).squeeze(-1)
        if multi is not None and multi.any():
            scores = torch.where(multi, self.multi_head(pooled).squeeze(-1), scores)
        return scores

    def forward(self, input_ids, attention_mask, group_index, option_index, num_groups,
                max_options, token_type_ids=None, kinds=None) -> torch.Tensor:
        """Raw logits of shape (num_groups, max_options); padding slots are -inf.

        `kinds` holds each group's index into TYPES; omitted means all choice.
        """
        multi = None if kinds is None else (kinds == MULTI)[group_index]
        scores = self.score_candidates(input_ids, attention_mask, token_type_ids, multi).float()
        logits = scores.new_full((num_groups, max_options), float("-inf"))
        return logits.index_put((group_index, option_index), scores)

    # ---- persistence -------------------------------------------------------

    def save_pretrained(self, directory: str, tokenizer=None) -> None:
        os.makedirs(directory, exist_ok=True)
        self.config.save(directory)
        self.encoder.save_pretrained(os.path.join(directory, ENCODER_DIR))
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(directory, ENCODER_DIR))
        torch.save(
            {"head": self.head.state_dict(), "multi_head": self.multi_head.state_dict(),
             "temperature": self.temperature.cpu()},
            os.path.join(directory, HEAD_NAME),
        )

    @classmethod
    def from_pretrained(cls, directory: str) -> "OpenJevModel":
        config = OpenJevConfig.load(directory)
        encoder = AutoModel.from_pretrained(os.path.join(directory, ENCODER_DIR))
        model = cls(config, encoder=encoder)
        state = torch.load(os.path.join(directory, HEAD_NAME), map_location="cpu")
        model.head.load_state_dict(state["head"])
        if "multi_head" in state:  # older checkpoints were choice-only
            model.multi_head.load_state_dict(state["multi_head"])
        model.temperature.copy_(_per_type(state["temperature"]))
        return model


def _per_type(temperature: torch.Tensor) -> torch.Tensor:
    """Temperatures from older checkpoints, stretched to one per current type.

    Very old checkpoints stored one shared value; ones from before `score`
    existed stored three. Types a checkpoint doesn't know reuse the choice
    temperature, the closest calibrated softmax.
    """
    temperature = temperature.flatten().float()
    if temperature.numel() == 1:
        return temperature.expand(len(TYPES))
    missing = len(TYPES) - temperature.numel()
    return torch.cat([temperature, temperature[CHOICE].repeat(missing)])


def load_tokenizer(config: OpenJevConfig, checkpoint_dir: str | None = None):
    source = os.path.join(checkpoint_dir, ENCODER_DIR) if checkpoint_dir else config.encoder
    return AutoTokenizer.from_pretrained(source)


def masked_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """log_softmax that returns 0 (not -inf) at padded slots."""
    mask = torch.isfinite(logits)
    return torch.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a target distribution (KL up to a constant)."""
    return -(targets * masked_log_softmax(logits)).sum(-1).mean()


def openjev_loss(logits: torch.Tensor, targets: torch.Tensor,
                 kinds: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over examples of each row's loss for its question type.

    noul/choice/score rows: soft cross-entropy over the row.
    multi rows: binary cross-entropy per option, averaged over real options.
    """
    ce = -(targets * masked_log_softmax(logits)).sum(-1)
    if kinds is None or not (kinds == MULTI).any():
        return ce.mean()
    real = torch.isfinite(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits.masked_fill(~real, 0.0), targets, reduction="none")
    bce = (bce * real).sum(-1) / real.sum(-1)
    return torch.where(kinds == MULTI, bce, ce).mean()


def probabilities(logits: torch.Tensor, kinds: torch.Tensor | None = None,
                  temperature: torch.Tensor | None = None) -> torch.Tensor:
    """Calibrated probabilities, same shape as `logits`; padding slots are 0.

    noul/choice/score rows sum to 1. multi rows hold one independent P(selected) per option.
    """
    if kinds is None:
        kinds = torch.full((logits.shape[0],), TYPES.index("choice"), device=logits.device)
    if temperature is not None:
        logits = logits / temperature.to(logits.device)[kinds].unsqueeze(-1)
    real = torch.isfinite(logits)
    softmax = masked_log_softmax(logits).exp() * real
    sigmoid = torch.sigmoid(logits) * real
    return torch.where((kinds == MULTI).unsqueeze(-1), sigmoid, softmax)
