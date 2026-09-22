"""OpenJev: a ModernBERT cross-encoder with an open-label scalar decision head.

The head is inverted relative to a standard BERT classifier. Rather than a
`Linear(hidden, k)` with k frozen at training time, options arrive as text in the
input and a single `Linear(hidden, 1)` scores each one:

    [CLS] state [SEP] instructions [SEP] option_i  ->  encoder  ->  pool  ->  s_i

The k scores are stacked and softmaxed. k therefore appears only in how many rows
we stack, never in a weight shape, so the label set can change per request.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-large"


@dataclass
class OpenJevOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor          # (num_questions, max_k) padded with -inf
    log_probs: torch.Tensor       # (num_questions, max_k) padded with -inf
    temperature: torch.Tensor


class OpenJevModel(nn.Module):
    """Encoder + scalar decision head + a learned calibration temperature."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        pooling: str = "cls",
        head_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        hidden = self.config.hidden_size
        self.pooling = pooling

        # The entire open-label trick: ONE output unit, reused for every option.
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )

        # Calibration temperature, stored in log space to keep it positive.
        # Fitted on held-out data after the main run; frozen during training.
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)

        self._init_head()

    def _init_head(self) -> None:
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden_states[:, 0]
        # Mean pooling over real tokens only; padding must not dilute the vector.
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def score_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score a flat batch of (state, option) pairs. Returns (num_pairs,)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.head(pooled).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,        # (num_pairs, seq_len) — options flattened
        attention_mask: torch.Tensor,
        group_index: torch.Tensor,      # (num_pairs,) which question each pair belongs to
        option_index: torch.Tensor,     # (num_pairs,) position within that question
        num_questions: int,
        max_k: int,
        labels: torch.Tensor | None = None,        # (num_questions,) gold option index
        target_dist: torch.Tensor | None = None,   # (num_questions, max_k) soft targets
        apply_temperature: bool = False,
    ) -> OpenJevOutput:
        scores = self.score_candidates(input_ids, attention_mask)

        # Scatter the flat scores back into a (num_questions, max_k) grid. Slots
        # for options a question doesn't have stay at -inf so softmax ignores them
        # — this is what lets one batch hold a k=2 noul and a k=17 choice.
        logits = scores.new_full((num_questions, max_k), float("-inf"))
        logits[group_index, option_index] = scores

        if apply_temperature:
            logits = logits / self.temperature

        log_probs = F.log_softmax(logits, dim=-1)

        loss = None
        if target_dist is not None:
            # Soft targets: KL(target || pred). Lets a label carry genuine
            # uncertainty ("60% billing, 40% technical") instead of forcing a
            # one-hot the model would have to be overconfident to fit.
            valid = torch.isfinite(logits)
            safe_log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
            loss = -(target_dist * safe_log_probs).sum(dim=-1).mean()
        elif labels is not None:
            loss = F.nll_loss(log_probs, labels)

        return OpenJevOutput(
            loss=loss,
            logits=logits,
            log_probs=log_probs,
            temperature=self.temperature.detach(),
        )

    @torch.no_grad()
    def set_temperature(self, value: float) -> None:
        self.log_temperature.data = torch.tensor([value], device=self.log_temperature.device).log()
