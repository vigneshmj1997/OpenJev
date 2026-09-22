"""Calibration metrics and temperature fitting.

A fine-tuned classifier produces a well-shaped softmax that is systematically
overconfident. Since the whole product here is the probability — not the argmax —
calibration is the metric that matters, and accuracy alone will hide the problem.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def expected_calibration_error(
    probs: torch.Tensor,     # (n, k)
    labels: torch.Tensor,    # (n,)
    n_bins: int = 15,
) -> float:
    """ECE: average |confidence - accuracy| over equal-width confidence bins.

    If the model says 0.9 on a bucket of inputs, ~90% of them should be right.
    ECE measures how far off that promise is.
    """
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()

    ece = torch.zeros(1, device=probs.device)
    boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        share = in_bin.float().mean()
        if share.item() > 0:
            ece += (correct[in_bin].mean() - confidences[in_bin].mean()).abs() * share
    return ece.item()


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean squared error against the one-hot target. Proper scoring rule:
    it is minimized only by reporting your true beliefs, so unlike accuracy it
    cannot be gamed by confident guessing."""
    onehot = F.one_hot(labels, num_classes=probs.size(-1)).float()
    return ((probs - onehot) ** 2).sum(dim=-1).mean().item()


def reliability_table(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Per-bin confidence vs. accuracy — the numbers behind a reliability diagram."""
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()
    rows = []
    boundaries = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum().item())
        rows.append(
            {
                "bin_low": round(lo.item(), 3),
                "bin_high": round(hi.item(), 3),
                "count": count,
                "confidence": round(confidences[in_bin].mean().item(), 4) if count else 0.0,
                "accuracy": round(correct[in_bin].mean().item(), 4) if count else 0.0,
            }
        )
    return rows


def fit_temperature(
    logits: torch.Tensor,    # (n, k), padded slots must be -inf
    labels: torch.Tensor,    # (n,)
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a single scalar T minimizing NLL on held-out data.

    One parameter, so it needs very little data (a few hundred rows is plenty)
    and cannot change the argmax — it only rescales confidence. Always fit on a
    split the model did not train on, or you will "calibrate" against memorized
    answers and make things worse.
    """
    log_t = torch.zeros(1, requires_grad=True, device=logits.device)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = logits / log_t.exp()
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_t.exp().item()
    if not math.isfinite(temperature) or temperature <= 0:
        return 1.0
    return temperature
