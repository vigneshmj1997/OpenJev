"""Temperature scaling and calibration metrics, per question type."""

from __future__ import annotations

import torch

from .data import MULTI, TYPES
from .model import openjev_loss, probabilities


@torch.no_grad()
def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, kinds: torch.Tensor | None = None,
                    t_min: float = 0.05, t_max: float = 20.0, steps: int = 400) -> float:
    """Temperature minimising NLL on held-out logits.

    A dense grid search in log space, not a gradient optimiser: it is a single
    scalar, and LBFGS line search divides by zero when the NLL is flat (e.g.
    near-identical logits early in training).
    """
    logits, targets = logits.float(), targets.float()
    grid = torch.logspace(torch.log10(torch.tensor(t_min)), torch.log10(torch.tensor(t_max)), steps)
    losses = torch.stack([openjev_loss(logits / t, targets, kinds) for t in grid])
    # Ties (flat NLL) resolve to the temperature closest to 1, i.e. leave logits alone.
    best = losses.min()
    near_best = torch.nonzero(losses <= best + 1e-6).squeeze(-1)
    pick = near_best[(grid[near_best].log()).abs().argmin()]
    return float(grid[pick])


def fit_temperatures(logits: torch.Tensor, targets: torch.Tensor, kinds: torch.Tensor) -> list[float]:
    """One temperature per question type, indexed like TYPES. Absent types get 1.0."""
    temperatures = []
    for i in range(len(TYPES)):
        rows = kinds == i
        temperatures.append(fit_temperature(logits[rows], targets[rows], kinds[rows]) if rows.any() else 1.0)
    return temperatures


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Top-label ECE: |confidence - accuracy| averaged over confidence bins."""
    confidence, predicted = probs.max(-1)
    correct = (predicted == labels).float()
    return _binned_gap(confidence, correct, n_bins)


def binary_calibration_error(probs: torch.Tensor, outcomes: torch.Tensor, n_bins: int = 15) -> float:
    """ECE for independent yes/no probabilities: |mean P - observed rate| per bin."""
    return _binned_gap(probs, outcomes.float(), n_bins)


def _binned_gap(predicted: torch.Tensor, observed: torch.Tensor, n_bins: int) -> float:
    edges = torch.linspace(0, 1, n_bins + 1)
    edges[0] = -1e-9  # so P = 0 lands in the first bin
    ece = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (predicted > lo) & (predicted <= hi)
        if in_bin.any():
            weight = in_bin.float().mean()
            ece += weight * (predicted[in_bin].mean() - observed[in_bin].mean()).abs()
    return float(ece)


def evaluate_logits(logits: torch.Tensor, targets: torch.Tensor, kinds: torch.Tensor | None = None,
                    temperatures=None) -> dict:
    """Metrics per question type present, plus example-weighted overall `accuracy` and `ece`.

    noul/choice/score: argmax accuracy and top-label ECE.
    multi: per-option accuracy at P >= 0.5 and binary ECE over every option.
    """
    logits, targets = logits.float(), targets.float()
    if kinds is None:
        kinds = torch.full((logits.shape[0],), TYPES.index("choice"))
    if temperatures is None:
        temperatures = [1.0] * len(TYPES)
    temperature = torch.as_tensor(temperatures, dtype=torch.float32)
    probs = probabilities(logits, kinds, temperature)
    scaled = logits / temperature[kinds].unsqueeze(-1)

    metrics = {"nll": float(openjev_loss(scaled, targets, kinds))}
    total = {"accuracy": 0.0, "ece": 0.0}
    for i, kind in enumerate(TYPES):
        rows = kinds == i
        if not rows.any():
            continue
        p, t = probs[rows], targets[rows]
        if i == MULTI:
            real = torch.isfinite(logits[rows])
            p, t = p[real], t[real]
            accuracy = float(((p >= 0.5) == (t >= 0.5)).float().mean())
            ece = binary_calibration_error(p, t)
        else:
            labels = t.argmax(-1)
            accuracy = float((p.argmax(-1) == labels).float().mean())
            ece = expected_calibration_error(p, labels)
        share = float(rows.float().mean())
        total["accuracy"] += share * accuracy
        total["ece"] += share * ece
        metrics.update({f"{kind}_accuracy": accuracy, f"{kind}_ece": ece,
                        f"{kind}_temperature": float(temperature[i])})
    return {**total, **metrics}
