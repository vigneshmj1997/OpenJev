"""Training loop for OpenJev.

Two stages, in this order:
  1. Fit the encoder + scalar head with a soft-target cross-entropy.
  2. Fit the calibration temperature on a held-out split, with the rest frozen.

Stage 2 is not optional. The model that comes out of stage 1 ranks well and is
overconfident; the temperature is what makes the reported probability usable as
a routing signal.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .calibration import (
    brier_score,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
)
from .data import JevCollator, JevDataset
from .model import DEFAULT_BASE_MODEL, OpenJevModel


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_BASE_MODEL
    train_path: str = "data/train.jsonl"
    eval_path: str = "data/eval.jsonl"
    calibration_path: str = "data/calibration.jsonl"
    output_dir: str = "checkpoints/openjev"

    max_length: int = 2048
    batch_size: int = 4              # questions per step; each expands to k pairs
    grad_accum: int = 8
    epochs: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-4            # fresh head wants a faster rate than the encoder
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0

    pooling: str = "cls"
    head_dropout: float = 0.1
    gradient_checkpointing: bool = True
    bf16: bool = True
    seed: int = 42
    eval_every: int = 500
    log_every: int = 25
    num_workers: int = 2


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device, apply_temperature: bool = True) -> dict:
    """Collect logits over the whole split, then score accuracy AND calibration."""
    model.eval()
    all_probs, all_labels, all_logits, losses = [], [], [], []

    for batch in loader:
        batch = _move(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            group_index=batch["group_index"],
            option_index=batch["option_index"],
            num_questions=batch["num_questions"],
            max_k=batch["max_k"],
            target_dist=batch["target_dist"],
            apply_temperature=apply_temperature,
        )
        losses.append(out.loss.item())

        # Pad every batch to a common k so splits with mixed question types stack.
        probs = out.log_probs.exp()
        all_probs.append(probs.float().cpu())
        all_logits.append(out.logits.float().cpu())
        all_labels.append(batch["labels"].cpu())

    width = max(p.size(-1) for p in all_probs)
    probs = torch.cat([F.pad(p, (0, width - p.size(-1))) for p in all_probs])
    logits = torch.cat(
        [F.pad(l, (0, width - l.size(-1)), value=float("-inf")) for l in all_logits]
    )
    labels = torch.cat(all_labels)

    model.train()
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": probs.argmax(dim=-1).eq(labels).float().mean().item(),
        "ece": expected_calibration_error(probs, labels),
        "brier": brier_score(probs, labels),
        "_logits": logits,
        "_labels": labels,
        "_probs": probs,
    }


def train(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    device = _device()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    # [OPT] marks the boundary between the instruction and the option under test.
    if "[OPT]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[OPT]"]})

    model = OpenJevModel(
        base_model=config.base_model,
        pooling=config.pooling,
        head_dropout=config.head_dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    collate = JevCollator(tokenizer=tokenizer, max_length=config.max_length)
    train_ds = JevDataset(config.train_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.num_workers,
        drop_last=True,
    )
    eval_loader = DataLoader(
        JevDataset(config.eval_path),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.num_workers,
    )

    # Two parameter groups: the pretrained encoder moves slowly, the new head fast.
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = steps_per_epoch * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    use_amp = config.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"device={device} | examples={len(train_ds)} | steps={total_steps} | bf16={use_amp}")

    history: list[dict] = []
    best_ece = float("inf")
    step = 0
    start = time.time()
    model.train()

    for epoch in range(config.epochs):
        for micro, batch in enumerate(train_loader):
            batch = _move(batch, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    group_index=batch["group_index"],
                    option_index=batch["option_index"],
                    num_questions=batch["num_questions"],
                    max_k=batch["max_k"],
                    target_dist=batch["target_dist"],
                )
                loss = out.loss / config.grad_accum

            loss.backward()

            if (micro + 1) % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % config.log_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"epoch {epoch} step {step}/{total_steps} "
                        f"loss {out.loss.item():.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e} "
                        f"({elapsed:.0f}s)"
                    )

                if step % config.eval_every == 0 or step == total_steps:
                    metrics = evaluate(model, eval_loader, device, apply_temperature=False)
                    record = {
                        "step": step,
                        "loss": round(metrics["loss"], 4),
                        "accuracy": round(metrics["accuracy"], 4),
                        "ece": round(metrics["ece"], 4),
                        "brier": round(metrics["brier"], 4),
                    }
                    history.append(record)
                    print(f"  eval {record}")

                    if metrics["ece"] < best_ece:
                        best_ece = metrics["ece"]
                        save(model, tokenizer, config, out_dir / "best")

    # ---- Stage 2: calibration -------------------------------------------------
    calib_path = Path(config.calibration_path)
    temperature = 1.0
    if calib_path.exists():
        calib_loader = DataLoader(
            JevDataset(calib_path),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=config.num_workers,
        )
        raw = evaluate(model, calib_loader, device, apply_temperature=False)
        temperature = fit_temperature(raw["_logits"].to(device), raw["_labels"].to(device))
        model.set_temperature(temperature)

        calibrated = evaluate(model, calib_loader, device, apply_temperature=True)
        print(
            f"\ntemperature={temperature:.4f}  "
            f"ECE {raw['ece']:.4f} -> {calibrated['ece']:.4f}  "
            f"Brier {raw['brier']:.4f} -> {calibrated['brier']:.4f}  "
            f"(accuracy unchanged: {calibrated['accuracy']:.4f})"
        )
        (out_dir / "reliability.json").write_text(
            json.dumps(reliability_table(calibrated["_probs"], calibrated["_labels"]), indent=2)
        )
    else:
        print(f"\nno calibration split at {calib_path}; leaving temperature at 1.0")

    save(model, tokenizer, config, out_dir / "final")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    final = evaluate(model, eval_loader, device, apply_temperature=True)
    summary = {
        "temperature": temperature,
        "accuracy": round(final["accuracy"], 4),
        "ece": round(final["ece"], 4),
        "brier": round(final["brier"], 4),
        "steps": step,
        "minutes": round((time.time() - start) / 60, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{json.dumps(summary, indent=2)}")
    return summary


def save(model: OpenJevModel, tokenizer, config: TrainConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "temperature": model.temperature.item(),
        },
        path / "openjev.pt",
    )
    tokenizer.save_pretrained(path)
