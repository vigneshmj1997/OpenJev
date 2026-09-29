# TODO not completed
"""Two-stage training: fit the encoder + heads, then fit temperatures.

Every epoch the model is scored on the calibration split to fit one
temperature per question type (noul, choice, multi, score), then on the eval split
with those temperatures. The checkpoint with the lowest eval ECE is kept: the
product is the probability, not the argmax.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from .calibration import evaluate_logits, fit_temperatures
from .config import CHECKPOINT_ROOT, DEFAULT_MODEL, OpenJevConfig, base_config, find_checkpoint
from .data import Collator, load_jsonl, split
from .model import OpenJevModel, load_tokenizer, openjev_loss

MODEL_KEYS = ("input_ids", "attention_mask", "token_type_ids", "group_index", "option_index", "kinds")


@dataclass
class TrainArgs:
    train_file: str
    model: str = DEFAULT_MODEL
    eval_file: str | None = None
    calibration_file: str | None = None
    output_dir: str | None = None  # default: checkpoints/<model name>
    heldout_fraction: float = 0.1  # carved from train when eval/calibration files are missing
    epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    max_length: int | None = None
    pooling: str | None = None
    gradient_checkpointing: bool = False
    resume: bool = False  # start from an existing trained checkpoint for `model`
    device: str | None = None
    seed: int = 42
    log_every: int = 10


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def model_inputs(batch: dict) -> dict:
    out = {k: batch[k] for k in MODEL_KEYS if k in batch}
    out["num_groups"], out["max_options"] = batch["num_groups"], batch["max_options"]
    return out


@torch.no_grad()
def collect_logits(model: OpenJevModel, loader: DataLoader, device: torch.device):
    """Run the model over a split; returns (logits, targets, kinds) padded to a common k."""
    model.eval()
    all_logits, all_targets, all_kinds = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = model(**model_inputs(batch))
        all_logits.append(logits.float().cpu())
        all_targets.append(batch["targets"].cpu())
        all_kinds.append(batch["kinds"].cpu())
    k = max(t.shape[1] for t in all_logits)
    pad = lambda t, value: torch.nn.functional.pad(t, (0, k - t.shape[1]), value=value)
    return (torch.cat([pad(t, float("-inf")) for t in all_logits]),
            torch.cat([pad(t, 0.0) for t in all_targets]),
            torch.cat(all_kinds))


def autocast(device: torch.device):
    enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def build_model(args: TrainArgs) -> tuple[OpenJevModel, object]:
    checkpoint = find_checkpoint(args.model) if args.resume else None
    if checkpoint:
        print(f"Resuming from trained checkpoint {checkpoint}")
        model = OpenJevModel.from_pretrained(checkpoint)
        model.temperature.fill_(1.0)
        return model, load_tokenizer(model.config, checkpoint)
    config = base_config(args.model, args.max_length)
    if args.pooling:
        config.pooling = args.pooling
    print(f"Building OpenJev on encoder {config.encoder}")
    return OpenJevModel(config), load_tokenizer(config)


def train(args: TrainArgs) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    output_dir = args.output_dir or os.path.join(CHECKPOINT_ROOT, os.path.basename(args.model.rstrip("/")))

    # ---- data --------------------------------------------------------------
    train_set = load_jsonl(args.train_file)
    if args.eval_file:
        eval_set = load_jsonl(args.eval_file)
    else:
        train_set, eval_set = split(train_set, args.heldout_fraction, args.seed)
    if args.calibration_file:
        cal_set = load_jsonl(args.calibration_file)
    else:
        train_set, cal_set = split(train_set, args.heldout_fraction, args.seed + 1)
    print(f"train={len(train_set)} calibration={len(cal_set)} eval={len(eval_set)} device={device}")

    # ---- model -------------------------------------------------------------
    model, tokenizer = build_model(args)
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    model.to(device)

    collate = Collator(tokenizer, model.config.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    cal_loader = DataLoader(cal_set, batch_size=args.eval_batch_size, collate_fn=collate)
    eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size, collate_fn=collate)

    # Encoder and head get separate learning rates; no decay on biases/norms.
    groups = [
        {"params": [p for n, p in model.encoder.named_parameters() if p.ndim >= 2],
         "lr": args.learning_rate, "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.encoder.named_parameters() if p.ndim < 2],
         "lr": args.learning_rate, "weight_decay": 0.0},
        {"params": [*model.head.parameters(), *model.multi_head.parameters()],
         "lr": args.head_learning_rate, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs)
    warmup = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / (warmup + 1)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- loop --------------------------------------------------------------
    best, history, step = None, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader, 1):
            batch = to_device(batch, device)
            with autocast(device):
                logits = model(**model_inputs(batch))
            loss = openjev_loss(logits.float(), batch["targets"], batch["kinds"]) / args.grad_accum_steps
            loss.backward()
            running += loss.item() * args.grad_accum_steps
            if i % args.grad_accum_steps == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {running / i:.4f}")

        # Stage 2 on held-out data: one temperature per question type from
        # calibration, ECE from eval.
        temperatures = fit_temperatures(*collect_logits(model, cal_loader, device))
        metrics = evaluate_logits(*collect_logits(model, eval_loader, device), temperatures)
        metrics.update(epoch=epoch, train_loss=running / len(train_loader))
        history.append(metrics)
        print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "epoch"))

        if best is None or (metrics["ece"], metrics["nll"]) < (best["ece"], best["nll"]):
            best = metrics
            model.temperature.copy_(torch.tensor(temperatures))
            model.save_pretrained(output_dir, tokenizer)
            print(f"  saved best checkpoint (ece={metrics['ece']:.4f}) to {output_dir}")

    summary = {"best": best, "history": history, "args": asdict(args), "output_dir": output_dir}
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
