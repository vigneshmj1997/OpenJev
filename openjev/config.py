"""Model presets and config resolution.

`--model` everywhere accepts one of:
  * a preset name from PRESETS ("openjev" is the default),
  * a directory written by `OpenJevModel.save_pretrained` (a trained model),
  * any Hugging Face encoder id or local encoder path (fresh scoring head).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

DEFAULT_MODEL = "openjev"
CONFIG_NAME = "openjev_config.json"
CHECKPOINT_ROOT = "checkpoints"

PRESETS: dict[str, dict] = {
    # Default. ModernBERT-base: 8k context, RoPE, bidirectional attention.
    "openjev": {"encoder": "answerdotai/ModernBERT-base", "max_length": 8192},
    # Small and fast: Google BERT-mini (4 layers, 256 hidden, ~11M params), good for CPU and quick experiments.
    "openjev-large": {"encoder": "answerdotai/ModernBERT-large", "max_length": 8192},
    # Mulilingual 
    "openjev-large": {"encoder":"gte-multilingual-mlm-base","max_length": 8192},
}


@dataclass
class OpenJevConfig:
    encoder: str
    max_length: int = 8192
    pooling: str = "cls"  # "cls" or "mean"
    dropout: float = 0.1

    def save(self, directory: str) -> None:
        with open(os.path.join(directory, CONFIG_NAME), "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "OpenJevConfig":
        with open(os.path.join(directory, CONFIG_NAME)) as f:
            return cls(**json.load(f))


def is_trained_checkpoint(path: str) -> bool:
    return os.path.isfile(os.path.join(path, CONFIG_NAME))


def find_checkpoint(name_or_path: str) -> str | None:
    """Return a trained checkpoint dir for `name_or_path`, if one exists.

    A preset name also matches `checkpoints/<name>`, which is where training
    writes by default, so `--model openjev` picks up your trained model.
    """
    if is_trained_checkpoint(name_or_path):
        return name_or_path
    if name_or_path in PRESETS:
        candidate = os.path.join(CHECKPOINT_ROOT, name_or_path)
        if is_trained_checkpoint(candidate):
            return candidate
    return None


def base_config(name_or_path: str, max_length: int | None = None) -> OpenJevConfig:
    """Config for training from a preset or an arbitrary encoder."""
    if name_or_path in PRESETS:
        config = OpenJevConfig(**PRESETS[name_or_path])
    else:
        config = OpenJevConfig(encoder=name_or_path)
    if max_length is not None:
        config.max_length = max_length
    return config
