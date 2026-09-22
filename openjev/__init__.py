"""OpenJev — an encoder-only, calibrated decision model."""

from typing import TYPE_CHECKING

from .schema import Question, QuestionType, confidence_from, serialize_answer

__all__ = [
    "OpenJev",
    "OpenJevModel",
    "Question",
    "QuestionType",
    "TrainConfig",
    "train",
    "serialize_answer",
    "confidence_from",
    "DEFAULT_BASE_MODEL",
]
__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover
    from .infer import OpenJev
    from .model import DEFAULT_BASE_MODEL, OpenJevModel
    from .train import TrainConfig, train


def __getattr__(name: str):
    """Defer torch-dependent imports so `schema` stays usable without torch."""
    if name in ("OpenJevModel", "DEFAULT_BASE_MODEL"):
        from . import model

        return getattr(model, name)
    if name == "OpenJev":
        from .infer import OpenJev

        return OpenJev
    if name in ("TrainConfig", "train"):
        from . import train as train_mod

        return getattr(train_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
