"""transformers.Pipeline wrapper around OpenJevModel."""

from __future__ import annotations

import collections.abc

from torch.utils.data import DataLoader
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineDataset, PipelineIterator

from .data import Collator, Example, question_from
from .infer import answer
from .model import OpenJevModel, probabilities
from .train import autocast, model_inputs


class OpenJevPipeline(Pipeline):
    """transformers.Pipeline over OpenJevModel. Takes the question dicts `OpenJev.ask_batch` takes.

        pipe = OpenJevPipeline(model=model, tokenizer=tokenizer, device="cpu")
        pipe({"type": "noul", "state": "The item arrived broken."})
        pipe([{"state": "...", "options": ["a", "b"]}, ...], batch_size=16, threshold=0.5)

    A question with k options is k (state, option) rows, so batching goes
    through `Collator` instead of the base class's per-key padding.
    """

    _load_tokenizer = True

    def __init__(self, model: OpenJevModel, tokenizer, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        self.model.eval()  # the base class never switches off dropout
        self.collate = Collator(tokenizer, model.config.max_length)

    def __repr__(self):
        # The base __repr__ calls PreTrainedModel-only methods.
        return f"{self.__class__.__name__}(encoder={self.model.config.encoder!r}, device={self.device})"

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        # Keep the checkpoint layout OpenJevModel.from_pretrained expects (tokenizer under encoder/).
        self.model.save_pretrained(save_directory, tokenizer=self.tokenizer)

    def _sanitize_parameters(self, threshold=None, **kwargs):
        postprocess = {"threshold": threshold} if threshold is not None else {}
        return {}, {}, postprocess

    def _example(self, question: dict) -> Example:
        return question_from(question, "question")

    def _batch(self, examples: list[Example]) -> dict:
        return {**self.collate(examples), "examples": examples}

    def preprocess(self, question: dict) -> dict:
        return self._batch([self._example(question)])

    def _forward(self, batch: dict) -> dict:
        with autocast(self.device):
            logits = self.model(**model_inputs(batch))
        probs = probabilities(logits.float(), batch["kinds"], self.model.temperature)
        # "probs" first: the base iterator reads the batch size off the first key.
        return {"probs": probs, "examples": batch["examples"]}

    def postprocess(self, output: dict, threshold: float = 0.5) -> dict:
        example = output["examples"]
        if isinstance(example, list):  # unbatched runs keep the list; batched runs unroll it
            example = example[0]
        return answer(example, output["probs"][0, : len(example.options)].tolist(), threshold)

    def get_iterator(self, inputs, num_workers, batch_size, preprocess_params, forward_params,
                     postprocess_params):
        # Same chain as the base class, but questions become Examples first and
        # Collator batches them, since per-key padding can't handle ragged k.
        if isinstance(inputs, collections.abc.Sized):
            dataset = PipelineDataset(inputs, self._example, {})
        else:
            dataset = PipelineIterator(inputs, self._example, {})
            num_workers = min(num_workers, 1)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=self._batch)
        model_iterator = PipelineIterator(loader, self.forward, forward_params, loader_batch_size=batch_size)
        return PipelineIterator(model_iterator, self.postprocess, postprocess_params)
