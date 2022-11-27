from ast import Dict
from dataclasses import dataclass
from typing import Any, Iterator, Tuple

from attr import field
from .decorators import collect_metrics

import torch
import torch.nn.functional as F
from hydra_zen import instantiate
from .utils import get_logger


logger = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class Evaluator(object):
    def __init__(self):
        pass


@dataclass
class EvaluatorOutput:
    step_idx: int
    metrics: Dict
    phase_name: str


class ClassificationEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    @collect_metrics
    def validation_step(self, model, batch, batch_idx, step_idx):
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = F.cross_entropy(logits, batch["labels"]).detach()

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="validation",
            metrics={"accuracy": accuracy, "loss": loss},
        )

    @collect_metrics
    def test_step(self, model, batch, batch_idx, step_idx):
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = F.cross_entropy(logits, batch["labels"]).detach()

        return EvaluatorOutput(
            step_idx=step_idx,
            phase_name="test",
            metrics={"accuracy": accuracy, "loss": loss},
        )
