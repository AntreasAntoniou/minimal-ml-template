from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

from mlproject.callbacks import Interval

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


class Trainer(object):
    def __init__(self):
        pass

@dataclass
class TrainerOutput:
    opt_loss: torch.Tensor
    step_idx: int
    metrics: Dict[str, Any]
    phase_name: str 

class ClassificationTrainer(Trainer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        scheduler_interval: str = Interval.STEP,
        fine_tunable: bool = False,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fine_tunable = fine_tunable

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval    

    def get_optimizer(self):
        return self.optimizer
    @collect_metrics
    def training_step(self, model, batch, batch_idx, step_idx) -> TrainerOutput:
        self.optimizer.zero_grad()
        logits = model(batch["pixel_values"])
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        opt_loss = F.cross_entropy(logits, batch["labels"])
        loss = opt_loss.detach()
        opt_loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            if self.scheduler_interval == "step":
                self.scheduler.step()
            elif self.scheduler_interval == "epoch" and batch_idx == 0:
                self.scheduler.step()
                
        return TrainerOutput(
            phase_name="training",
            opt_loss=opt_loss,
            step_idx=step_idx,
            metrics={"accuracy": accuracy, "loss": loss},
        )

    