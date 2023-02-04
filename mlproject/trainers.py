from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from hydra_zen import instantiate
from torch.utils.data import DataLoader

from mlproject.callbacks import Interval

from .decorators import collect_metrics
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
        experiment_tracker: wandb.wandb_sdk.wandb_run.Run = None,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.epoch_metrics = {}

        if self.scheduler is not None:
            assert scheduler_interval in {"step", "epoch"}
            self.scheduler_interval = scheduler_interval

    def get_optimizer(self):
        return self.optimizer

    @collect_metrics
    def training_step(
        self,
        model,
        batch,
        batch_idx,
        step_idx,
        epoch_idx,
        accelerator: Accelerator,
    ) -> TrainerOutput:
        model.train()
        self.optimizer.zero_grad()
        logits = model(batch["pixel_values"]).logits
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        opt_loss = F.cross_entropy(logits, batch["labels"])
        loss = opt_loss.detach()
        accelerator.backward(loss=opt_loss)
        self.optimizer.step()

        if self.scheduler is not None:
            if self.scheduler_interval == "step":
                self.scheduler.step(epoch=step_idx)
            elif self.scheduler_interval == "epoch" and batch_idx == 0:
                self.scheduler.step(epoch=epoch_idx)
        metrics = {"accuracy": accuracy, "loss": loss}
        for key, value in metrics.items():
            self.epoch_metrics.setdefault(key, []).append(value.detach().cpu())

        return TrainerOutput(
            phase_name="training",
            opt_loss=opt_loss,
            step_idx=step_idx,
            metrics={
                "accuracy": accuracy,
                "loss": loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            },
        )

    @collect_metrics
    def start_training(
        self,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader = None,
    ):
        self.epoch_metrics = {}
        return TrainerOutput(
            opt_loss=None, step_idx=step_idx, metrics={}, phase_name="training"
        )

    @collect_metrics
    def end_training(
        self,
        epoch_idx: int,
        step_idx: int,
        train_dataloader: DataLoader = None,
    ):
        epoch_metrics = {}
        for key, value in self.epoch_metrics.items():
            epoch_metrics[f"{key}-epoch-mean"] = torch.stack(value).mean()
            epoch_metrics[f"{key}-epoch-std"] = torch.stack(value).std()

        return TrainerOutput(
            opt_loss=None,
            step_idx=step_idx,
            metrics=epoch_metrics,
            phase_name="training",
        )
