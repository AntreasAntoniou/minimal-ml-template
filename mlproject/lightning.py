from typing import Any, Iterator, Tuple
from .decorators import collect_metrics

import torch
from hydra_zen import instantiate
from pytorch_lightning import LightningDataModule, LightningModule
from .utils import get_logger


logger = get_logger(__name__)


def get_dict_shapes(x):
    if not isinstance(x, dict):
        return get_dict_shapes(x.__dict__)
    return {
        key: value.shape if isinstance(value, torch.Tensor) else len(value)
        for key, value in x.items()
    }


class TrainingEvaluationAgent(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        fine_tunable: bool = False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fine_tunable = fine_tunable

    def forward(self, batch):
        return self.model.forward(**batch)

    @collect_metrics
    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        logits = outputs.logits
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = outputs["loss"].detach()
        opt_loss = outputs["loss"]

        return dict(
            phase_name="training",
            loss=opt_loss,
            metrics={"accuracy": accuracy, "loss": loss},
        )

    @collect_metrics
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        logits = outputs.logits
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = outputs["loss"].detach()

        return dict(
            phase_name="validation",
            metrics={"accuracy": accuracy, "loss": loss},
        )

    @collect_metrics
    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        logits = outputs.logits
        accuracy = (logits.argmax(dim=-1) == batch["labels"]).float().mean()
        loss = outputs["loss"].detach()

        return dict(phase_name="test", metrics={"accuracy": accuracy, "loss": loss})

    def parameters(self):
        if self.fine_tunable:
            return self.model.parameters()
        else:
            return self.model.classifier.parameters()

    def named_parameters(self) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.fine_tunable:
            return self.model.named_parameters()
        else:
            return self.model.classifier.named_parameters()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        for key, value in self.named_parameters():
            logger.info(
                f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
            )
        if self.scheduler is not None:
            return dict(
                optimizer=self.optimizer,
                lr_scheduler_config=dict(
                    scheduler=self.scheduler, interval="step", monitor="validation_loss"
                ),
            )
        return self.optimizer
