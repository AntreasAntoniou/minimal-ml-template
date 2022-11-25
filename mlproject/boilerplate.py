from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from mlproject.trainers import Trainer
from mlproject.evaluators import Evaluator


class Callback(object):
    def __init__(self) -> None:
        pass


@dataclass
class EvalInterval:
    EPOCH: str = "epoch"
    STEP: str = "step"


class Experiment(nn.Module):
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: Union[str, Path],
        model: torch.nn.Module,
        evaluate_every_n_steps: int = None,
        evaluate_every_n_epochs: int = None,
        checkpoint_every_n_steps: int = None,
        checkpoint_after_validation: bool = False,
        train_iters: int = None,
        train_epochs: int = None,
        train_dataloaders: Union[List[DataLoader], DataLoader] = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
        trainers: Union[List[Trainer], Trainer] = None,
        evaluators: Union[List[Evaluator], Evaluator] = None,
        callbacks: Union[List[Callback], Callback] = None,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.experiment_dir = (
            experiment_dir if isinstance(experiment_dir, Path) else Path(experiment_dir)
        )
        self.model = model
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_after_validation = checkpoint_after_validation
        self.step_idx = 0
        self.epoch_idx = 0
        self.train_iters = train_iters
        self.train_epochs = train_epochs

        self.eval_mode = (
            EvalInterval.STEP if self.evaluate_every_n_steps else EvalInterval.EPOCH
        )

        if evaluate_every_n_steps is not None and evaluate_every_n_epochs is not None:
            raise ValueError(
                "You can only specify one of `evaluate_every_n_steps` and `evaluate_every_n_epochs`"
            )

        self.train_dataloaders = (
            [self.train_dataloaders]
            if isinstance(train_dataloaders, DataLoader)
            else train_dataloaders
        )
        self.val_dataloaders = (
            [val_dataloaders]
            if isinstance(val_dataloaders, DataLoader)
            else val_dataloaders
        )
        self.test_dataloaders = (
            [test_dataloaders]
            if isinstance(test_dataloaders, DataLoader)
            else test_dataloaders
        )
        self.trainers = [trainers] if isinstance(trainers, Trainer) else trainers
        self.evaluators = (
            [evaluators] if isinstance(evaluators, Evaluator) else evaluators
        )
        self.callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks

    def run(self):
        print(f"Running experiment with config: {self.config}")

    def __call__(self):
        self.run()

    def __repr__(self):
        return "Experiment(config={self.config})"

    def __str__(self):
        return self.__repr__()

    def training_step(self, model, batch, batch_idx):
        pass

    def validation_step(self, model, batch, batch_idx):
        pass

    def testing_step(self, model, batch, batch_idx):
        pass

    def training_loop(self):
        for epoch_idx in range(self.epoch_idx, self.train_epochs):
            self.epoch_idx = epoch_idx

            if (
                self.eval_mode == EvalInterval.EPOCH
                and epoch_idx % self.evaluate_every_n_epochs == 0
            ):
                self.validation_loop()

            for batch_idx, batch in enumerate(self.train_dataloader):
                self.training_step(self.model, batch, batch_idx)

                if (
                    self.eval_mode == EvalInterval.STEP
                    and self.step_idx % self.evaluate_every_n_steps == 0
                ):
                    self.validation_loop()

                if self.step_idx % self.checkpoint_every_n_steps == 0:
                    self.save_checkpoint()

                self.step_idx += 1

                if self.step_idx >= self.train_iters:
                    self.end_training()
        self.end_training()

    def save_checkpoint(self):
        experiment_hyperparameters = dict(
            step_idx=self.step_idx, epoch_idx=self.epoch_idx
        )
        optimizers: List[torch.optim.Optimizer] = [
            trainer.get_optimizer().state_dict() for trainer in self.trainers
        ]
        model = self.model.state_dict()

        state = dict(
            exp_state=experiment_hyperparameters, optimizers=optimizers, model=model
        )
        torch.save(state, self.experiment_dir / f"{self.experiment_name}.pt")


# Ensure continued experiments work properly, especially dataloaders continuing from the right step_idx
# Add load checkpoint functionality
# Add validation and testing loops
# Add explicit call for validation and testing without training
# Add callback functionality for training, validation, testing, and checkpointing
# Build a standard Trainer and Evaluator for classification
