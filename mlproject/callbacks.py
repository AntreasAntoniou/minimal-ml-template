import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from .utils import get_logger

logger = get_logger(__name__, set_default_rich_handler=True)


@dataclass
class Interval:
    EPOCH: str = "epoch"
    STEP: str = "step"


class Callback(object):
    def __init__(self) -> None:
        pass

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_epoch_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_epoch_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_batch_start(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        pass

    def on_batch_end(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        pass

    def on_training_step_start(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_training_step_end(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_validation_step_start(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_validation_step_end(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_testing_step_start(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_testing_step_end(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        pass

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
    ) -> None:
        pass

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ):
        pass

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_testing_start(
        self,
        experiment: Any,
        model: nn.Module,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_testing_end(
        self,
        experiment: Any,
        model: nn.Module,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        pass

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        pass

    # class UploadCodeAsArtifact(Callback):
    # """Upload all code files to wandb as an artifact, at the beginning of the run."""

    # def __init__(self, code_dir: str):
    #     """

    #     Args:
    #         code_dir: the code directory
    #         use_git: if using git, then upload all files that are not ignored by git.
    #         if not using git, then upload all '*.py' file
    #     """
    #     self.code_dir = code_dir

    # @rank_zero_only
    # def on_train_start(self, trainer, pl_module):
    #     logger = get_wandb_logger(trainer=trainer)
    #     experiment = logger.experiment

    #     code = wandb.Artifact("project-source", type="code")

    #     for path in Path(self.code_dir).resolve().rglob("*.py"):
    #         code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

    #     experiment.log_artifact(code)


class UploadCheckpointsToHuggingFace(Callback):
    def __init__(self, repo_name: str, repo_owner: str):
        from huggingface_hub import HfApi

        super().__init__()
        self.repo_name = repo_name
        self.repo_owner = repo_owner
        self.hf_api = HfApi()

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        self.hf_api.upload_file(
            repo_id=f"{self.repo_owner}/{self.repo_name}",
            path_or_fileobj=checkpoint_path.as_posix(),
            path_in_repo=f"checkpoints/{checkpoint_path.name}",
        )

    # class LogConfigInformation(Callback):
    # """Logs a validation batch and their predictions to wandb.
    # Example adapted from:
    #     https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY
    # """

    # def __init__(self, exp_config=None):
    #     super().__init__()
    #     self.done = False
    #     self.exp_config = exp_config

    # @rank_zero_only
    # def on_batch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     if not self.done:
    #         logger = get_wandb_logger(trainer=trainer)

    #         trainer_hparams = trainer.__dict__.copy()

    #         hparams = {
    #             "trainer": trainer_hparams,
    #             "config": self.exp_config,
    #         }

    #         logger.log_hyperparams(hparams)
    #         self.done = True
