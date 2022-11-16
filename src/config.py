from dataclasses import dataclass, field
import multiprocessing
import os
from typing import Any, List, Optional, Union
from .callbacks import LogConfigInformation, UploadCodeAsArtifact
from pytorch_lightning import Trainer
import torch
from hydra_zen import MISSING, ZenField, builds, make_config
from hydra.core.config_store import ConfigStore

from .data import build_dataset
from .models import build_model

from torch.utils.data import DataLoader

from dataclasses import MISSING, dataclass
from datetime import timedelta
from typing import Dict, Optional

from hydra_zen import builds, hydrated_dataclass
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import LoggerCollection, WandbLogger, TensorBoardLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

CHECKPOINT_DIR = "${hf_repo_dir}"
CODE_DIR = "${code_dir}"
DATASET_DIR = "${data_dir}"
EXPERIMENT_NAME = "${exp_name}"
EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
BATCH_SIZE = "${batch_size}"
CURRENT_EXPERIMENT_DIR = "${current_experiment_dir}"
TOTAL_STEPS = "${trainer.max_steps}"
REPO_PATH = "${repo_path}"
EXP_NAME = "${exp_name}"
SEED = "${seed}"


@hydrated_dataclass(target=WandbLogger)
class WeightsAndBiasesLoggerConfig:
    id: str = EXPERIMENT_NAME
    project: str = os.environ["WANDB_PROJECT"]
    offline: bool = False  # set True to store all logs only locally
    resume: str = "allow"  # allow, True, False, must
    save_dir: str = CURRENT_EXPERIMENT_DIR
    log_model: Union[str, bool] = False
    prefix: str = ""
    job_type: str = "train"
    group: str = ""
    tags: List[str] = field(default_factory=list)


@hydrated_dataclass(target=TensorBoardLogger)
class TensorboardLoggerConfig:
    save_dir: str = CURRENT_EXPERIMENT_DIR
    name: str = EXPERIMENT_NAME
    version: Optional[str] = None
    log_graph: bool = False
    default_hp_metric: Optional[bool] = None


@hydrated_dataclass(target=timedelta)
class TimerConfig:
    seconds: int = 60
    # minutes: int = 60


@hydrated_dataclass(target=ModelCheckpoint)
class ModelCheckpointingConfig:
    monitor: str = MISSING
    mode: str = MISSING
    save_top_k: int = MISSING
    save_last: bool = MISSING
    verbose: bool = MISSING
    filename: str = MISSING
    auto_insert_metric_name: bool = MISSING
    save_on_train_epoch_end: Optional[bool] = None
    train_time_interval: Optional[TimerConfig] = None
    dirpath: str = CHECKPOINT_DIR


@hydrated_dataclass(target=TQDMProgressBar)
class RichProgressBar:
    refresh_rate: int = 1
    process_position: int = 0


@hydrated_dataclass(target=LearningRateMonitor)
class LearningRateMonitorConfig:
    logging_interval: str = "step"


@hydrated_dataclass(target=UploadCodeAsArtifact)
class UploadCodeAsArtifactConfig:
    code_dir: str = "${code_dir}"


@hydrated_dataclass(target=LogConfigInformation)
class LogConfigInformationConfig:
    exp_config: Optional[Dict] = None


ModelSummaryConfig = builds(RichModelSummary, max_depth=7)


model_checkpoint_eval: ModelCheckpointingConfig = ModelCheckpointingConfig(
    monitor="validation/accuracy_epoch",
    mode="max",
    save_top_k=3,
    save_last=False,
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="eval_epoch.pt",
    auto_insert_metric_name=False,
)

model_checkpoint_train = ModelCheckpointingConfig(
    monitor="training/loss_epoch",
    save_on_train_epoch_end=True,
    save_top_k=0,
    save_last=True,
    train_time_interval=TimerConfig(),
    mode="min",
    verbose=False,
    dirpath=CHECKPOINT_DIR,
    filename="latest.pt",
    auto_insert_metric_name=False,
)

base_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitorConfig(),
)

wandb_callbacks = dict(
    model_checkpoint_eval=model_checkpoint_eval,
    model_checkpoint_train=model_checkpoint_train,
    model_summary=ModelSummaryConfig(),
    progress_bar=RichProgressBar(),
    lr_monitor=LearningRateMonitorConfig(),
    code_upload=UploadCodeAsArtifactConfig(),
    log_config=LogConfigInformationConfig(),
)


adamw_optimizer_config = builds(
    torch.optim.AdamW,
    populate_full_signature=True,
    zen_partial=True,
)


linear_learning_rate_scheduler_config = builds(
    torch.optim.lr_scheduler.LinearLR,
    populate_full_signature=True,
    zen_partial=True,
)

model_config = builds(build_model, populate_full_signature=True)

dataset_config = builds(build_dataset, populate_full_signature=True)

dataloader_config = builds(DataLoader, dataset=None, populate_full_signature=True)

trainer_config = builds(Trainer, populate_full_signature=True)


@dataclass
class BaseConfig:

    # Must be passed at command line -- neccesary arguments

    exp_name: str = MISSING
    hf_username: str = MISSING

    # Defaults for these are provided in the collect_config_store method,
    # but will be often overridden at command line

    model: Any = MISSING
    dataset: Any = MISSING
    dataloader: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    trainer: Any = MISSING

    callbacks: Any = MISSING
    loggers: Any = MISSING

    seed: int = 42

    fine_tunable: bool = False
    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = False
    batch_size: int = 16
    num_workers: int = multiprocessing.cpu_count()

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "/experiments"
    )

    data_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "/data"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    repo_path: str = "${hf_username}/${exp_name}"
    hf_repo_dir: str = "${current_experiment_dir}/hf_repo"
    code_dir: str = "${hydra:runtime.cwd}"


# Using hydra might look a bit more verbose but it saves having to manually define
# future args, and makes it a lot easier to add whatever we need from the command line


def collect_config_store():

    config_store = ConfigStore.instance()
    ###################################################################################
    vit_model_config = model_config(
        model_name="vit_base_patch16_224", pretrained=True, num_classes=1000, in_chans=3
    )

    tiny_imagenet_config = dataset_config(dataset_name="food101", data_dir=DATASET_DIR)

    ###################################################################################

    config_store.store(
        group="model",
        name="vit_base_patch16_224",
        node=vit_model_config,
    )

    config_store.store(
        group="dataset",
        name="tiny_imagenet",
        node=tiny_imagenet_config,
    )

    config_store.store(
        group="dataloader",
        name="default",
        node=dataloader_config(
            batch_size=BATCH_SIZE,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            shuffle=True,
        ),
    )

    config_store.store(group="optimizer", name="adamw", node=adamw_optimizer_config)

    config_store.store(
        group="scheduler",
        name="linear-annealing",
        node=linear_learning_rate_scheduler_config(
            start_factor=0.1, total_iters=TOTAL_STEPS
        ),
    )

    config_store.store(
        group="callbacks",
        name="base",
        node=base_callbacks,
    )

    config_store.store(
        group="callbacks",
        name="wandb",
        node=wandb_callbacks,
    )
    ###################################################################################
    config_store.store(
        group="loggers",
        name="wandb",
        node=dict(wandb=WeightsAndBiasesLoggerConfig()),
    )

    config_store.store(
        group="loggers",
        name="tb",
        node=dict(tensorboard_logger=TensorboardLoggerConfig()),
    )

    config_store.store(
        group="loggers",
        name="wandb+tb",
        node=dict(
            tensorboard=TensorboardLoggerConfig(),
            wandb=WeightsAndBiasesLoggerConfig(),
        ),
    )

    config_store.store(
        group="trainer",
        name="default",
        node=trainer_config(
            auto_select_gpus=True,
            gpus=1,
            max_steps=1000,
            val_check_interval=0.50,
            log_every_n_steps=1,
            precision=32,
        ),
    )

    config_store.store(
        group="hydra",
        name="custom_logging_path",
        node=dict(
            job_logging=dict(
                version=1,
                formatters=dict(
                    simple=dict(
                        level="INFO",
                        format="%(message)s",
                        datefmt="[%X]",
                    )
                ),
                handlers=dict(
                    rich={
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                ),
                root={"handlers": ["rich"], "level": "INFO"},
                disable_existing_loggers=False,
            ),
            hydra_logging={
                "version": 1,
                "formatters": {
                    "simple": {
                        "level": "INFO",
                        "format": "%(message)s",
                        "datefmt": "[%X]",
                    }
                },
                "handlers": {
                    "rich": {
                        "class": "rich.logging.RichHandler",
                        "formatter": "simple",
                    }
                },
                "root": {"handlers": ["rich"], "level": "INFO"},
                "disable_existing_loggers": False,
            },
            run={"dir": "${current_experiment_dir}/hydra-run/${now:%Y-%m-%d_%H-%M-%S}"},
            sweep={
                "dir": "${current_experiment_dir}/hydra-multirun/${now:%Y-%m-%d_%H-%M-%S}",
                "subdir": "${hydra.job.num}",
            },
        ),
    )

    zen_config = []

    for key, value in BaseConfig.__dataclass_fields__.items():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            {"trainer": "default"},
            {"optimizer": "adamw"},
            {"scheduler": "linear-annealing"},
            {"model": "vit_base_patch16_224"},
            {"dataset": "tiny_imagenet"},
            {"dataloader": "default"},
            {"hydra": "custom_logging_path"},
            {"loggers": "wandb"},
            {"callbacks": "wandb"},
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store
