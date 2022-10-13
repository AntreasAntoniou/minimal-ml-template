import multiprocessing
import os
from dataclasses import dataclass, MISSING
from typing import Any, Optional

import hydra
import timm
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    builds,
    hydrated_dataclass,
    make_config,
    ZenField,
    instantiate,
    just,
)
from rich import print
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments

from utils import get_logger, pretty_config

os.environ["HYDRA_FULL_ERROR"] = "1"

logger = get_logger(__name__, set_default_rich_handler=True)


# Using hydra might look a bit more verbose but it saves having to manually define
# future args, and makes it a lot easier to add whatever we need from the command line
def build_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    num_classes: int = 100,
    in_chans: int = 3,
):

    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans
    )
    return model


def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include=None,
):
    if sets_to_include is None:
        sets_to_include = ["train", "val", "test"]

    dataset = dict()
    for set_name in sets_to_include:
        data = load_dataset(
            path=dataset_name,
            split=set_name,
            download=True,
            cache_dir=data_dir,
        )
        dataset[set_name] = data

    return dataset


@hydrated_dataclass(target=torch.optim.AdamW)
class AdamWOptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    amsgrad: bool = False


@hydrated_dataclass(target=torch.optim.lr_scheduler.LinearLR)
class LinearAnnealingSchedulerConfig:
    optimizer: torch.optim.Optimizer
    start_factor: float
    total_iters: int


build_model_config = builds(build_model, populate_full_signature=True)

build_dataset_config = builds(build_dataset, populate_full_signature=True)

trainer_args = builds(TrainingArguments, populate_full_signature=True)


@dataclass
class BaseConfig:
    exp_name: str
    seed: int = 42

    model: Any = build_model
    datamodule: Any = build_dataset_config
    optimizer: Any = AdamWOptimizerConfig
    scheduler: Any = None

    resume: bool = False
    resume_from_checkpoint: Optional[int] = None
    print_config: bool = False
    batch_size: int = 128
    num_workers: int = multiprocessing.cpu_count()

    root_experiment_dir: str = (
        os.environ["EXPERIMENTS_DIR"]
        if "EXPERIMENTS_DIR" in os.environ
        else "experiment_root"
    )

    data_dir: str = (
        os.environ["DATASET_DIR"] if "DATASET_DIR" in os.environ else "data/"
    )

    current_experiment_dir: str = "${root_experiment_dir}/${exp_name}"
    code_dir: str = "${hydra:runtime.cwd}"


def collect_config_store():
    DATASET_DIR = "${data_dir}"
    EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
    EXPERIMENT_DIR = "${current_experiment_dir}"
    SEED = "${seed}"

    config_store = ConfigStore.instance()
    ###################################################################################
    vit_model_config = build_model_config(
        model_name="vit_base_patch16_224", pretrained=True, num_classes=1000, in_chans=3
    )

    tiny_imagenet_config = build_dataset_config(
        dataset_name="tiny_imagenet", data_dir=DATASET_DIR
    )

    zen_config = [
        ZenField(name=value.name, hint=value.type, default=value.default)
        if value.default is not MISSING
        else ZenField(name=value.name, hint=value.type)
        for key, value in BaseConfig.__dict__["__dataclass_fields__"].items()
    ]

    config = make_config(
        *zen_config,
    )
    # Config
    config_store.store(name="config", node=config)
    ###################################################################################

    config_store.store(
        group="model",
        name="vit_base_patch16_224",
        node=vit_model_config,
    )

    config_store.store(
        group="datamodule",
        name="tiny_imagenet",
        node=tiny_imagenet_config,
    )

    config_store.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)

    config_store.store(
        group="scheduler", name="linear-annealing", node=AdamWOptimizerConfig
    )

    default_training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        output_dir=EXPERIMENT_DIR,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=1,
        max_steps=10000,
        seed=SEED,
        data_seed=SEED,
        log_level="info",
        save_steps=1000,
        save_strategy="steps",
        save_total_limit=10,
        bf16=False,
        fp16=True,
    )
    config_store.store(
        group="training_args", name="default", node=default_training_args
    )

    return config_store


config_store = collect_config_store()


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: Any) -> None:
    print(pretty_config(cfg, resolve=True))

    model: nn.Module = instantiate(cfg.model)
    dataset: Dataset = instantiate(cfg.datamodule)
    optimizer: torch.optim.Optimizer = instantiate(cfg.optimizer)
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler, optimizer=optimizer
    )
    training_args: TrainingArguments = instantiate(cfg.training_args)
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        optimizers=(optimizer, scheduler),
    )

    checkpoint_filepath = None

    train_results = trainer.train(resume_from_checkpoint=checkpoint_filepath)
    test_results = trainer.evaluate(eval_dataset=dataset["test"])


if __name__ == "__main__":
    run()
