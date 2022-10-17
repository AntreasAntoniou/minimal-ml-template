import multiprocessing
import os
from dataclasses import dataclass, MISSING
from typing import Any, Optional, Callable, Dict

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
)
from rich import print
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from transformers import Trainer, TrainingArguments

from utils import get_logger, pretty_config

os.environ["HYDRA_FULL_ERROR"] = "1"

logger = get_logger(__name__, set_default_rich_handler=True)


@dataclass
class ModelAndTransform:
    model: nn.Module
    transform: Any


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

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    class Convert1ChannelTo3Channel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):

            if x.shape[0] == 1:
                x = x.repeat([3, 1, 1])
            return x

    transform.transforms[-2] = transforms.Compose(
        [ToTensor(), Convert1ChannelTo3Channel()]
    )

    def transform_wrapper(input_dict: Dict):
        input_dict["image"] = torch.stack(
            [transform(input_dict["image"][i]) for i in range(len(input_dict["image"]))]
        )
        input_dict["labels"] = torch.tensor(input_dict["labels"])
        print(input_dict["image"].shape, input_dict["labels"].shape)

        return dict(x=input_dict["image"], y=input_dict["labels"])

    return ModelAndTransform(model=model, transform=transform_wrapper)


def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include=None,
):
    if sets_to_include is None:
        sets_to_include = ["train", "valid"]

    dataset = dict()
    for set_name in sets_to_include:
        data = load_dataset(
            path=dataset_name,
            split=set_name,
            cache_dir=data_dir,
            task="image-classification",
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
    total_iters: int
    optimizer: Optional[Any] = None
    start_factor: float = 0.1


build_model_config = builds(build_model, populate_full_signature=True)

build_dataset_config = builds(build_dataset, populate_full_signature=True)

trainer_args_config = builds(TrainingArguments, populate_full_signature=True)


@dataclass
class BaseConfig:
    exp_name: str

    model: Any
    datamodule: Any
    optimizer: Any
    scheduler: Any
    training_args: Any

    seed: int = 42

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
    TOTAL_STEPS = "${training_args.max_steps}"
    SEED = "${seed}"

    config_store = ConfigStore.instance()
    ###################################################################################
    vit_model_config = build_model_config(
        model_name="vit_base_patch16_224", pretrained=True, num_classes=1000, in_chans=3
    )

    tiny_imagenet_config = build_dataset_config(
        dataset_name="Maysee/tiny-imagenet", data_dir=DATASET_DIR
    )

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
        group="scheduler",
        name="linear-annealing",
        node=LinearAnnealingSchedulerConfig(start_factor=0.1, total_iters=TOTAL_STEPS),
    )

    default_training_args = trainer_args_config(
        evaluation_strategy="STEPS",
        eval_steps=1000,
        logging_strategy="STEPS",
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
        save_strategy="STEPS",
        save_total_limit=10,
        bf16=False,
        fp16=True,
        tf32=False,
        lr_scheduler_type="LINEAR",
        optim="ADAMW_HF",
        hub_strategy="EVERY_SAVE",
        remove_unused_columns=False,
    )
    config_store.store(
        group="training_args", name="default", node=default_training_args
    )

    zen_config = [
        ZenField(name=value.name, hint=value.type, default=value.default)
        if value.default is not MISSING
        else ZenField(name=value.name, hint=value.type)
        for key, value in BaseConfig.__dict__["__dataclass_fields__"].items()
    ]

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            {"training_args": "default"},
            {"optimizer": "adamw"},
            {"scheduler": "linear-annealing"},
            {"model": "vit_base_patch16_224"},
            {"datamodule": "tiny_imagenet"},
        ]
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store


config_store = collect_config_store()


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: Any) -> None:
    print(pretty_config(cfg, resolve=True))

    model_and_transform: ModelAndTransform = instantiate(cfg.model)
    model: nn.Module = model_and_transform.model
    transform: Callable = model_and_transform.transform

    dataset: Dataset = instantiate(cfg.datamodule)
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["valid"]

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters()
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler, optimizer=optimizer
    )
    training_args: TrainingArguments = instantiate(cfg.training_args)
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        optimizers=(optimizer, scheduler),
    )

    checkpoint_filepath = None

    train_results = trainer.train(resume_from_checkpoint=checkpoint_filepath)
    # test_results = trainer.evaluate(eval_dataset=dataset["test"])


if __name__ == "__main__":
    run()
