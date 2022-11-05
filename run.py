import copy
import multiprocessing
import os
from dataclasses import dataclass
import pathlib
from typing import Any, Optional, Callable, Dict

import hydra
import timm
import torch
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    MISSING,
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
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from transformers import Trainer, TrainingArguments
from huggingface_hub import hf_hub_download, create_repo, Repository

from utils import get_logger, pretty_config

os.environ["HYDRA_FULL_ERROR"] = "1"

logger = get_logger(__name__, set_default_rich_handler=True)


class CustomTrainer(Trainer):
    pass


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
    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=num_classes
    )
    transform = lambda image: feature_extractor(images=image, return_tensors="pt")

    class Convert1ChannelTo3Channel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            temp = None
            if hasattr(x, "pixel_values"):
                temp = copy.copy(x)
                x = x["pixel_values"]
            x = ToTensor()(x)
            if len(x.shape) == 3 and x.shape[0] == 1:
                x = x.repeat([3, 1, 1])
            elif len(x.shape) == 4 and x.shape[1] == 1:
                x = x.repeat([1, 3, 1, 1])

            if temp is not None:
                temp["pixel_values"] = x
                x = temp

            return x

    post_transform = Convert1ChannelTo3Channel()

    def transform_wrapper(input_dict: Dict):
        input_dict["image"][0] = post_transform(input_dict["image"][0])
        output_dict = {}
        output_dict["pixel_values"] = transform(input_dict["image"])["pixel_values"]
        output_dict["labels"] = input_dict["labels"]

        return output_dict

    return ModelAndTransform(model=model, transform=transform_wrapper)


def build_dataset(
    dataset_name: str,
    data_dir: str,
    sets_to_include=None,
):
    if sets_to_include is None:
        sets_to_include = ["train", "valid"]

    dataset = {}
    for set_name in sets_to_include:
        data = load_dataset(
            path=dataset_name,
            split=set_name,
            cache_dir=data_dir,
            task="image-classification",
        )
        dataset[set_name] = data

    return dataset


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

trainer_args_config = builds(TrainingArguments, populate_full_signature=True)


@dataclass
class BaseConfig:

    model: Any = MISSING
    datamodule: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING
    training_args: Any = MISSING

    exp_name: str = MISSING
    hf_username: str = MISSING

    seed: int = 42

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


def collect_config_store():
    DATASET_DIR = "${data_dir}"
    EXPERIMENTS_ROOT_DIR = "${root_experiment_dir}"
    BATCH_SIZE = "${batch_size}"
    EXPERIMENT_DIR = "${current_experiment_dir}"
    TOTAL_STEPS = "${training_args.max_steps}"
    HF_REPO_DIR = "${hf_repo_dir}"
    REPO_PATH = "${repo_path}"
    EXP_NAME = "${exp_name}"
    SEED = "${seed}"

    config_store = ConfigStore.instance()
    ###################################################################################
    vit_model_config = model_config(
        model_name="vit_base_patch16_224", pretrained=True, num_classes=1000, in_chans=3
    )

    tiny_imagenet_config = dataset_config(
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

    config_store.store(group="optimizer", name="adamw", node=adamw_optimizer_config)

    config_store.store(
        group="scheduler",
        name="linear-annealing",
        node=linear_learning_rate_scheduler_config(
            start_factor=0.1, total_iters=TOTAL_STEPS
        ),
    )

    default_training_args = trainer_args_config(
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        output_dir=HF_REPO_DIR,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
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
        tf32=False,
        lr_scheduler_type="linear",
        optim="adamw_hf",
        hub_strategy="every_save",
        remove_unused_columns=False,
        push_to_hub=True,
        push_to_hub_model_id=EXP_NAME,
        report_to="wandb"
    )
    config_store.store(
        group="training_args", name="default", node=default_training_args
    )

    zen_config = []

    for key, value in BaseConfig.__dataclass_fields__.items():
        print(key, value)
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
            {"training_args": "default"},
            {"optimizer": "adamw"},
            {"scheduler": "linear-annealing"},
            {"model": "vit_base_patch16_224"},
            {"datamodule": "tiny_imagenet"},
        ],
    )
    # Config
    config_store.store(name="config", node=config)

    return config_store


config_store = collect_config_store()


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    print(pretty_config(cfg, resolve=True))

    repo_path = cfg.repo_path

    create_repo(repo_path, repo_type="model", exist_ok=True)
    print(f"Created repo {repo_path}, {cfg.hf_repo_dir}")

    repo = Repository(
        local_dir=cfg.hf_repo_dir,
        clone_from=repo_path,
    )

    logger.info(
        "Download existing checkpoints, if they exist, from the huggingface hub"
    )

    hf_repo = pathlib.Path(repo.local_dir)

    if not pathlib.Path(cfg.current_experiment_dir).exists():
        pathlib.Path(cfg.current_experiment_dir).mkdir(parents=True, exist_ok=True)

    found_hub_repo = False

    try:
        hf_hub_download(
            repo_id=cfg.exp_name,
            cache_dir=hf_repo,
            resume_download=True,
            filename="checkpoint_latest.pt",
        )
        found_hub_repo = True
    except Exception as e:
        logger.exception(
            f"Could not download checkpoint_latest.pt from huggingface hub: {e}"
        )

    model_and_transform: ModelAndTransform = instantiate(cfg.model)
    model: nn.Module = model_and_transform.model
    transform: Callable = model_and_transform.transform

    dataset: Dataset = instantiate(cfg.datamodule)
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["valid"]

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler, optimizer=optimizer, _partial_=False
    )
    print("Optimizer", optimizer, "Scheduler", scheduler)

    training_args: TrainingArguments = instantiate(cfg.training_args)
    trainer: CustomTrainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        optimizers=(optimizer, scheduler),
    )

    checkpoint_filepath = None

    if cfg.resume:
        checkpoint_filepath = pathlib.Path(repo.local_dir / "checkpoint_latest.pt")

        if checkpoint_filepath.exists() and found_hub_repo:
            logger.info(f"Resuming from checkpoint: {checkpoint_filepath}")
        else:
            checkpoint_filepath = None

    train_results = trainer.train(resume_from_checkpoint=checkpoint_filepath)
    test_results = trainer.evaluate(eval_dataset=dataset["test"])


if __name__ == "__main__":
    run()
