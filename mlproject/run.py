import os
import shutil

import wandb
from rich import print
from rich.traceback import install

from mlproject.models import ModelAndTransform
from mlproject.utils import save_json

os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups

install()  # beautiful and clean tracebacks for debugging


import pathlib
from typing import Callable, List, Optional, Union

import hydra
import torch
from huggingface_hub import (
    Repository,
    create_repo,
    hf_hub_download,
    login,
    snapshot_download,
)
from hydra_zen import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from mlproject.boilerplate import Learner
from mlproject.callbacks import Callback
from mlproject.config import BaseConfig, collect_config_store
from mlproject.evaluators import ClassificationEvaluator
from mlproject.trainers import ClassificationTrainer
from mlproject.utils import get_logger, pretty_config, set_seed

config_store = collect_config_store()

logger = get_logger(name=__name__)


def instantiate_callbacks(callback_dict: dict) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        callbacks.append(instantiate(cb_conf))

    return callbacks


def create_hf_model_repo_and_download_maybe(cfg: BaseConfig):
    import orjson
    import yaml
    from huggingface_hub import HfApi

    if (
        cfg.download_checkpoint_with_name is not None
        and cfg.download_latest is True
    ):
        raise ValueError(
            "Cannot use both continue_from_checkpoint_with_name and continue_from_latest"
        )

    repo_path = cfg.repo_path
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
    print(
        f"Logged in to huggingface with token {os.environ['HF_TOKEN']}, creating repo {repo_path}"
    )
    repo_url = create_repo(repo_path, repo_type="model", exist_ok=True)

    logger.info(f"Created repo {repo_path}, {cfg.hf_repo_dir}")

    if not pathlib.Path(cfg.hf_repo_dir).exists():
        pathlib.Path(cfg.hf_repo_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(pathlib.Path(cfg.hf_repo_dir) / "checkpoints").mkdir(
            parents=True, exist_ok=True
        )

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    hf_api = HfApi()
    config_json_path: pathlib.Path = save_json(
        filepath=pathlib.Path(cfg.hf_repo_dir) / "config.json",
        dict_to_store=config_dict,
        overwrite=True,
    )
    hf_api.upload_file(
        repo_id=repo_path,
        path_or_fileobj=config_json_path.as_posix(),
        path_in_repo="config.json",
    )

    config_yaml_path = pathlib.Path(cfg.hf_repo_dir) / "config.yaml"
    with open(config_yaml_path, "w") as file:
        documents = yaml.dump(config_dict, file)

    hf_api.upload_file(
        repo_id=repo_path,
        path_or_fileobj=config_yaml_path.as_posix(),
        path_in_repo="config.yaml",
    )

    try:
        if cfg.download_checkpoint_with_name is not None:
            logger.info(
                f"Download {cfg.download_checkpoint_with_name} checkpoint, if it exists, from the huggingface hub 👨🏻‍💻"
            )

            ckpt_filepath = hf_hub_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
                subfolder="checkpoints",
                filename=cfg.download_checkpoint_with_name,
                repo_type="model",
            )
            if pathlib.Path(
                pathlib.Path(cfg.hf_repo_dir) / "checkpoints"
            ).exists():
                pathlib.Path(
                    pathlib.Path(cfg.hf_repo_dir) / "checkpoints"
                ).mkdir(parents=True, exist_ok=True)

            shutil.copy(
                pathlib.Path(ckpt_filepath),
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / cfg.download_checkpoint_with_name,
            )
            logger.info(
                f"Downloaded checkpoint from huggingface hub to {cfg.hf_repo_dir}"
            )
            return (
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / cfg.download_checkpoint_with_name
            ), repo_url

        elif cfg.download_latest:
            logger.info(
                "Download latest checkpoint, if it exists, from the huggingface hub 👨🏻‍💻"
            )

            optimizer_filepath = hf_hub_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
                subfolder="checkpoints/latest",
                filename="optimizer.bin",
                repo_type="model",
            )

            model_filepath = hf_hub_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
                subfolder="checkpoints/latest",
                filename="pytorch_model.bin",
                repo_type="model",
            )

            random_states_filepath = hf_hub_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
                subfolder="checkpoints/latest",
                filename="random_states_0.pkl",
                repo_type="model",
            )

            trainer_state_filepath = hf_hub_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
                subfolder="checkpoints/latest",
                filename="trainer_state.pt",
                repo_type="model",
            )

            if not pathlib.Path(
                pathlib.Path(cfg.hf_repo_dir) / "checkpoints" / "latest"
            ).exists():
                pathlib.Path(
                    pathlib.Path(cfg.hf_repo_dir) / "checkpoints" / "latest"
                ).mkdir(parents=True, exist_ok=True)

            shutil.copy(
                pathlib.Path(optimizer_filepath),
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / "latest"
                / "optimizer.bin",
            )

            shutil.copy(
                pathlib.Path(model_filepath),
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / "latest"
                / "pytorch_model.bin",
            )

            shutil.copy(
                pathlib.Path(random_states_filepath),
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / "latest"
                / "random_states_0.pkl",
            )

            shutil.copy(
                pathlib.Path(trainer_state_filepath),
                pathlib.Path(cfg.hf_repo_dir)
                / "checkpoints"
                / "latest"
                / "trainer_state.pt",
            )

            logger.info(
                f"Downloaded checkpoint from huggingface hub to {cfg.hf_repo_dir}"
            )
            return (
                pathlib.Path(cfg.hf_repo_dir) / "checkpoints" / "latest",
                repo_url,
            )
        else:
            logger.info(
                "Download all available checkpoints, if they exist, from the huggingface hub 👨🏻‍💻"
            )

            ckpt_folderpath = snapshot_download(
                repo_id=repo_path,
                cache_dir=pathlib.Path(cfg.hf_repo_dir),
                resume_download=True,
            )
            latest_checkpoint = (
                pathlib.Path(cfg.hf_repo_dir) / "checkpoints" / "latest"
            )

            if pathlib.Path(
                pathlib.Path(cfg.hf_repo_dir) / "checkpoints"
            ).exists():
                pathlib.Path(
                    pathlib.Path(cfg.hf_repo_dir) / "checkpoints"
                ).mkdir(parents=True, exist_ok=True)

            shutil.copy(
                pathlib.Path(ckpt_folderpath), cfg.hf_repo_dir / "checkpoints"
            )

            if latest_checkpoint.exists():
                logger.info(
                    f"Downloaded checkpoint from huggingface hub to {latest_checkpoint}"
                )
            return cfg.hf_repo_dir / "checkpoints" / "latest"
        return None, repo_url

    except Exception as e:
        logger.exception(
            f"Could not download latest checkpoint from huggingface hub: {e}"
        )
        return None, repo_url


def upload_code_to_wandb(code_dir: Union[pathlib.Path, str]):
    if isinstance(code_dir, str):
        code_dir = pathlib.Path(code_dir)

    code = wandb.Artifact("project-source", type="code")

    for path in code_dir.resolve().rglob("*.py"):
        code.add_file(str(path), name=str(path.relative_to(code_dir)))

    wandb.log_artifact(code)


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    wandb_args = {
        key: value
        for key, value in cfg.wandb_args.items()
        if key != "_target_"
    }
    ckpt_path, repo_url = create_hf_model_repo_and_download_maybe(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_args["config"] = config_dict
    wandb_args["notes"] = repo_url
    wandb.init(**wandb_args)  # init wandb and log config

    upload_code_to_wandb(cfg.code_dir)  # log code to wandb
    print(pretty_config(cfg, resolve=True))

    set_seed(seed=cfg.seed)

    model_and_transform: ModelAndTransform = instantiate(cfg.model)
    model: nn.Module = model_and_transform.model
    transform: Callable = model_and_transform.transform

    dataset: Dataset = instantiate(cfg.dataset)
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["validation"]
    test_dataset: Dataset = dataset["validation"]

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
    )
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )
    test_dataloader = instantiate(
        cfg.dataloader,
        dataset=test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
    )

    params = (
        model.classifier.parameters()
        if cfg.freeze_backbone
        else model.parameters()
    )

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=params, _partial_=False
    )

    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler,
        optimizer=optimizer,
        t_initial=cfg.learner.train_iters,
        _partial_=False,
    )

    learner: Learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[
            ClassificationTrainer(
                optimizer=optimizer,
                scheduler=scheduler,
                experiment_tracker=wandb,
            )
        ],
        evaluators=[ClassificationEvaluator(experiment_tracker=wandb)],
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        callbacks=instantiate_callbacks(cfg.callbacks),
        resume=ckpt_path,
    )

    if cfg.train:
        learner.train()

    if cfg.test:
        learner.test(test_dataloaders=[test_dataloader])


if __name__ == "__main__":
    run()
