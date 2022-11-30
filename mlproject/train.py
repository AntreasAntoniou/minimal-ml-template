from gc import callbacks
import os

import dotenv
import wandb
from rich import print
from rich.traceback import install

from mlproject.boilerplate import Learner
from mlproject.callbacks import Callback
from mlproject.evaluators import ClassificationEvaluator
from mlproject.trainers import ClassificationTrainer

os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups

install()  # beautiful and clean tracebacks for debugging
dotenv.load_dotenv(override=True, verbose=True)


import pathlib
from typing import Callable, List, Optional

import dotenv
import hydra
import torch
from huggingface_hub import Repository, create_repo, snapshot_download, hf_hub_download
from hydra_zen import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from mlproject.config import BaseConfig, collect_config_store
from mlproject.models import ModelAndTransform
from mlproject.utils import get_logger, pretty_config

config_store = collect_config_store()

logger = get_logger(name=__name__)


def instantiate_callbacks(callback_dict: dict) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        callbacks.append(instantiate(cb_conf))

    return callbacks


def create_hf_model_repo_and_download_maybe(
    cfg: BaseConfig, download_only_latest: bool = True
):
    repo_path = cfg.repo_path

    create_repo(repo_path, repo_type="model", exist_ok=True)
    logger.info(f"Created repo {repo_path}, {cfg.hf_repo_dir}")

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

    download_success = False

    try:
        if download_only_latest:
            hf_hub_download(
                repo_id=cfg.exp_name,
                cache_dir=hf_repo,
                resume_download=True,
                filename="checkpoints/latest.pt",
            )
            logger.info(f"Downloaded checkpoint from huggingface hub to {hf_repo}")
            download_success = True
        else:
            snapshot_download(
                repo_id=cfg.exp_name,
                cache_dir=hf_repo,
                resume_download=True,
            )
            latest_checkpoint = hf_repo / "checkpoints" / "latest.pt"
            if latest_checkpoint.exists():
                logger.info(
                    f"Downloaded checkpoint from huggingface hub to {latest_checkpoint}"
                )
                download_success = True
            else:
                download_success = False
            return repo, download_success
    except Exception as e:
        logger.exception(
            f"Could not download checkpoint_latest.pt from huggingface hub: {e}"
        )
        return repo, download_success


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    print(pretty_config(cfg, resolve=True))

    repo, download_success = create_hf_model_repo_and_download_maybe(cfg)

    model_and_transform: ModelAndTransform = instantiate(cfg.model)
    model: nn.Module = model_and_transform.model
    transform: Callable = model_and_transform.transform

    dataset: Dataset = instantiate(cfg.dataset)
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["validation"]

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)

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

    wandb_args = {
        key: value for key, value in cfg.wandb_args.items() if key != "_target_"
    }

    wandb.init(**wandb_args)

    params = (
        model.classifier.parameters() if cfg.freeze_backbone else model.parameters()
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

    learner = instantiate(
        cfg.learner,
        model=model,
        trainers=[ClassificationTrainer(optimizer=optimizer, scheduler=scheduler)],
        evaluators=[ClassificationEvaluator()],
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        callbacks=instantiate_callbacks(cfg.callbacks),
    )

    learner.train()


if __name__ == "__main__":
    run()
