import os
from mlproject.lightning import TrainingEvaluationAgent
from rich import print
from rich.traceback import install
import dotenv

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
from huggingface_hub import Repository, create_repo, hf_hub_download
from hydra_zen import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from torch import nn
from torch.utils.data import Dataset

from mlproject.config import BaseConfig, collect_config_store
from mlproject.models import ModelAndTransform
from mlproject.utils import get_logger, pretty_config

config_store = collect_config_store()

logger = get_logger(name=__name__)


def instantiate_callbacks(callback_dict: dict, config: BaseConfig) -> List[Callback]:
    callbacks = []
    for cb_conf in callback_dict.values():
        if "LogConfigInformation" in cb_conf["_target_"]:
            callbacks.append(
                instantiate(
                    config=cb_conf,
                    exp_config=OmegaConf.to_container(config, resolve=True),
                    _recursive_=False,
                )
            )

        else:
            callbacks.append(instantiate(cb_conf))

    return callbacks


def create_hf_model_repo_and_download_maybe(cfg: BaseConfig):
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
        hf_hub_download(
            repo_id=cfg.exp_name,
            cache_dir=hf_repo,
            resume_download=True,
            filename="last.ckpt",
        )
        download_success = True
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
    test_dataset: Dataset = dataset["validation"]

    train_dataset.set_transform(transform)
    val_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    train_dataloader = instantiate(cfg.dataloader, dataset=train_dataset, shuffle=True)
    val_dataloader = instantiate(cfg.dataloader, dataset=val_dataset, shuffle=False)
    test_dataloader = instantiate(cfg.dataloader, dataset=test_dataset, shuffle=False)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.optimizer, params=model.parameters(), _partial_=False
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
        cfg.scheduler, optimizer=optimizer, _partial_=False
    )
    callbacks: List[Callback] = instantiate_callbacks(
        callback_dict=cfg.callbacks, config=cfg
    )

    loggers: List[LightningLoggerBase] = [
        instantiate(value) for key, value in cfg.loggers.items()
    ]

    lightning_model = TrainingEvaluationAgent(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        fine_tunable=cfg.fine_tunable,
    )

    trainer: Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
        _convert_="partial",
    )

    checkpoint_filepath = None

    if cfg.resume:
        checkpoint_filepath = pathlib.Path(repo.local_dir / "last.ckpt")

        if checkpoint_filepath.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_filepath}")
        else:
            checkpoint_filepath = None

    train_results = trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_filepath,
    )
    test_results = trainer.test(
        model=lightning_model,
        data_loaders=test_dataloader,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )


if __name__ == "__main__":
    run()
