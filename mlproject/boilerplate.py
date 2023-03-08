import copy
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tabnanny import check
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlproject.callbacks import Callback, CallbackHandler, Interval
from mlproject.evaluators import ClassificationEvaluator, Evaluator
from mlproject.trainers import ClassificationTrainer, Trainer
from mlproject.utils import get_logger

logger = get_logger(__name__)


class Learner(nn.Module):
    def __init__(
        self,
        experiment_name: str,
        experiment_dir: Union[str, Path],
        model: torch.nn.Module,
        resume: Union[bool, str] = False,
        evaluate_every_n_steps: int = None,
        evaluate_every_n_epochs: int = None,
        checkpoint_every_n_steps: int = None,
        checkpoint_after_validation: bool = False,
        train_iters: int = None,
        train_epochs: int = None,
        train_dataloader: DataLoader = None,
        limit_train_iters: int = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        limit_val_iters: int = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
        trainers: Union[List[Trainer], Trainer] = None,
        evaluators: Union[List[Evaluator], Evaluator] = None,
        callbacks: Union[List[Callback], Callback] = None,
        print_model_parameters: bool = False,
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.experiment_dir = (
            experiment_dir
            if isinstance(experiment_dir, Path)
            else Path(experiment_dir)
        )
        self.background_threads = []
        self.checkpoints_dir = Path(self.experiment_dir / "checkpoints")

        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)

        if not self.checkpoints_dir.exists():
            self.checkpoints_dir.mkdir(parents=True)
        self.model = model
        self.evaluate_every_n_steps = evaluate_every_n_steps
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_after_validation = checkpoint_after_validation
        self.step_idx = 0
        self.epoch_idx = 0
        self.limit_train_iters = limit_train_iters
        self.limit_val_iters = limit_val_iters

        if train_iters is None and train_epochs is None:
            raise ValueError(
                "Either train_iters or train_epochs must be specified"
            )

        self.train_iters = train_iters
        self.train_epochs = (
            99999999 if train_iters is not None else train_epochs
        )

        self.train_dataloader = train_dataloader

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

        for name, params in self.model.named_parameters():
            logger.info(f"{name}, {params.shape}")

        self.callbacks = (
            [callbacks] if isinstance(callbacks, Callback) else callbacks
        )

        if self.callbacks is None:
            self.callbacks = []

        self.callback_handler = CallbackHandler(self.callbacks)

        self.callback_handler.on_init_start(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloaders=self.val_dataloaders,
            test_dataloaders=self.test_dataloaders,
        )

        self.resume = resume

        self.eval_mode = (
            Interval.STEP if self.evaluate_every_n_steps else Interval.EPOCH
        )

        if (
            evaluate_every_n_steps is not None
            and evaluate_every_n_epochs is not None
        ):
            raise ValueError(
                "You can only specify one of `evaluate_every_n_steps` and `evaluate_every_n_epochs`"
            )

        self.trainers = (
            [trainers] if isinstance(trainers, Trainer) else trainers
        )
        self.evaluators = (
            [evaluators] if isinstance(evaluators, Evaluator) else evaluators
        )

        self.callback_handler.on_init_end(
            experiment=self,
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloaders=self.val_dataloaders,
            test_dataloaders=self.test_dataloaders,
        )

        # use if you want to debug unused parameter errors in DDP
        self.accelerator = Accelerator(
            # kwargs_handlers=[
            #     DistributedDataParallelKwargs(find_unused_parameters=True)
            # ]
        )

        self.model = self.model.to(self.accelerator.device)
        self.model, self.train_dataloader = self.accelerator.prepare(
            self.model, self.train_dataloader
        )

        for trainer in self.trainers:
            trainer.optimizer = self.accelerator.prepare(
                trainer.get_optimizer()
            )
            if trainer.scheduler is not None:
                trainer.scheduler = self.accelerator.prepare(trainer.scheduler)

        if self.val_dataloaders is not None:
            for i in range(len(self.val_dataloaders)):
                self.val_dataloaders[i] = self.accelerator.prepare(
                    self.val_dataloaders[i]
                )

        if self.test_dataloaders is not None:
            for i in range(len(self.test_dataloaders)):
                self.test_dataloaders[i] = self.accelerator.prepare(
                    self.test_dataloaders[i]
                )

        if isinstance(resume, str):
            checkpoint_path = Path(resume)
            if not checkpoint_path.exists():
                raise ValueError(
                    f"Checkpoint path {checkpoint_path} does not exist, please check your resume= argument"
                )
            self.load_checkpoint(checkpoint_path=checkpoint_path)

        elif isinstance(resume, Path):
            self.load_checkpoint(checkpoint_path=resume)

        elif resume is True:
            checkpoint_path = Path(self.checkpoints_dir / "latest")
            if not checkpoint_path.exists():
                logger.info(
                    f"Checkpoint path {checkpoint_path} does not exist, "
                    "starting from scratch :start:"
                )
            else:
                self.load_checkpoint(checkpoint_path=checkpoint_path)

        if print_model_parameters:
            for key, value in self.named_parameters():
                logger.info(
                    f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
                )

    def run(self):
        self.train()

    def forward(self, x):
        return self.model(x, accelerator=self.accelerator)

    def __repr__(self):
        attributes = "\n".join(
            [f"{key}={value}" for key, value in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}\n {attributes}"

    def __str__(self):
        return self.__repr__()

    def training_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_training_step_start(model, batch, batch_idx)

        for trainer in self.trainers:
            trainer.training_step(
                model=model,
                batch=batch,
                batch_idx=batch_idx,
                step_idx=self.step_idx,
                epoch_idx=self.epoch_idx,
                accelerator=self.accelerator,
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_training_step_end(model, batch, batch_idx)

    def validation_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_validation_step_start(model, batch, batch_idx)

        for evaluator in self.evaluators:
            evaluator.validation_step(
                model=model,
                batch=batch,
                batch_idx=batch_idx,
                step_idx=self.step_idx,
                epoch_idx=self.epoch_idx,
                accelerator=self.accelerator,
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_validation_step_end(model, batch, batch_idx)

    def testing_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_testing_step_start(model, batch, batch_idx)

        for evaluator in self.evaluators:
            evaluator.testing_step(
                model=model,
                batch=batch,
                batch_idx=batch_idx,
                step_idx=self.step_idx,
                epoch_idx=self.epoch_idx,
                accelerator=self.accelerator,
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_testing_step_end(model, batch, batch_idx)

    def start_training(self, train_dataloader: DataLoader):
        self.callback_handler.on_train_start(
            experiment=self,
            model=self.model,
            train_dataloader=train_dataloader,
        )

        for trainer in self.trainers:
            trainer.start_training(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                train_dataloader=train_dataloader,
            )

        logger.info("Starting training 🏋🏽")

    def end_training(self, train_dataloader: DataLoader):
        self.callback_handler.on_train_end(
            experiment=self,
            model=self.model,
            train_dataloader=train_dataloader,
        )

        for trainer in self.trainers:
            trainer.end_training(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                train_dataloader=train_dataloader,
            )

        for background_thread in self.background_threads:
            background_thread.join()

        logger.info("Training finished 🎉")

    def start_validation(self, val_dataloaders: List[DataLoader]):
        self.callback_handler.on_validation_start(
            experiment=self, model=self.model, val_dataloaders=val_dataloaders
        )

        for evaluator in self.evaluators:
            evaluator.start_validation(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                val_dataloaders=val_dataloaders,
            )

        logger.info("Starting validation 🧪")

    def end_validation(self, val_dataloaders: List[DataLoader]):
        self.callback_handler.on_validation_end(
            experiment=self, model=self.model, val_dataloaders=val_dataloaders
        )

        for evaluator in self.evaluators:
            evaluator.end_validation(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                val_dataloaders=val_dataloaders,
            )

        logger.info("Validation finished 🎉")

    def start_testing(self, test_dataloaders: List[DataLoader]):
        self.callback_handler.on_testing_start(
            experiment=self,
            model=self.model,
            test_dataloaders=test_dataloaders,
        )

        for evaluator in self.evaluators:
            evaluator.start_testing(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                test_dataloaders=test_dataloaders,
            )

        logger.info("Starting testing 🧪")

    def end_testing(self, test_dataloaders: List[DataLoader]):
        self.callback_handler.on_testing_end(
            experiment=self,
            model=self.model,
            test_dataloaders=test_dataloaders,
        )

        for evaluator in self.evaluators:
            evaluator.end_testing(
                epoch_idx=self.epoch_idx,
                step_idx=self.step_idx,
                test_dataloaders=test_dataloaders,
            )

        logger.info("Testing finished 🎉")

    def train(self, train_dataloader: DataLoader = None):
        self._training_loop(train_dataloader=train_dataloader)

    def validate(self, val_dataloaders: List[DataLoader] = None):
        self._validation_loop(val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders: List[DataLoader] = None):
        self._testing_loop(test_dataloaders=test_dataloaders)

    def _validation_loop(self, val_dataloaders: List[DataLoader] = None):
        if val_dataloaders is None:
            val_dataloaders = self.val_dataloaders

        if val_dataloaders is not None:
            self.start_validation(val_dataloaders=val_dataloaders)

            with tqdm(total=len(val_dataloaders)) as pbar_dataloaders:
                for val_dataloader in val_dataloaders:
                    with tqdm(total=len(val_dataloader)) as pbar:
                        for batch_idx, batch in enumerate(val_dataloader):
                            if self.limit_val_iters is not None:
                                if batch_idx >= self.limit_val_iters:
                                    break
                            self.validation_step(
                                model=self.model,
                                batch=batch,
                                batch_idx=batch_idx,
                            )
                            pbar.update(1)
                    pbar_dataloaders.update(1)

            self.end_validation(val_dataloaders=val_dataloaders)

    def _testing_loop(self, test_dataloaders: List[DataLoader] = None):
        if test_dataloaders is None:
            test_dataloaders = self.test_dataloaders

        if test_dataloader is not None:
            self.start_testing(test_dataloaders=test_dataloaders)

            with tqdm(total=len(test_dataloaders)) as pbar_dataloaders:
                for test_dataloader in test_dataloaders:
                    with tqdm(total=len(test_dataloader)) as pbar:
                        for batch_idx, batch in enumerate(test_dataloader):
                            self._testing_loop(
                                model=self.model,
                                batch=batch,
                                batch_idx=batch_idx,
                            )
                            pbar.update(1)
                    pbar_dataloaders.update(1)

            self.end_testing(test_dataloaders=test_dataloaders)

    def _training_loop(self, train_dataloader: DataLoader = None):
        # sourcery skip: extract-method

        if train_dataloader is None:
            train_dataloader = self.train_dataloader

        if train_dataloader is not None:
            self.start_training(train_dataloader=train_dataloader)
            with tqdm(
                initial=self.step_idx, total=self.train_iters
            ) as pbar_steps:
                for epoch_idx in range(self.epoch_idx, self.train_epochs):
                    self.epoch_idx = epoch_idx

                    if self.limit_train_iters is not None:
                        if self.step_idx >= self.limit_train_iters:
                            break

                    if (
                        self.eval_mode == Interval.EPOCH
                        and epoch_idx % self.evaluate_every_n_epochs == 0
                    ):
                        self._validation_loop()

                    for batch_idx, batch in enumerate(train_dataloader):
                        self.training_step(
                            model=self.model, batch=batch, batch_idx=batch_idx
                        )

                        if (
                            self.eval_mode == Interval.STEP
                            and self.step_idx % self.evaluate_every_n_steps
                            == 0
                        ):
                            self._validation_loop()

                        if (
                            self.step_idx % self.checkpoint_every_n_steps == 0
                            and self.step_idx > 0
                        ):
                            self.save_checkpoint(
                                checkpoint_name=f"ckpt_{self.step_idx}"
                            )
                            self.save_checkpoint(checkpoint_name="latest")

                        self.step_idx += 1

                        if self.step_idx >= self.train_iters:
                            return self.end_training(
                                train_dataloader=train_dataloader
                            )

                        pbar_steps.update(1)

            self.end_training(train_dataloader=train_dataloader)

    def save_checkpoint(
        self,
        checkpoint_name: str,
    ):
        ckpt_save_path = self.checkpoints_dir / checkpoint_name

        if not ckpt_save_path.exists():
            ckpt_save_path.mkdir(parents=True)

        experiment_hyperparameters = dict(
            step_idx=self.step_idx, epoch_idx=self.epoch_idx
        )
        torch.save(
            obj=experiment_hyperparameters,
            f=ckpt_save_path / "trainer_state.pt",
        )
        self.accelerator.save_state(ckpt_save_path)

        self.callback_handler.on_save_checkpoint(
            model=self.model,
            optimizers=[trainer.optimizer for trainer in self.trainers],
            experiment=self,
            checkpoint_path=ckpt_save_path,
        )

        return ckpt_save_path

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ):
        checkpoint_path = (
            checkpoint_path
            if isinstance(checkpoint_path, Path)
            else Path(checkpoint_path)
        )
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        trainer_state = torch.load(
            pathlib.Path(checkpoint_path) / "trainer_state.pt"
        )
        self.step_idx = trainer_state["step_idx"]
        self.epoch_idx = trainer_state["epoch_idx"]

        self.accelerator.load_state(checkpoint_path)

        self.callback_handler.on_load_checkpoint(
            model=self.model,
            optimizers=[trainer.get_optimizer() for trainer in self.trainers],
            experiment=self,
            checkpoint_path=checkpoint_path,
        )


if __name__ == "__main__":
    # a minimal example of how to use the Learner class
    import torch
    from datasets import Image, load_dataset
    from rich import print
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import ColorJitter, Compose, Resize, ToTensor

    train_dataset = load_dataset("beans", split="train")
    val_dataset = load_dataset("beans", split="validation")
    test_dataset = load_dataset("beans", split="test")

    jitter = Compose(
        [
            Resize(size=(96, 96)),
            ColorJitter(brightness=0.5, hue=0.5),
            ToTensor(),
        ]
    )

    def transforms(examples):
        examples["pixel_values"] = [
            jitter(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    train_dataset = train_dataset.with_transform(transforms)
    val_dataset = val_dataset.with_transform(transforms)
    test_dataset = test_dataset.with_transform(transforms)

    def collate_fn(examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["pixel_values"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=256,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=256, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=256, num_workers=4
    )

    model = torch.hub.load(
        "pytorch/vision:v0.9.0", "resnet18", pretrained=False
    )
    model.fc = torch.nn.Linear(512, 4)

    optimizer = Adam(model.parameters(), lr=1e-3)

    criterion = CrossEntropyLoss()

    experiment = Learner(
        experiment_name="debug_checkpointing",
        experiment_dir="/exp/debug_checkpointing",
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        test_dataloaders=[test_dataloader],
        trainers=[ClassificationTrainer(optimizer=optimizer)],
        evaluators=[ClassificationEvaluator()],
        evaluate_every_n_steps=5,
        checkpoint_every_n_steps=5,
        checkpoint_after_validation=True,
        train_iters=1000,
        resume=True,
    )

    experiment.run()
