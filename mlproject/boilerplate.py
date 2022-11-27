from dataclasses import dataclass
from pathlib import Path
from tabnanny import check
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from mlproject.callbacks import Callback, Interval
from mlproject.trainers import ClassificationTrainer, Trainer
from mlproject.evaluators import ClassificationEvaluator, Evaluator
from mlproject.utils import get_logger

logger = get_logger(__name__, set_default_rich_handler=True)


class CallbackHandler(Callback):
    def __init__(self, callbacks: List[Callback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def on_init_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_start(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_init_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_init_end(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_epoch_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_epoch_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_batch_start(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_start(model, batch, batch_idx)

    def on_batch_end(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(model, batch, batch_idx)

    def on_training_step_start(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        for callback in self.callbacks:
            callback.on_training_step_start(model, batch, batch_idx)

    def on_training_step_end(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        for callback in self.callbacks:
            callback.on_training_step_stop(model, batch, batch_idx)

    def on_validation_step_start(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_start(model, batch, batch_idx)

    def on_validation_step_end(
        self, model: nn.Module, batch: Dict, batch_idx: int
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_step_stop(model, batch, batch_idx)

    def on_test_step_start(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_test_step_start(model, batch, batch_idx)

    def on_test_step_end(self, model: nn.Module, batch: Dict, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_test_step_stop(model, batch, batch_idx)

    def on_train_start(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_start(experiment, model, train_dataloader)

    def on_train_end(
        self,
        experiment: Any,
        model: nn.Module,
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_end(
                experiment,
                model,
                train_dataloader,
                val_dataloaders,
                test_dataloaders,
            )

    def on_validation_start(
        self,
        experiment: Any,
        model: nn.Module,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ):
        for callback in self.callbacks:
            callback.on_validation_start(experiment, model, val_dataloaders)

    def on_validation_end(
        self,
        experiment: Any,
        model: nn.Module,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(experiment, model, val_dataloaders)

    def on_test_start(
        self,
        experiment: Any,
        model: nn.Module,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_test_start(experiment, model, test_dataloaders)

    def on_test_end(
        self,
        experiment: Any,
        model: nn.Module,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_test_end(experiment, model, test_dataloaders)

    def on_save_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_save_checkpoint(model, optimizers, experiment, checkpoint_path)

    def on_load_checkpoint(
        self,
        model: nn.Module,
        optimizers: List[torch.optim.Optimizer],
        experiment: Any,
        checkpoint_path: Path,
    ) -> None:
        for callback in self.callbacks:
            callback.on_load_checkpoint(model, optimizers, experiment, checkpoint_path)


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
        train_dataloader: DataLoader = None,
        val_dataloaders: Union[List[DataLoader], DataLoader] = None,
        test_dataloaders: Union[List[DataLoader], DataLoader] = None,
        trainers: Union[List[Trainer], Trainer] = None,
        evaluators: Union[List[Evaluator], Evaluator] = None,
        callbacks: Union[List[Callback], Callback] = None,
    ):
        super().__init__()
        wandb.init(project=experiment_name, dir=experiment_dir)
        self.experiment_name = experiment_name
        self.experiment_dir = (
            experiment_dir if isinstance(experiment_dir, Path) else Path(experiment_dir)
        )
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

        if train_iters is None and train_epochs is None:
            raise ValueError("Either train_iters or train_epochs must be specified")

        self.train_iters = train_iters
        self.train_epochs = 99999999 if train_iters is not None else train_epochs

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

        self.callbacks = [callbacks] if isinstance(callbacks, Callback) else callbacks

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

        self.eval_mode = (
            Interval.STEP if self.evaluate_every_n_steps else Interval.EPOCH
        )

        if evaluate_every_n_steps is not None and evaluate_every_n_epochs is not None:
            raise ValueError(
                "You can only specify one of `evaluate_every_n_steps` and `evaluate_every_n_epochs`"
            )

        self.trainers = [trainers] if isinstance(trainers, Trainer) else trainers
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

        for key, value in self.named_parameters():
            logger.info(
                f"Parameter {key} -> {value.shape} requires grad {value.requires_grad}"
            )

    def run(self):
        self.train()

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(experiment_name={self.experiment_name}, "
            f"experiment_dir={self.experiment_dir}, "
            f"model={self.model}, "
            f"evaluate_every_n_steps={self.evaluate_every_n_steps}, "
            f"evaluate_every_n_epochs={self.evaluate_every_n_epochs}, "
            f"checkpoint_every_n_steps={self.checkpoint_every_n_steps}, "
            f"checkpoint_after_validation={self.checkpoint_after_validation}, "
            f"train_iters={self.train_iters}, "
            f"train_epochs={self.train_epochs}, "
            f"train_dataloader={self.train_dataloader}, "
            f"val_dataloaders={self.val_dataloaders}, "
            f"test_dataloaders={self.test_dataloaders}, "
            f"trainers={self.trainers}, "
        )

    def __str__(self):
        return self.__repr__()

    def training_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_training_step_start(model, batch, batch_idx)

        for trainer in self.trainers:
            trainer.training_step(
                model=model, batch=batch, batch_idx=batch_idx, step_idx=self.step_idx
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_training_step_end(model, batch, batch_idx)

    def validation_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_validation_step_start(model, batch, batch_idx)

        for evaluator in self.evaluators:
            evaluator.validation_step(
                model=model, batch=batch, batch_idx=batch_idx, step_idx=self.step_idx
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_validation_step_end(model, batch, batch_idx)

    def testing_step(self, model, batch, batch_idx):
        self.callback_handler.on_batch_start(model, batch, batch_idx)
        self.callback_handler.on_test_step_start(model, batch, batch_idx)

        for evaluator in self.evaluators:
            evaluator.testing_step(
                model=model, batch=batch, batch_idx=batch_idx, step_idx=self.step_idx
            )

        self.callback_handler.on_batch_end(model, batch, batch_idx)
        self.callback_handler.on_test_step_end(model, batch, batch_idx)

    def start_training(self, train_dataloader: DataLoader):
        self.callback_handler.on_train_start(
            experiment=self, model=self.model, train_dataloader=train_dataloader
        )
        print("Starting training... :runner:")

    def end_training(self, train_dataloader: DataLoader):
        self.callback_handler.on_train_end(
            experiment=self, model=self.model, train_dataloader=train_dataloader
        )
        print("Training finished :tada:")

    def start_validation(self, val_dataloaders: List[DataLoader]):
        self.callback_handler.on_validation_start(
            experiment=self, model=self.model, val_dataloaders=val_dataloaders
        )

        print("Starting validation...")

    def end_validation(self, val_dataloaders: List[DataLoader]):
        self.callback_handler.on_validation_end(
            experiment=self, model=self.model, val_dataloaders=val_dataloaders
        )

        print("Validation finished :tada:")

    def start_testing(self, test_dataloaders: List[DataLoader]):
        self.callback_handler.on_test_start(
            experiment=self, model=self.model, test_dataloaders=test_dataloaders
        )

        print("Starting testing...")

    def end_testing(self, test_dataloaders: List[DataLoader]):
        self.callback_handler.on_test_end(
            experiment=self, model=self.model, test_dataloaders=test_dataloaders
        )

        print("Testing finished :tada:")

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
                            self.validation_step(
                                model=self.model, batch=batch, batch_idx=batch_idx
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
                                model=self.model, batch=batch, batch_idx=batch_idx
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
            with tqdm(total=self.train_iters) as pbar_steps:
                for epoch_idx in range(self.epoch_idx, self.train_epochs):
                    self.epoch_idx = epoch_idx

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
                            and self.step_idx % self.evaluate_every_n_steps == 0
                        ):
                            self._validation_loop()

                        if self.step_idx % self.checkpoint_every_n_steps == 0:
                            self.save_checkpoint()

                        self.step_idx += 1

                        if self.step_idx >= self.train_iters:
                            self.end_training()

                        pbar_steps.update(1)

            self.end_training(train_dataloader=train_dataloader)

    def save_checkpoint(self):

        experiment_hyperparameters = dict(
            step_idx=self.step_idx, epoch_idx=self.epoch_idx
        )

        optimizers: List[torch.optim.Optimizer] = [
            list(trainer.get_optimizer())
            if isinstance(trainer.get_optimizer(), List)
            else trainer.get_optimizer()
            for trainer in self.trainers
        ]

        optimizer_states = []
        for item in optimizers:
            if isinstance(item, List):
                optimizer_states.append([optimizer.state_dict() for optimizer in item])
            else:
                optimizer_states.append(item.state_dict())

        model = self.model.state_dict()

        state = dict(
            exp=experiment_hyperparameters, optimizers=optimizer_states, model=model
        )
        ckpt_save_path = self.experiment_dir / f"{self.experiment_name}.pt"
        torch.save(obj=state, f=ckpt_save_path)

        self.callback_handler.on_save_checkpoint(
            model=model,
            optimizers=optimizer_states,
            experiment=self,
            checkpoint_path=ckpt_save_path,
        )

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        checkpoint_path = (
            checkpoint_path
            if isinstance(checkpoint_path, Path)
            else Path(checkpoint_path)
        )
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.step_idx = state["exp_state"]["step_idx"]
        self.epoch_idx = state["exp_state"]["epoch_idx"]

        for idx, (trainer, optimizer_state) in enumerate(
            zip(self.trainers, state["optimizers"])
        ):
            if isinstance(optimizer_state, List):
                for optimizer, state in zip(trainer.get_optimizer(), optimizer_state):
                    optimizer.load_state_dict(state)
            else:
                trainer.get_optimizer().load_state_dict(optimizer_state)

        self.callback_handler.on_load_checkpoint(
            model=self.model,
            optimizers=[trainer.get_optimizer() for trainer in self.trainers],
            experiment=self,
            checkpoint_path=checkpoint_path,
        )


# Ensure continued experiments work properly, especially dataloaders continuing from the right step_idx
# Add load checkpoint functionality
# Add validation and testing loops
# Add explicit call for validation and testing without training
# Add callback functionality for training, validation, testing, and checkpointing
# Build a standard Trainer and Evaluator for classification
if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose, ColorJitter, Resize, ToTensor
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn import CrossEntropyLoss

    from datasets import load_dataset, Image
    from rich import print
    import torch

    train_dataset = load_dataset("beans", split="train")
    val_dataset = load_dataset("beans", split="validation")
    test_dataset = load_dataset("beans", split="test")

    jitter = Compose(
        [Resize(size=(96, 96)), ColorJitter(brightness=0.5, hue=0.5), ToTensor()]
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

    model = torch.hub.load("pytorch/vision:v0.9.0", "resnet18", pretrained=False)
    model.fc = torch.nn.Linear(512, 4)

    optimizer = Adam(model.parameters(), lr=1e-3)

    criterion = CrossEntropyLoss()

    experiment = Experiment(
        experiment_name="debug",
        experiment_dir="debug_folder",
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        test_dataloaders=[test_dataloader],
        trainers=[ClassificationTrainer(optimizer=optimizer)],
        evaluators=[ClassificationEvaluator()],
        evaluate_every_n_steps=250,
        checkpoint_every_n_steps=250,
        checkpoint_after_validation=True,
        train_iters=1000,
    )

    experiment.run()
