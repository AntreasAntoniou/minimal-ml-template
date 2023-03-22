import dataclasses
from typing import Any, Callable, List, Optional, Tuple, Union

from torchvision import transforms

from evolution_augment import constants
from mlproject import callbacks


@dataclasses.dataclass
class Dataset:
    name: str = ''
    num_classes: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0
    image_means: List[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    image_stds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 1.0])


@dataclasses.dataclass
class Model:
    num_classes: int = 100


@dataclasses.dataclass
class WideResNet(Model):
    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.0


@dataclasses.dataclass
class EvolutionAugment:
    augmentation_ops: Union[List, Tuple] = dataclasses.field(
        default_factory=lambda: constants.AUGMENTATION_OPS)
    num_candidates: int = 20
    min_augmentations: int = 1
    max_augmentations: int = 5
    max_magnitude: int = 10
    num_magnitude_bins: int = 31
    interpolation: transforms.InterpolationMode = transforms.InterpolationMode.NEAREST
    normalize: bool = True


@dataclasses.dataclass
class Dataloader:
    dataset: Dataset = Dataset()
    batch_size: int = 8
    evolution_augment_config: Optional[EvolutionAugment] = None
    is_training: bool = False
    num_workers: int = 1


@dataclasses.dataclass
class Optimizer:
    lr: float = 0.001


@dataclasses.dataclass
class SGD(Optimizer):
    momentum: float = 0.0
    weight_decay: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False


@dataclasses.dataclass
class LRScheduler:
    name: str = 'cosine'
    num_training_steps: int = 0
    num_warmup_steps: int = 0


@dataclasses.dataclass
class ClassificationTrainer:
    optimizer: Optimizer = Optimizer()
    scheduler: LRScheduler = LRScheduler()
    scheduler_interval: str = callbacks.Interval.STEP


@dataclasses.dataclass
class Experiment:
    experiment_name: str = "debug"
    experiment_dir: str = "/exp/debug_checkpointing"
    model: Model = WideResNet()
    train_dataloader: Dataloader = Dataloader(is_training=True)
    val_dataloaders: Dataloader = Dataloader(is_training=False)
    test_dataloaders: Optional[Dataloader] = None
    trainers: ClassificationTrainer = ClassificationTrainer(optimizer=SGD())
    evaluate_every_n_steps: int = 5
    checkpoint_every_n_steps: int = 5
    checkpoint_after_validation: bool = True
    train_iters: int = 1000
    resume: bool = True


def wideresnet28x18_cifar100():
    dataset = Dataset(name='cifar100',
                      num_classes=100,
                      num_train_samples=50000,
                      num_val_samples=5000,
                      image_means=[0.5071, 0.4866, 0.4409],
                      image_stds=[0.2009, 0.1984, 0.2023])
    train_batch_size = 256
    val_batch_size = 256

    epochs = 200
    train_iters = int(epochs * dataset.num_train_samples / train_batch_size)
    warmup_iters = int(5 * dataset.num_train_samples / train_batch_size)

    config = Experiment(
        experiment_name='wideresnet28x18_cifar100_gpu',
        experiment_dir='/exps/wideresnet28x18_cifar100_gpu',
        model=WideResNet(num_classes=dataset.num_classes,
                         depth=28,
                         widen_factor=10,
                         dropout_rate=0.0),
        train_dataloader=Dataloader(
            dataset=dataset,
            batch_size=train_batch_size,
            evolution_augment_config=EvolutionAugment(
                augmentation_ops=constants.AUGMENTATION_OPS,
                num_candidates=4,
                min_augmentations=2,
                max_augmentations=6,
                max_magnitude=5,
                num_magnitude_bins=31,
                interpolation='bilinear',
                normalize=True),
            num_workers=4),
        val_dataloaders=Dataloader(dataset=dataset,
                                   batch_size=val_batch_size,
                                   num_workers=4,
                                   is_training=False),
        trainers=ClassificationTrainer(
            optimizer=SGD(lr=0.1, weight_decay=0.0005, momentum=0.9, nesterov=True),
            scheduler=LRScheduler(name='cosine',
                                  num_training_steps=train_iters,
                                  num_warmup_steps=warmup_iters),
            scheduler_interval=callbacks.Interval.STEP),
        train_iters=train_iters,
        evaluate_every_n_steps=int(dataset.num_val_samples / val_batch_size),
        checkpoint_every_n_steps=500,
        checkpoint_after_validation=True)

    return config
