import os

import dataclasses
from typing import List, Optional, Tuple, Union

from torchvision import transforms

from evolution_augment import constants
from mlproject import callbacks


@dataclasses.dataclass
class AugmentationOp:
    name: str = ''


@dataclasses.dataclass
class RandomCrop(AugmentationOp):
    size: int = 0
    padding: int = 0


@dataclasses.dataclass
class CutoutDefault(AugmentationOp):
    length: int = 0


@dataclasses.dataclass
class RandAugment(AugmentationOp):
    num_ops: int = 0
    magnitude: int = 0


@dataclasses.dataclass
class EvolutionAugment(AugmentationOp):
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
class Augmentation:
    rand_augment: Optional[RandAugment] = None
    evolution_augment: Optional[EvolutionAugment] = None
    cutout: Optional[CutoutDefault] = None
    random_crop: Optional[RandomCrop] = None


@dataclasses.dataclass
class Dataset:
    name: str = ''
    image_key: str = 'img'
    label_key: str = 'label'
    num_classes: int = 0
    num_train_samples: int = 0
    num_val_samples: int = 0
    image_means: List[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0])
    image_stds: List[float] = dataclasses.field(
        default_factory=lambda: [1.0, 1.0, 1.0])


@dataclasses.dataclass
class Dataloader:
    dataset: Dataset = Dataset()
    batch_size: int = 0
    output_size: Optional[List[int]] = None
    augmentation: Optional[Augmentation] = None
    is_training: bool = False
    num_workers: int = 1


@dataclasses.dataclass
class Model:
    num_classes: int = 100


@dataclasses.dataclass
class WideResNet(Model):
    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.0


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
class ExperimentTracker:
    project: str = ''


@dataclasses.dataclass
class WandBTracker:
    resume: str = 'allow',  # allow, True, False, must
    dir: str = '/tmp'
    save_code: bool = False


@dataclasses.dataclass
class ClassificationTrainer:
    optimizer: Optimizer = Optimizer()
    scheduler: LRScheduler = LRScheduler()
    scheduler_interval: str = callbacks.Interval.STEP
    experiment_tracker: ExperimentTracker = WandBTracker()


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


def wideresnet28x18_randaugment_cifar10_bs256_ep200():
    dataset = Dataset(name='cifar10',
                      num_classes=10,
                      num_train_samples=50000,
                      num_val_samples=5000,
                      image_means=(0.4914, 0.4822, 0.4465),
                      image_stds=(0.2023, 0.1994, 0.2010))
    train_batch_size = 256
    val_batch_size = 256

    epochs = 200
    iters_per_epoch = int(dataset.num_train_samples / train_batch_size)
    train_iters = int(epochs * dataset.num_train_samples / train_batch_size)
    warmup_iters = int(5 * dataset.num_train_samples / train_batch_size)

    config = Experiment(
        experiment_name='wideresnet28x18_randaugment_cifar10_bs256_ep200',
        experiment_dir='/home/exps/wideresnet28x18_randaugment_cifar10_bs256_ep200',
        model=WideResNet(num_classes=dataset.num_classes,
                         depth=28,
                         widen_factor=10,
                         dropout_rate=0.0),
        train_dataloader=Dataloader(
            dataset=dataset,
            batch_size=train_batch_size,
            output_size=(32, 32),
            augmentation=Augmentation(
                random_crop=RandomCrop(size=32, padding=4),
                cutout=CutoutDefault(length=16),
                rand_augment=RandAugment(num_ops=3, magnitude=5)),
            is_training=True,
            num_workers=os.cpu_count() // 2),
        val_dataloaders=Dataloader(
            dataset=dataset,
            batch_size=val_batch_size,
            output_size=(32, 32),
            num_workers=os.cpu_count() // 2,
            is_training=False),
        trainers=ClassificationTrainer(
            optimizer=SGD(lr=0.2,
                          weight_decay=0.0005,
                          momentum=0.9,
                          nesterov=True),
            scheduler=LRScheduler(name='cosine',
                                  num_training_steps=train_iters,
                                  num_warmup_steps=warmup_iters),
            scheduler_interval=callbacks.Interval.STEP),
        train_iters=train_iters,
        evaluate_every_n_steps=200,
        checkpoint_every_n_steps=int(train_iters / 10),
        checkpoint_after_validation=False)

    return config


def wideresnet28x18_randaugment_cifar100_bs256_ep200():
    dataset = Dataset(name='cifar100',
                      num_classes=100,
                      num_train_samples=50000,
                      num_val_samples=10000,
                      label_key='fine_label',
                      image_means=(0.4914, 0.4822, 0.4465),
                      image_stds=(0.2023, 0.1994, 0.2010))
    train_batch_size = 256
    val_batch_size = 256

    epochs = 200
    iters_per_epoch = int(dataset.num_train_samples / train_batch_size)
    train_iters = int(epochs * dataset.num_train_samples / train_batch_size)
    warmup_iters = int(5 * dataset.num_train_samples / train_batch_size)

    config = Experiment(
        experiment_name='wideresnet28x18_randaugment_cifar100_bs256_ep200',
        experiment_dir='/home/exps/wideresnet28x18_randaugment_cifar100_bs256_ep200',
        model=WideResNet(num_classes=dataset.num_classes,
                         depth=28,
                         widen_factor=10,
                         dropout_rate=0.0),
        train_dataloader=Dataloader(
            dataset=dataset,
            batch_size=train_batch_size,
            output_size=(32, 32),
            augmentation=Augmentation(
                random_crop=RandomCrop(size=32, padding=4),
                cutout=CutoutDefault(length=16),
                rand_augment=RandAugment(num_ops=2, magnitude=14)),
            is_training=True,
            num_workers=os.cpu_count() // 2),
        val_dataloaders=Dataloader(
            dataset=dataset,
            batch_size=val_batch_size,
            output_size=(32, 32),
            num_workers=os.cpu_count() // 2,
            is_training=False),
        trainers=ClassificationTrainer(
            optimizer=SGD(lr=0.2,
                          weight_decay=0.0005,
                          momentum=0.9,
                          nesterov=True),
            scheduler=LRScheduler(name='cosine',
                                  num_training_steps=train_iters,
                                  num_warmup_steps=warmup_iters),
            scheduler_interval=callbacks.Interval.STEP),
        train_iters=train_iters,
        evaluate_every_n_steps=200,
        checkpoint_every_n_steps=int(train_iters / 10),
        checkpoint_after_validation=False)

    return config