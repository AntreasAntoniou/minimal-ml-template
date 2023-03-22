import os

import torch
import torch
import datasets
from rich import print
from torch.nn import CrossEntropyLoss
from torchvision import transforms
import wandb

from mlproject.boilerplate import Learner
from mlproject.evaluators import ClassificationEvaluator
from mlproject.trainers import ClassificationTrainer
from mlproject.utils import get_logger

from evolution_augment import config
from evolution_augment.transforms import CutoutDefault
from evolution_augment.models import factory
from evolution_augment import optimizers


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example['img']))
        labels.append(example['label'])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {'pixel_values': pixel_values, 'labels': labels}


class Transforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        example['img'] = [self.transforms(example['img'][0])]
        return example


if __name__ == '__main__':
    EXPERIMENT_NAME = 'debug_cifar10'
    CURRENT_EXPERIMENT_DIR = os.path.join(os.sep, 'tmp', EXPERIMENT_NAME)

    DATASET = 'cifar10'
    _CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    _CIFAR_STD = (0.2023, 0.1994, 0.2010)
    TRAIN_SAMPLES = 50000
    VAL_SAMPLES = 5000

    LR = 0.1
    BATCH_SIZE = 256
    EPOCHS = 200
    TRAIN_STEPS = int(EPOCHS * TRAIN_SAMPLES / BATCH_SIZE)
    WARMUP_STEPS = int(5 * TRAIN_SAMPLES / BATCH_SIZE)
    WORKERS = 1

    logger = get_logger(__name__)

    train_transforms = Transforms(transforms=transforms.Compose([
        transforms.RandAugment(num_ops=3, magnitude=5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        CutoutDefault(16)
    ]))
    test_transforms = Transforms(transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ]))

    train_dataset = datasets.load_dataset(DATASET,
                                          split='train',
                                          cache_dir='/data')

    train_dataset = train_dataset.with_transform(train_transforms)

    test_dataset = datasets.load_dataset(DATASET,
                                         split='test',
                                         cache_dir='/data')
    test_dataset = test_dataset.with_transform(test_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   collate_fn=collate_fn,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=WORKERS,
                                                   pin_memory=True,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  collate_fn=collate_fn,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=WORKERS,
                                                  pin_memory=True,
                                                  drop_last=False)

    model = factory.ModelFactory.build(
        config.WideResNet(num_classes=10,
                          depth=28,
                          widen_factor=10,
                          dropout_rate=0.0))

    optimizer = optimizers.get_optimizer(params=model.parameters(),
                                         config=config.SGD(lr=LR,
                                                           weight_decay=0.0005,
                                                           momentum=0.9,
                                                           nesterov=True))
    lr_sched = optimizers.get_lr_scheduler(optimizer=optimizer,
                                           config=config.LRScheduler(
                                               name='cosine',
                                               num_training_steps=TRAIN_STEPS,
                                               num_warmup_steps=WARMUP_STEPS))
    criterion = CrossEntropyLoss()

    os.makedirs(CURRENT_EXPERIMENT_DIR, exist_ok=True)
    wandb.init(
        project=os.environ.get('WANDB_PROJECT', 'debug'),
        resume='allow',  # allow, True, False, must
        dir=CURRENT_EXPERIMENT_DIR,
        save_code=False)

    experiment = Learner(
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=CURRENT_EXPERIMENT_DIR,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=[test_dataloader],
        trainers=[
            ClassificationTrainer(optimizer=optimizer,
                                  scheduler=lr_sched,
                                  experiment_tracker=wandb)
        ],
        evaluators=[ClassificationEvaluator(experiment_tracker=wandb)],
        evaluate_every_n_steps=int(TRAIN_SAMPLES / BATCH_SIZE),
        checkpoint_every_n_steps=int(TRAIN_SAMPLES / BATCH_SIZE),
        checkpoint_after_validation=True,
        train_iters=TRAIN_STEPS,
        resume=True,
    )
    experiment.run()
