import os

import torch
import torch
import datasets
from torchvision import transforms

from mlproject.utils import get_logger

from evolution_augment.transforms import CutoutDefault
from evolution_augment.dataloaders import DataloaderFactory


class Transforms:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        example['img'] = [self.transforms(example['img'][0])]
        return example


class _CifarBase:

    def __init__(self, config: dict):
        '''
          config: A `dict` with following  keys `name`, `batch_size`,
            `workers`, `image_key`, `label_key`, `is_training`,
            `image_means`, `image_stds`
        '''
        self.config = config
        self.transforms = None

    def _collate_fn(self, examples):
        images = []
        labels = []
        for example in examples:
            images.append((example[self.config['image_key']]))
            labels.append(example[self.config['label_key']])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {'pixel_values': pixel_values, 'labels': labels}

    def _transform_example(self, example):
        example['img'] = [self.transforms(example['img'][0])]
        return example

    def build(self):
        is_training = self.config['is_training']

        if 'DATASETS_CACHE_DIR' not in os.environ:
            os.environ['DATASETS_CACHE_DIR'] = '/data'

        normalize = transforms.Normalize(self.config['image_means'],
                                         self.config['image_stds'])

        if is_training:
            print('Constructing transforms for train mode')
            aug_cfg = self.config['augmentation']
            self.transforms = transforms.Compose([
                transforms.RandAugment(
                    num_ops=aug_cfg['rand_augment']['num_ops'],
                    magnitude=aug_cfg['rand_augment']['magnitude']),
                transforms.RandomCrop(
                    size=aug_cfg['random_crop']['size'],
                    padding=aug_cfg['random_crop']['padding']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), normalize,
                CutoutDefault(length=aug_cfg['cutout']['length'])
            ])
        else:
            print('Constructing transforms for test mode')
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        dataset = datasets.load_dataset(
            self.config['name'],
            split='train' if is_training else 'test',
            cache_dir=os.getenv('DATASETS_CACHE_DIR'))
        dataset = dataset.with_transform(self._transform_example)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config['batch_size'],
            shuffle=True if is_training else False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True if is_training else False)
        return dataloader


@DataloaderFactory.register()
class cifar10(_CifarBase):

    def __init__(self, config):
        config['name'] = 'cifar10'
        super(cifar10, self).__init__(config=config)


@DataloaderFactory.register()
class cifar100(_CifarBase):

    def __init__(self, config):
        config['name'] = 'cifar100'
        super(cifar100, self).__init__(config=config)
