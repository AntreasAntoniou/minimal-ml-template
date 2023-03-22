import dataclasses

import datasets
import torch
from torchvision import transforms

from evolution_augment import constants
from evolution_augment.transforms import EvolutionAugment


def create_dataloader(name,
                      batch_size,
                      image_means=None,
                      image_stds=None,
                      evolution_augment_config=None,
                      is_training=False,
                      num_workers=1):
    if 'cifar' in name:

        def _collate_fn(samples):
            images = []
            labels = []
            for sample in samples:
                images += [sample['augmented_images']]
                labels += [sample['fine_label']]
            images = torch.stack(images, dim=0)
            _, _, c, h, w = images.shape
            images = torch.reshape(images, [-1, c, h, w])
            labels = torch.tensor(labels, dtype=torch.long)
            return {'pixel_values': images, 'labels': labels}

        if is_training:
            print('Building {} train dataloader'.format(name))

            _random_crop_fn = transforms.RandomCrop(size=32, padding=4)

            def _random_crop(sample):
                sample['img'] = [_random_crop_fn(sample['img'][0])]
                return sample

            train_transforms = transforms.Compose([_random_crop])
            if evolution_augment_config is not None:
                train_transforms.transforms.append(
                    EvolutionAugment(**dataclasses.asdict(
                        evolution_augment_config),
                                     image_means=image_means,
                                     image_stds=image_stds))
            print('Building EvolutionAugment with config: \n',
                  evolution_augment_config)

            dataset = datasets.load_dataset(name,
                                            split='train',
                                            cache_dir='/data')
            dataset = dataset.with_transform(train_transforms)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     collate_fn=_collate_fn,
                                                     pin_memory=True,
                                                     num_workers=num_workers)
        else:
            _to_tensor = transforms.ToTensor()
            _normalize = transforms.Normalize(mean=image_means, std=image_stds)

            def _transform(sample):
                img = _to_tensor(sample['img'][0])
                return {
                    'pixel_values': [_normalize(img)],
                    'labels': sample['fine_label']
                }

            print('Building {} val dataloader'.format(name))
            dataset = datasets.load_dataset(name,
                                            split='test',
                                            cache_dir='/data')
            dataset = dataset.with_transform(_transform)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     pin_memory=True,
                                                     num_workers=num_workers)
        return dataloader
