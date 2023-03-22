import dataclasses
import json

from evolution_augment import config
from evolution_augment import dataloaders
from evolution_augment import models
from evolution_augment import optimizers

if __name__ == '__main__':
    config = config.wideresnet28x18_cifar100()
    # print(json.dumps(dataclasses.asdict(config), indent=4))

    # train_dataloader_config = config.train_dataloader
    # val_dataloader_config = config.val_dataloaders
    # evolution_augment_config = train_dataloader_config.evolution_augment_config

    # train_dataloader = dataloaders.create_dataloader(
    #     is_training=True,
    #     name=train_dataloader_config.dataset.name,
    #     batch_size=train_dataloader_config.batch_size,
    #     evolution_augment_config=evolution_augment_config,
    #     image_means=train_dataloader_config.dataset.image_means,
    #     image_stds=train_dataloader_config.dataset.image_stds,
    #     num_workers=train_dataloader_config.num_workers)

    # val_dataloader = dataloaders.create_dataloader(
    #     is_training=False,
    #     name=val_dataloader_config.dataset.name,
    #     batch_size=val_dataloader_config.batch_size,
    #     image_means=val_dataloader_config.dataset.image_means,
    #     image_stds=val_dataloader_config.dataset.image_stds,
    #     num_workers=val_dataloader_config.num_workers)

    model = models.ModelFactory.build(config=config.model)
    optimizer = optimizers.get_optimizer(params=model.parameters(),
                                         config=config.trainers.optimizer)
    lr_scheduler = optimizers.get_lr_scheduler(
        optimizer=optimizer, config=config.trainers.scheduler)

    # for sample in train_dataloader:
    #     image, labels = sample['pixel_values'], sample['labels']
    #     print(image.shape, labels.shape)
    #     break

    # for sample in val_dataloader:
    #     image, labels = sample['pixel_values'], sample['labels']
    #     print(image.shape, labels.shape)
    #     break