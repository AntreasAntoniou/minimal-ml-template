import os
import random
from datetime import datetime

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb

from evolution_augment import config
from evolution_augment import dataloaders
from evolution_augment import models
from evolution_augment import optimizers

os.environ['DATASETS_CACHE_DIR'] = '/home/shumbarw/dataset_cache'

os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
os.environ['XLA_USE_BF16'] = '1'


def mp_fn(index, experiment_cfg):
    print('Starting process on rank:', index)
    device = xm.xla_device()

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    set_seed(42)

    print('Using learning rata:', experiment_cfg.trainers.optimizer.lr)
    model = models.ModelFactory.build(config=experiment_cfg.model).to(device)

    optimizer = optimizers.get_optimizer(
        params=model.parameters(), config=experiment_cfg.trainers.optimizer)
    lr_sched = optimizers.get_lr_scheduler(
        optimizer=optimizer, config=experiment_cfg.trainers.scheduler)

    train_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.train_dataloader)

    mp_train_dataloader = pl.MpDeviceLoader(train_dataloader, device)

    val_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.val_dataloaders)
    mp_val_dataloader = pl.MpDeviceLoader(val_dataloader, device)

    criterion = CrossEntropyLoss()

    train_iteration = 0
    total_train_iterations = experiment_cfg.train_iters

    if xm.is_master_ordinal():
        train_progress_bar = tqdm(initial=train_iteration,
                                  total=total_train_iterations)

    while train_iteration < total_train_iterations:
        for batch in mp_train_dataloader:
            optimizer.zero_grad()
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            xm.optimizer_step(optimizer, barrier=True)
            lr_sched.step()
            train_iteration += 1

            global_train_accuracy = xm.mesh_reduce(
                'accuracy',
                (logits.argmax(dim=-1) == batch["labels"]).float().mean(),
                lambda x: sum(x) / len(x))
            global_train_loss = xm.mesh_reduce('global_loss', loss,
                                               lambda x: sum(x) / len(x))

            if xm.is_master_ordinal():
                train_progress_bar.set_description(
                    'Rank: {} | Step: {}/{} | Loss: {:.3f} | LR: {:.4f} | Accuracy: {:.3f}'
                    .format(index, train_iteration, total_train_iterations,
                            global_train_loss.item(),
                            lr_sched.get_last_lr()[0], global_train_accuracy))
                train_progress_bar.update(1)

            if train_iteration % experiment_cfg.evaluate_every_n_steps == 0:
                if xm.is_master_ordinal():
                    train_progress_bar.close()
                    val_progress_bar = tqdm(total=len(mp_val_dataloader))

                validation_accuracies = []
                validation_losses = []
                for val_iteration, batch in enumerate(mp_val_dataloader):
                    images = batch['pixel_values'].to(device)
                    labels = batch['labels'].to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)

                    global_validation_accuracy = xm.mesh_reduce(
                        'accuracy', (logits.argmax(
                            dim=-1) == batch["labels"]).float().mean(),
                        lambda x: sum(x) / len(x))
                    global_validation_loss = xm.mesh_reduce(
                        'global_loss', loss, lambda x: sum(x) / len(x))
                    validation_accuracies += [
                        global_validation_accuracy.item()
                    ]
                    validation_losses += [global_validation_loss.item()]

                    if xm.is_master_ordinal():
                        val_progress_bar.set_description(
                            'Train Step: {} | Validation Step {}/{} | Mean Validation Loss: {:.3f} | Mean Validation Accuracy: {:.3f}'
                            .format(train_iteration, val_iteration + 1,
                                    len(mp_val_dataloader),
                                    float(np.mean(validation_losses)),
                                    float(np.mean(validation_accuracies))))
                        val_progress_bar.update(1)

                if xm.is_master_ordinal():
                    train_progress_bar = tqdm(initial=train_iteration,
                                              total=total_train_iterations)
                    val_progress_bar.close()

    if xm.is_master_ordinal():
        train_progress_bar.close()
        xm.save(model.state_dict(), './model.pt')


if __name__ == '__main__':
    xmp.spawn(
        fn=mp_fn,
        nprocs=8,
        args=(config.wideresnet28x18_randaugment_cifar10_bs256_ep200_tpu(), ),
        start_method='fork')
