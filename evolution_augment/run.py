import os
import time
from datetime import datetime

from torch.nn import CrossEntropyLoss
import wandb

from mlproject.boilerplate import Learner
from mlproject.evaluators import ClassificationEvaluator
from mlproject.trainers import ClassificationTrainer

from evolution_augment import config
from evolution_augment import dataloaders
from evolution_augment import models
from evolution_augment import optimizers

if __name__ == '__main__':
    experiment_cfg = config.wideresnet28x18_randaugment_cifar100_bs256_ep200()

    os.makedirs(experiment_cfg.experiment_dir, exist_ok=True)
    wandb.init(
        name='wideresnet28x18_randaugment_cifar100_bs256_ep200-[{}]'.format(
            datetime.now().strftime('%d-%m-%Y|%H:%M:%S')),
        project=os.environ.get('WANDB_PROJECT',
                               experiment_cfg.train_dataloader.dataset.name),
        resume='allow',  # allow, True, False, must
        dir=experiment_cfg.experiment_dir,
        save_code=False)

    model = models.ModelFactory.build(config=experiment_cfg.model)

    optimizer = optimizers.get_optimizer(
        params=model.parameters(), config=experiment_cfg.trainers.optimizer)
    lr_sched = optimizers.get_lr_scheduler(
        optimizer=optimizer, config=experiment_cfg.trainers.scheduler)

    train_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.train_dataloader)
    val_dataloader = dataloaders.DataloaderFactory.build(
        experiment_cfg.val_dataloaders)
    criterion = CrossEntropyLoss()

    experiment = Learner(
        experiment_name=experiment_cfg.experiment_name,
        experiment_dir=experiment_cfg.experiment_dir,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=[val_dataloader],
        trainers=[
            ClassificationTrainer(optimizer=optimizer,
                                  scheduler=lr_sched,
                                  experiment_tracker=wandb)
        ],
        evaluators=[ClassificationEvaluator(experiment_tracker=wandb)],
        evaluate_every_n_steps=500,
        checkpoint_every_n_steps=experiment_cfg.train_iters // 10,
        checkpoint_after_validation=True,
        train_iters=experiment_cfg.train_iters,
        resume=True,
    )
    experiment.run()