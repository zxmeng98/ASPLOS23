import logging
from ray.tune.analysis import experiment_analysis

import torch
import torchvision
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler
from ray.tune import CLIReporter
from ray.util.sgd.torch import TorchTrainer, TrainingOperator
from ray.util.sgd.utils import override

import models.cifar as Cifar
from utils import get_datasets

SEARCH_SPACE = {
    "lr": tune.qloguniform(1e-4, 1, 1e-4),
    "momentum": tune.quniform(0.1, 0.99, 0.01),
    "weight_decay": tune.qloguniform(1e-5, 1e-2, 1e-5),
    "batch_size": tune.choice([32, 64, 128, 256, 512])
}

EXPERIMENT_CONFIG = {
    "model": "densenet121",
    "dataset": "cifar10"
}


class VisionTrainingOperator(TrainingOperator):
    @ override(TrainingOperator)
    def setup(self, config):
        assert torch.cuda.is_available(), "No GPU is detected."

        # Create model
        dataset = config.get("dataset")
        if dataset == "imagenet":
            model = torchvision.models.__dict__[config.get("model")]()
        elif dataset == "cifar10" or "cifar100":
            model = Cifar.__dict__[config.get("model")](dataset)

        # Load in training and validation data
        train_set, val_set = get_datasets(dataset)
        train_loader = DataLoader(train_set, batch_size=config.get("batch_size"),
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=config.get("batch_size"),
                                num_workers=4, pin_memory=True)

        # Create loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.get("lr", 0.01),
                                    momentum=config.get("momentum", 0.9),
                                    weight_decay=config.get("weight_decay", 0.001))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[50, 100, 150], gamma=0.2)

        self.model, self.optimizer, self.criterion, self.scheduler = self.register(
            models=model, optimizers=optimizer, criterion=criterion, schedulers=lr_scheduler)

        self.register_data(train_loader=train_loader,
                           validation_loader=val_loader)


if __name__ == "__main__":
    ray.init(address="auto")

    config = dict(SEARCH_SPACE, **EXPERIMENT_CONFIG)

    VisionTrainable = TorchTrainer.as_trainable(
        training_operator_cls=VisionTrainingOperator,
        config=config,
        num_workers=1,
        # num_cpus_per_worker=2,
        scheduler_step_freq="epoch",
        use_gpu=True)

    # scheduler = ASHAScheduler(max_t=200,
    #                           grace_period=3,
    #                           reduction_factor=3)

    # scheduler = HyperBandScheduler(max_t=81, reduction_factor=3)

    reporter = CLIReporter()
    reporter.add_metric_column("val_loss", "loss")
    reporter.add_metric_column("val_accuracy", "acc")

    experiment_name = f'{config["dataset"]}_{config["model"]}_GRID_200sample_200epoch'

    analysis = tune.run(
        VisionTrainable,
        # scheduler=scheduler,
        num_samples=200,
        name=experiment_name,
        config=SEARCH_SPACE,
        metric="val_accuracy",
        mode="max",
        stop={"val_accuracy": 0.96,
              "training_iteration": 200},
        # time_budget_s=36000,
        # resume=True,
        checkpoint_at_end=True,
        checkpoint_freq=3,
        max_failures=3,
        keep_checkpoints_num=1,
        progress_reporter=reporter,
    )

    print("Best config is:", analysis.best_config)