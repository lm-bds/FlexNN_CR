from typing import Any, Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from metrics import Metric
from tracking import ExperimentTracker, Stage


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.run_count = 0
        self.loader = loader

        self.mae_metric = Metric()
        self.avg_loss = 0
        self.model = model
        self.optimizer = optimizer
        # Objective (loss) function
        self.compute_loss = torch.nn.MSELoss()
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        # Assume Stage based on presence of optimizer
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN


    @property
    def avg_mae(self):
        return self.mae_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        self.model.train(self.stage is Stage.TRAIN)
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            
            loss, batch_mae, avg_loss = self._run_single(x, y)

            experiment.add_batch_metric("mae", batch_mae, self.run_count)
            experiment.add_batch_metric("loss", avg_loss, self.run_count)

            if self.optimizer:
                # Reverse-mode AutoDiff (backpropagation)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
      
    def _run_single(self, x: Any, y: Any):
        self.run_count += 1
        batch_size: int = x.shape[0]
        prediction = self.model(x)
        loss = self.compute_loss(prediction, y.view(-1,1))
        avg_loss = loss / len(y)

        # Compute Batch Validation Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(prediction.detach().numpy(), axis=1)
        #y_prediction_np = prediction.detach().numpy()

        batch_mae: float = mean_absolute_error(y_np, y_prediction_np)

        self.mae_metric.update(batch_mae, batch_size)
        self.avg_loss =avg_loss

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]
        return loss, batch_mae, avg_loss

    def reset(self):
        self.mae_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
    test_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch_id: int,
):
    # Training Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.reset()
    train_runner.run("Train Batches", experiment)

    # Log Training Epoch Metrics
    #experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_metric("mae", train_runner.avg_mae, epoch_id)

    # Testing Loop
    experiment.set_stage(Stage.VAL)
    test_runner.reset()
    test_runner.run("Validation Batches", experiment)

    # Log Validation Epoch Metrics
    #experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch_id)
    experiment.add_epoch_metric("mae", test_runner.avg_mae, epoch_id)
