"""Custom Execution example: gradient accumulation, EarlyStoppingHook, and experiment.stop()."""

from __future__ import annotations

import torch
from torch import nn

import reframx as rf
from reframx.integrations.pytorch import (
    MSEObjective,
    TorchDatasetAdapter,
    TorchModelAdapter,
    TorchRegressionMethod,
    TorchStrategy,
)


class EarlyStopOnLossHook(rf.BaseHook):
    """Stops training gracefully once train loss drops below a target threshold."""

    def __init__(self, target_loss: float, experiment: rf.Experiment) -> None:
        self.target_loss = target_loss
        self.experiment = experiment

    def on_epoch_end(self, ctx: rf.ExecutionContext, epoch_metrics) -> None:
        loss = epoch_metrics.get("train/loss", float("inf"))
        print(f"Epoch {ctx.state.epoch:2d} | train/loss={loss:.4f}")
        if loss <= self.target_loss:
            print(f"Reached target loss <= {self.target_loss}; calling experiment.stop()")
            self.experiment.stop(f"target_loss_{self.target_loss}_reached")


def main() -> None:
    torch.manual_seed(0)
    raw_model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    x = torch.randn(64, 4)
    y = x[:, :1] * 3.0 - 0.5
    ds = torch.utils.data.TensorDataset(x, y)

    experiment = rf.Experiment(
        name="CustomExecutionDemo",
        task=rf.RegressionTask(),
        method=TorchRegressionMethod(),
        model=TorchModelAdapter(raw_model),
        dataset=TorchDatasetAdapter(datasets={"train": ds}, dataloader_kwargs={"batch_size": 16}),
        objectives=MSEObjective(),
        optimizers=torch.optim.Adam(raw_model.parameters(), lr=0.05),
        strategy=TorchStrategy(
            device="auto",
            grad_accumulation_steps=2,
            max_grad_norm=1.0,
        ),
    )
    experiment.hooks.append(EarlyStopOnLossHook(target_loss=0.05, experiment=experiment))

    run = experiment.run(max_epochs=100, eval_split=None, catch_interrupt=True)
    print(f"Stopped at epoch {len(run.epoch_history)} with status={run.status.value} ({run.error})")


if __name__ == "__main__":
    main()
