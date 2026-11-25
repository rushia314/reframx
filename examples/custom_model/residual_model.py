"""Custom Model example: plain PyTorch nn.Module + TorchModelAdapter + custom validator + .rfx."""

from __future__ import annotations

import tempfile
from pathlib import Path

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


class CustomResidualNet(nn.Module):
    """Plain PyTorch model with a custom validator (no framework base class)."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(x + torch.relu(self.fc1(x)))

    def validate_component(self, ctx: rf.ValidationContext) -> rf.ValidationReport:
        report = rf.ValidationReport()
        if self.dim < 8:
            report.add(
                rf.ValidationIssue(
                    severity=rf.ValidationSeverity.WARNING,
                    code="SMALL_DIM",
                    message=f"dim={self.dim} is very small and may underfit.",
                )
            )
        return report


def main() -> None:
    torch.manual_seed(0)
    raw_model = CustomResidualNet(dim=16)
    # TorchModelAdapter automatically adds Capability("forward") and StateProvider.
    model = TorchModelAdapter(raw_model)

    x = torch.randn(32, 16)
    y = x.sum(dim=-1, keepdim=True)
    ds = torch.utils.data.TensorDataset(x, y)

    experiment = rf.Experiment(
        name="CustomResidualModel",
        task=rf.RegressionTask(),
        method=TorchRegressionMethod(),
        model=model,
        dataset=TorchDatasetAdapter(datasets={"train": ds}, dataloader_kwargs={"batch_size": 16}),
        objectives=MSEObjective(),
        optimizers=torch.optim.Adam(raw_model.parameters(), lr=1e-2),
        strategy=TorchStrategy(device="auto"),
    )

    report = experiment.validate()
    print(f"Validation valid={report.is_valid}, params={model.spec.attributes['total_parameters']}")

    run = experiment.run(max_epochs=10, eval_split=None)
    print(
        f"Loss: epoch 1={run.epoch_history[0]['train/loss']:.4f} -> "
        f"epoch 10={run.epoch_history[-1]['train/loss']:.4f}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rfx_path = experiment.export(Path(tmpdir) / "residual.rfx", profile="standard")
        imported = rf.load_experiment(rfx_path)
        fresh = CustomResidualNet(dim=16)
        imported.restore_component_state("model", fresh)
        print(f"Restored weights into fresh model from {rfx_path.name}")


if __name__ == "__main__":
    main()
