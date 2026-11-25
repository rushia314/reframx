"""Concise end-to-end PyTorch experiment using built-in components.

Requires PyTorch to be installed in your active environment (this library does
not install or overwrite your PyTorch/CUDA build).
"""

from __future__ import annotations

import reframx as rf
from reframx.integrations.pytorch import (
    BCEWithLogitsObjective,
    TorchAccuracyEvaluator,
    TorchClassificationMethod,
    TorchDatasetAdapter,
    TorchModelAdapter,
    TorchStrategy,
    require_torch,
)


def main() -> None:
    torch = require_torch()
    torch.manual_seed(42)

    # 1. Plain PyTorch nn.Module wrapped non-invasively.
    raw_model = torch.nn.Sequential(
        torch.nn.Linear(2, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )

    # 2. Standard PyTorch (x, y) TensorDataset — works directly with built-ins!
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    xor_ds = torch.utils.data.TensorDataset(x, y)

    # 3. Compose Experiment with built-in Task, Method, Objective, Evaluator, Strategy.
    experiment = rf.Experiment(
        name="PyTorch_XOR_Builtins",
        task=rf.ClassificationTask(num_classes=2),
        method=TorchClassificationMethod(),
        model=TorchModelAdapter(raw_model),
        dataset=TorchDatasetAdapter(
            datasets={"train": xor_ds, "val": xor_ds},
            dataloader_kwargs={"batch_size": 4},
        ),
        objectives=BCEWithLogitsObjective(),
        optimizers=torch.optim.Adam(raw_model.parameters(), lr=0.05),
        evaluator=TorchAccuracyEvaluator(),
        strategy=TorchStrategy(device="auto", precision="fp32", max_grad_norm=5.0),
    )

    run = experiment.run(max_epochs=100)
    print(f"Run {run.run_id} ({run.status.value}) on {experiment.strategy.device}")
    print(f"Final eval: {run.eval_history[-1]}")


if __name__ == "__main__":
    main()
