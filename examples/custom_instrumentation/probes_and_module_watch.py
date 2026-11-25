"""Custom Instrumentation example: combining rf.Probe and non-invasive watch_modules()."""

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


class ProbedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        # Explicit Probe on any intermediate tensor (zero-overhead when unwatched).
        h = rf.probe("mlp/post_blocks", h)
        return self.head(h)


def main() -> None:
    torch.manual_seed(0)
    raw_model = ProbedMLP()
    model = TorchModelAdapter(raw_model)

    inst = rf.InstrumentationManager()
    # 1. Watch explicit Probe for summary stats & NaN/Inf detection.
    inst.watch_probe("mlp/*", rf.tensor_stats)
    inst.watch_probe("mlp/*", lambda ev: rf.check_nan_inf(ev, raise_on_error=True))

    # 2. Non-invasively watch Linear layers (both forward output & backward grad_output).
    inst.watch_modules(
        raw_model,
        pattern="blocks.*",
        stages=("post_forward", "backward"),
        callback=lambda ev: {"l2_norm": rf.tensor_stats(ev).get("l2_norm", 0.0)},
    )

    x = torch.randn(32, 8)
    y = x.mean(dim=-1, keepdim=True)
    ds = torch.utils.data.TensorDataset(x, y)

    experiment = rf.Experiment(
        name="InstrumentationDemo",
        task=rf.RegressionTask(),
        method=TorchRegressionMethod(),
        model=model,
        dataset=TorchDatasetAdapter(datasets={"train": ds}, dataloader_kwargs={"batch_size": 16}),
        objectives=MSEObjective(),
        optimizers=torch.optim.Adam(raw_model.parameters(), lr=1e-2),
        strategy=TorchStrategy(device="auto"),
        hooks=[rf.InstrumentationHook(inst)],
    )

    run = experiment.run(max_epochs=2, eval_split=None)
    records = run.artifacts["instrumentation"]
    print(f"Captured {len(records)} instrumentation records across {len(run.train_history)} steps.")
    for rec in records[:4]:
        print(f"  step={rec.step} | {rec.name} [{rec.stage}] -> {rec.metrics}")


if __name__ == "__main__":
    main()
