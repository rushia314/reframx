"""Custom Method example: full manual freedom with 2 models, 2 losses, and 2 optimizers.

Shows how BaseMethod lets researchers write non-standard training procedures
(e.g. adversarial / GAN / dual-branch updates) with multiple ctx.backward()
and ctx.step(optimizer_name) calls inside a single train_step().
"""

from __future__ import annotations

import torch
from torch import nn

import reframx as rf
from reframx.integrations.pytorch import TorchDatasetAdapter, TorchStrategy


class DualBranchMethod(rf.BaseMethod):
    """Trains two independent networks with separate losses and optimizers in one step."""

    def __init__(self) -> None:
        super().__init__(capabilities="dual_predictions")

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        ctx: rf.ExecutionContext,
    ) -> rf.MethodResult:
        net_a, net_b = ctx.model["branch_a"], ctx.model["branch_b"]

        # 1. Forward + backward + step for branch A
        pred_a = net_a(batch["x"])
        loss_a = torch.nn.functional.mse_loss(pred_a, batch["y_a"])
        ctx.backward(loss_a)
        ctx.step("opt_a")

        # 2. Forward + backward + step for branch B
        pred_b = net_b(batch["x"])
        loss_b = torch.nn.functional.mse_loss(pred_b, batch["y_b"])
        ctx.backward(loss_b)
        ctx.step("opt_b")

        return rf.MethodResult(
            task_outputs={"dual_predictions": (pred_a.detach(), pred_b.detach())},
            objectives={"loss_a": loss_a, "loss_b": loss_b},
            metrics={"loss_a": loss_a, "loss_b": loss_b},
        )


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    branch_a = nn.Linear(4, 1).to(device)
    branch_b = nn.Linear(4, 1).to(device)

    x = torch.randn(32, 4)
    batches = [{"x": x, "y_a": x[:, :1] * 2.0, "y_b": -x[:, 1:2] * 3.0}]

    experiment = rf.Experiment(
        name="DualOptimizerDemo",
        task=rf.BaseTask(requirements="dual_predictions"),
        method=DualBranchMethod(),
        model={"branch_a": branch_a, "branch_b": branch_b},
        dataset=TorchDatasetAdapter(loaders={"train": batches}),
        optimizers={
            "opt_a": torch.optim.Adam(branch_a.parameters(), lr=0.1),
            "opt_b": torch.optim.Adam(branch_b.parameters(), lr=0.1),
        },
        strategy=TorchStrategy(device=device),
    )

    run = experiment.run(max_epochs=25, eval_split=None)
    first = run.epoch_history[0]
    last = run.epoch_history[-1]
    print(f"Branch A loss: {first['train/loss_a']:.4f} -> {last['train/loss_a']:.4f}")
    print(f"Branch B loss: {first['train/loss_b']:.4f} -> {last['train/loss_b']:.4f}")


if __name__ == "__main__":
    main()
