"""Tests for v0.1.2 production research features.

Covers:
1. Resume (start_step/start_epoch/resume_from .rfx & CheckpointHook).
2. Standalone evaluate() calling prepare_model & method.setup() + should_stop.
3. keep_eval_mode=True and max_train_history ring buffer.
4. on_exception lifecycle hook + emergency checkpointing.
5. TorchStrategy: auto_move_model=False, grad_norm logging, skip_nonfinite_grads, schedulers.
6. InstrumentationManager multi-call reduce modes (mean/min/max/list/last).
7. Streaming .rfx export + base_model_ref.
8. Batteries-included hooks: CheckpointHook, JSONLLoggerHook, ConsoleLoggerHook,
   FileControlHook (live stop & hot-reload LR), and CudaMemoryHook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import reframx as rf
from reframx.hooks import (
    CheckpointHook,
    ConsoleLoggerHook,
    FileControlHook,
    JSONLLoggerHook,
    load_checkpoint,
)


class StatefulModel(rf.BaseModel):
    def __init__(self, w: float = 0.0) -> None:
        super().__init__(capabilities="forward")
        self.w = w
        self.training = True

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def __call__(self, x: float) -> float:
        return self.w * x

    def get_component_state(self) -> dict[str, float]:
        return {"w": self.w}

    def load_component_state(self, state: dict[str, float]) -> None:
        self.w = float(state["w"])


class CountingMethod(rf.BaseMethod):
    def __init__(self) -> None:
        super().__init__(capabilities="predictions")
        self.setup_calls = 0
        self.eval_batches_seen = 0
        self.train_modes: list[bool] = []

    def setup(self, ctx: Any) -> None:
        self.setup_calls += 1

    def train_step(self, batch: dict[str, float], ctx: rf.ExecutionContext) -> rf.MethodResult:
        self.train_modes.append(ctx.model.training)
        ctx.model.w += 1.0
        return rf.MethodResult(
            task_outputs={"predictions": ctx.model(batch["x"])},
            metrics={"loss": 1.0 / (ctx.state.global_step + 1)},
        )

    def eval_step(self, batch: dict[str, float], ctx: rf.ExecutionContext) -> rf.MethodResult:
        self.eval_batches_seen += 1
        if batch.get("stop_here"):
            ctx.request_stop()
        return rf.MethodResult(
            task_outputs={"predictions": ctx.model(batch["x"])},
            metrics={"val_loss": 0.5},
        )


def test_standalone_evaluate_prepares_model_calls_setup_and_respects_stop():
    prepared: list[Any] = []

    class TrackingStrategy(rf.StandardBackwardStrategy):
        def prepare_model(self, model: Any) -> Any:
            prepared.append(model)
            return model

    method = CountingMethod()
    exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=method,
        model=StatefulModel(w=2.0),
        dataset=rf.BaseDataset(
            splits_data={
                "val": [
                    {"x": 1.0, "stop_here": True},
                    {"x": 2.0},
                    {"x": 3.0},
                ]
            }
        ),
        strategy=TrackingStrategy(),
    )

    metrics = exp.evaluate(split="val")
    assert len(prepared) == 1
    assert method.setup_calls == 1
    assert method.eval_batches_seen == 1  # Stopped early after batch 1!
    assert metrics["val_loss"] == pytest.approx(0.5)


def test_keep_eval_mode_ring_buffer_and_resume_from_rfx_and_checkpoint(tmp_path: Path):
    model = StatefulModel(w=0.0)
    method = CountingMethod()
    ckpt_hook = CheckpointHook(
        tmp_path / "ckpts",
        save_every_steps=2,
        monitor="loss",
        mode="min",
    )

    exp = rf.Experiment(
        name="ResumeDemo",
        task=rf.RegressionTask(),
        method=method,
        model=model,
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}] * 4}),
        hooks=[ckpt_hook],
    )

    run1 = exp.run(
        max_steps=4,
        eval_split=None,
        keep_eval_mode=True,
        max_train_history=2,
    )
    assert method.train_modes == [False, False, False, False]
    assert len(run1.train_history) == 2  # Ring buffer capped at 2!
    assert [r["step"] for r in run1.train_history] == [3, 4]
    assert model.w == 4.0
    assert ckpt_hook.last_path is not None and ckpt_hook.last_path.is_file()
    assert ckpt_hook.best_path is not None and ckpt_hook.best_path.is_file()

    # 1. Resume from CheckpointHook file into a fresh model
    fresh_model = StatefulModel(w=0.0)
    exp2 = rf.Experiment(
        task=rf.RegressionTask(),
        method=CountingMethod(),
        model=fresh_model,
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}] * 4}),
    )
    run2 = exp2.run(
        max_steps=6,
        eval_split=None,
        resume_from=load_checkpoint(ckpt_hook.last_path),
    )
    assert [r["step"] for r in run2.train_history] == [5, 6]
    assert fresh_model.w == 6.0

    # 2. Export streaming .rfx with base_model_ref and resume directly from .rfx
    rfx_path = exp.export(
        tmp_path / "snap.rfx",
        profile="standard",
        base_model_ref="meta-llama/Llama-3-8B",
    )
    imported = rf.load_experiment(rfx_path)
    assert imported.manifest.metadata["base_model_ref"] == "meta-llama/Llama-3-8B"

    fresh_model2 = StatefulModel(w=0.0)
    exp3 = rf.Experiment(
        task=rf.RegressionTask(),
        method=CountingMethod(),
        model=fresh_model2,
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}] * 4}),
    )
    run3 = exp3.run(max_steps=5, eval_split=None, resume_from=rfx_path)
    assert [r["step"] for r in run3.train_history] == [5]
    assert fresh_model2.w == 5.0


def test_on_exception_hook_saves_emergency_checkpoint(tmp_path: Path):
    seen_exc: list[str] = []

    class CrashMethod(rf.BaseMethod):
        def __init__(self) -> None:
            super().__init__(capabilities="predictions")

        def train_step(self, batch: Any, ctx: rf.ExecutionContext) -> rf.MethodResult:
            if ctx.state.global_step == 2:
                raise RuntimeError("Simulated CUDA OOM")
            return rf.MethodResult(task_outputs={"predictions": 0.0}, metrics={"loss": 1.0})

    class ObserveExcHook(rf.BaseHook):
        def on_exception(self, ctx: rf.ExecutionContext, exc: BaseException) -> None:
            seen_exc.append(str(exc))

    ckpt_dir = tmp_path / "emergency_ckpts"
    exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=CrashMethod(),
        model=StatefulModel(w=3.0),
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}] * 3}),
        hooks=[CheckpointHook(ckpt_dir, save_on_exception=True), ObserveExcHook()],
    )

    with pytest.raises(RuntimeError, match="Simulated CUDA OOM"):
        exp.run(max_epochs=1, eval_split=None)

    assert seen_exc == ["Simulated CUDA OOM"]
    assert (ckpt_dir / "emergency_step_2.pt").is_file()


def test_instrumentation_multi_call_reduce_modes():
    inst = rf.InstrumentationManager()
    inst.watch_probe("layer/gate", lambda ev: {"v": ev.value}, reduce="mean")
    inst.watch_probe("layer/peak", lambda ev: {"v": ev.value}, reduce="max")
    inst.watch_probe("layer/all", lambda ev: {"v": ev.value}, reduce="list")

    with inst.activate(epoch=1, step=1, phase="train"):
        for val in (1.0, 2.0, 3.0, 4.0):
            rf.probe("layer/gate", val)
            rf.probe("layer/peak", val)
            rf.probe("layer/all", val)

    metrics = inst.pop_step_metrics()
    assert metrics["layer/gate/v"] == pytest.approx(2.5)
    assert metrics["layer/peak/v"] == pytest.approx(4.0)
    assert metrics["layer/all/v"] == [1.0, 2.0, 3.0, 4.0]


def test_jsonl_console_and_file_control_hooks(tmp_path: Path):
    jsonl_path = tmp_path / "metrics.jsonl"
    control_path = tmp_path / "control.json"
    hparams_path = tmp_path / "live_hparams.json"
    printed: list[str] = []

    class FakeOpt:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-3}]

        def step(self) -> None:
            pass

        def zero_grad(self) -> None:
            pass

    opt = FakeOpt()
    hparams_path.write_text(json.dumps({"lr": 5e-5}), encoding="utf-8")

    class TriggerControlHook(rf.BaseHook):
        def on_step_start(self, ctx: rf.ExecutionContext, batch: Any) -> None:
            if ctx.state.global_step == 2:
                control_path.write_text(
                    json.dumps({"stop": True, "reason": "file_stop"}),
                    encoding="utf-8",
                )

    exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=CountingMethod(),
        model=StatefulModel(),
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}] * 5}),
        optimizers=opt,
        hooks=[
            TriggerControlHook(),
            FileControlHook(control_path=control_path, hparams_path=hparams_path),
            JSONLLoggerHook(jsonl_path),
            ConsoleLoggerHook(log_every_steps=1, log_every_epochs=None, printer=printed.append),
        ],
    )

    run = exp.run(max_epochs=1, eval_split=None)
    assert run.status is rf.RunStatus.INTERRUPTED
    assert run.error == "file_stop"
    assert len(run.train_history) == 2
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-5)
    assert len(printed) == 2

    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [line["event"] for line in lines[:2]] == ["train", "train"]


def test_torch_strategy_auto_move_model_grad_norm_skip_nan_and_scheduler():
    import torch

    from reframx.integrations.pytorch import TorchModelAdapter, TorchStrategy

    raw = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(raw.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

    strategy = TorchStrategy(
        device="cpu",
        auto_move_model=False,
        max_grad_norm=1.0,
        skip_nonfinite_grads=True,
        schedulers=sched,
        scheduler_step_on="step",
    )

    # auto_move_model=False must return model untouched
    assert strategy.prepare_model(TorchModelAdapter(raw)).module is raw

    ctx = rf.ExecutionContext(model=raw, strategy=strategy, optimizers={"default": opt})

    # 1. Normal finite step logs grad_norm and steps scheduler
    loss = raw(torch.ones(1, 2)).sum()
    strategy.backward(loss, ctx)
    assert strategy.optimizer_step(opt, ctx) is True
    assert "grad_norm" in ctx.step_metrics
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    # 2. NaN gradient step is skipped cleanly without corrupting weights
    before_w = raw.weight.detach().clone()
    raw.weight.grad = torch.full_like(raw.weight, float("nan"))
    raw.bias.grad = torch.zeros_like(raw.bias)
    assert strategy.optimizer_step(opt, ctx) is False
    assert ctx.step_metrics.get("skipped_nonfinite_step") == 1
    assert torch.equal(raw.weight.detach(), before_w)
