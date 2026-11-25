"""Tests for pluggable ExecutionBackend and optional MMEngineBackend."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import reframx as rf
from reframx.integrations.mmengine import (
    MMEngineBackend,
    require_mmengine,
)


def _make_fake_mmengine() -> Any:
    """Create a lightweight fake ``mmengine`` module simulating Runner lifecycle."""
    fake_mm = types.ModuleType("mmengine")

    class FakeBaseModel:
        def __init__(self) -> None:
            pass

    class FakeBaseMetric:
        def __init__(self) -> None:
            self.results: list[Any] = []

    class FakeHook:
        pass

    class FakeOptimWrapper:
        def __init__(self, optimizer: Any) -> None:
            self.optimizer = optimizer
            self.backward_calls = 0
            self.step_calls = 0
            self.zero_grad_calls = 0

        def backward(self, loss: Any) -> None:
            self.backward_calls += 1
            loss.backward()

        def step(self) -> None:
            self.step_calls += 1
            self.optimizer.step()

        def zero_grad(self) -> None:
            self.zero_grad_calls += 1
            self.optimizer.zero_grad()

    class FakeOptimWrapperDict(dict):
        pass

    class FakeRunner:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.train_dataloader = kwargs.get("train_dataloader")
            self.val_dataloader = kwargs.get("val_dataloader")
            self.val_evaluator = kwargs.get("val_evaluator") or []
            self.train_cfg = kwargs.get("train_cfg") or {}
            self.optim_wrapper = kwargs.get("optim_wrapper")
            self.custom_hooks = kwargs.get("custom_hooks") or []
            self.epoch = 0

        def _call_hooks(self, name: str, *args: Any, **kwargs: Any) -> None:
            for hook in self.custom_hooks:
                fn = getattr(hook, name, None)
                if callable(fn):
                    fn(self, *args, **kwargs)

        def val(self) -> dict[str, Any]:
            self._call_hooks("before_val_epoch")
            for batch in self.val_dataloader or []:
                outputs = self.model.val_step(batch)
                for metric in self.val_evaluator:
                    metric.process(batch, outputs)
            merged: dict[str, Any] = {}
            for metric in self.val_evaluator:
                merged.update(metric.compute_metrics([]))
            self._call_hooks("after_val_epoch", metrics=merged)
            return merged

        def train(self) -> Any:
            max_epochs = int(self.train_cfg.get("max_epochs", 1))
            self._call_hooks("before_run")
            for ep in range(max_epochs):
                self.epoch = ep
                self._call_hooks("before_train_epoch")
                for idx, batch in enumerate(self.train_dataloader or []):
                    self._call_hooks("before_train_iter", idx, data_batch=batch)
                    outputs = self.model.train_step(batch, self.optim_wrapper)
                    self._call_hooks(
                        "after_train_iter",
                        idx,
                        data_batch=batch,
                        outputs=outputs,
                    )
                if self.val_dataloader is not None:
                    self.val()
                self._call_hooks("after_train_epoch")
            self._call_hooks("after_run")
            return self.model

    fake_mm.model = types.SimpleNamespace(BaseModel=FakeBaseModel)  # type: ignore[attr-defined]
    fake_mm.evaluator = types.SimpleNamespace(BaseMetric=FakeBaseMetric)  # type: ignore[attr-defined]
    fake_mm.hooks = types.SimpleNamespace(Hook=FakeHook)  # type: ignore[attr-defined]
    fake_mm.optim = types.SimpleNamespace(  # type: ignore[attr-defined]
        OptimWrapper=FakeOptimWrapper,
        OptimWrapperDict=FakeOptimWrapperDict,
    )
    fake_mm.runner = types.SimpleNamespace(Runner=FakeRunner)  # type: ignore[attr-defined]
    return fake_mm


def test_require_mmengine_raises_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mmengine", None)
    with pytest.raises(ImportError, match="mmengine"):
        require_mmengine()


def test_native_and_mmengine_backends_implement_execution_backend_protocol():
    assert isinstance(rf.NativeBackend(), rf.ExecutionBackend)
    assert isinstance(MMEngineBackend(), rf.ExecutionBackend)


def test_same_experiment_runs_on_both_native_and_mmengine_backends(monkeypatch):
    fake_mm = _make_fake_mmengine()
    monkeypatch.setitem(sys.modules, "mmengine", fake_mm)

    class ScalarLoss:
        def __init__(self, value: float, on_bwd) -> None:
            self.value = value
            self._on_bwd = on_bwd

        def item(self) -> float:
            return self.value

        def backward(self) -> None:
            self._on_bwd()

    class ToyModel(rf.BaseModel):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("forward"),))
            self.w = 0.0
            self.grad = 0.0

        def __call__(self, x: float) -> float:
            return self.w * x

    class ToyOpt:
        def __init__(self, model: ToyModel, lr: float = 0.2) -> None:
            self.model = model
            self.lr = lr
            self.steps = 0

        def step(self) -> None:
            self.steps += 1
            self.model.w -= self.lr * self.model.grad

        def zero_grad(self) -> None:
            self.model.grad = 0.0

    class ToyObjective(rf.BaseObjective):
        def __init__(self, model: ToyModel) -> None:
            super().__init__(capabilities=(rf.Capability("loss"),))
            self.model = model

        def compute(self, pred: float, target: float, x: float) -> ScalarLoss:
            diff = pred - target

            def _bwd() -> None:
                self.model.grad += 2.0 * diff * x

            return ScalarLoss(diff * diff, _bwd)

    class ToyMethod(rf.SupervisedMethod):
        def __init__(self) -> None:
            super().__init__(
                requirements=(
                    rf.Requirement("forward", provider_type="model"),
                    rf.Requirement("loss", provider_type="objective"),
                ),
                capabilities=(rf.Capability("predictions"),),
            )

        def compute_step(
            self,
            batch: dict[str, float],
            ctx: rf.ExecutionContext,
        ) -> rf.MethodResult:
            pred = ctx.model(batch["x"])
            loss = ctx.objective(pred, batch["y"], batch["x"])
            return rf.MethodResult(task_outputs={"predictions": pred}, loss=loss)

    class ToyEvaluator(rf.BaseEvaluator):
        def __init__(self) -> None:
            super().__init__(requirements=(rf.Requirement("predictions"),))
            self.errs: list[float] = []

        def reset(self) -> None:
            self.errs.clear()

        def update(self, result: rf.MethodResult, batch: dict[str, float]) -> None:
            self.errs.append(abs(result.task_outputs["predictions"] - batch["y"]))

        def compute(self) -> dict[str, float]:
            return {"mae": sum(self.errs) / max(len(self.errs), 1)}

    def _run_with_backend(backend: rf.ExecutionBackend):
        model = ToyModel()
        opt = ToyOpt(model)
        events: list[str] = []

        class TraceHook(rf.BaseHook):
            def on_run_start(self, ctx: rf.ExecutionContext) -> None:
                events.append("start")

            def on_step_end(self, ctx: rf.ExecutionContext, batch, result) -> None:
                events.append("step")

            def on_run_end(self, ctx: rf.ExecutionContext, run: rf.Run) -> None:
                events.append("end")

        exp = rf.Experiment(
            task=rf.RegressionTask(),
            method=ToyMethod(),
            model=model,
            dataset=rf.BaseDataset(
                splits_data={
                    "train": [{"x": 1.0, "y": 2.0}],
                    "val": [{"x": 1.0, "y": 2.0}],
                }
            ),
            objectives=ToyObjective(model),
            optimizers=opt,
            evaluator=ToyEvaluator(),
            backend=backend,
            hooks=[TraceHook()],
        )
        run = exp.run(max_epochs=4)
        eval_metrics = exp.evaluate(split="val")
        return run, model.w, opt.steps, events, eval_metrics

    native_run, native_w, native_steps, native_events, native_eval = _run_with_backend(
        rf.NativeBackend()
    )
    mm_run, mm_w, mm_steps, mm_events, mm_eval = _run_with_backend(
        MMEngineBackend(work_dir="./work_dirs/test_mm")
    )

    assert native_run.status is rf.RunStatus.COMPLETED
    assert mm_run.status is rf.RunStatus.COMPLETED
    assert native_steps == mm_steps == 4
    assert native_w == pytest.approx(mm_w)
    assert len(mm_run.train_history) == 4
    assert len(mm_run.eval_history) == 4
    assert mm_eval["mae"] == pytest.approx(native_eval["mae"])
    assert mm_events[0] == "start"
    assert mm_events[-1] == "end"
    assert mm_events.count("step") == 4
