"""Bridge adapters translating Research Framework components for MMEngine Runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reframx.execution.context import ExecutionContext
from reframx.integrations.mmengine.strategy import MMEngineStrategy
from reframx.integrations.mmengine.utils import require_mmengine_submodule
from reframx.methods.interface import MethodResult


class _GracefulStopError(Exception):
    """Internal control-flow signal raised when ``ctx.request_stop()`` is set."""


def _normalize_result(raw: Any) -> MethodResult:
    if isinstance(raw, MethodResult):
        return raw
    if isinstance(raw, Mapping):
        return MethodResult(task_outputs=dict(raw))
    return MethodResult(task_outputs={"output": raw})


def build_mm_model(ctx: ExecutionContext, mm_strategy: MMEngineStrategy) -> Any:
    """Wrap ``(ctx.model, ctx.method)`` as an ``mmengine.model.BaseModel``."""
    base_model_cls = require_mmengine_submodule("model").BaseModel

    class _RFMethodMMModel(base_model_cls):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._rf_ctx = ctx
            self._rf_strategy = mm_strategy
            # Expose underlying nn.Module so MMEngine / DDP / checkpoints find params.
            raw_module = getattr(ctx.model, "module", ctx.model)
            self.inner_model = raw_module

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self._rf_ctx.model(*args, **kwargs)

        def train_step(self, data: Any, optim_wrapper: Any) -> dict[str, Any]:
            self._rf_ctx.state.phase = "train"
            self._rf_ctx.state.global_step += 1
            self._rf_ctx.state.step_in_epoch += 1
            self._rf_ctx.step_metrics.clear()

            self._rf_strategy.bind_optim_wrapper(optim_wrapper)
            if isinstance(optim_wrapper, Mapping):
                self._rf_ctx.optimizers = dict(optim_wrapper)
            elif optim_wrapper is not None:
                self._rf_ctx.optimizers = {"default": optim_wrapper}

            prepared = self._rf_strategy.prepare_batch(data, self._rf_ctx)
            result = _normalize_result(
                self._rf_ctx.method.train_step(prepared, self._rf_ctx)
            )
            metrics = {
                **result.resolved_metrics(),
                **self._rf_ctx.step_metrics,
            }
            if self._rf_ctx.run is not None:
                self._rf_ctx.run.train_history.append(
                    {
                        "epoch": self._rf_ctx.state.epoch,
                        "step": self._rf_ctx.state.global_step,
                        **metrics,
                    }
                )
            self._rf_ctx.state.extra["_last_result"] = result
            if self._rf_ctx.state.should_stop:
                raise _GracefulStopError()
            return metrics

        def val_step(self, data: Any) -> list[dict[str, Any]]:
            self._rf_ctx.state.phase = "eval"
            self._rf_ctx.step_metrics.clear()
            prepared = self._rf_strategy.prepare_batch(data, self._rf_ctx)
            result = _normalize_result(
                self._rf_ctx.method.eval_step(prepared, self._rf_ctx)
            )
            return [{"result": result, "batch": prepared}]

        def test_step(self, data: Any) -> list[dict[str, Any]]:
            return self.val_step(data)

    return _RFMethodMMModel()


def build_mm_metric(evaluator: Any, ctx: ExecutionContext) -> Any:
    """Wrap a Research Framework Evaluator as an ``mmengine.evaluator.BaseMetric``."""
    base_metric_cls = require_mmengine_submodule("evaluator").BaseMetric

    class _RFEvaluatorMMMetric(base_metric_cls):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            self._rf_evaluator = evaluator
            self._rf_ctx = ctx
            self._reset_needed = True

        def process(self, data_batch: Any, data_samples: Any) -> None:
            if self._reset_needed:
                reset_fn = getattr(self._rf_evaluator, "reset", None)
                if callable(reset_fn):
                    reset_fn()
                self._reset_needed = False

            samples = data_samples if isinstance(data_samples, list) else [data_samples]
            for sample in samples:
                if isinstance(sample, Mapping) and "result" in sample:
                    self._rf_evaluator.update(sample["result"], sample.get("batch", data_batch))
                else:
                    self._rf_evaluator.update(sample, data_batch)
                # MMEngine BaseMetric.evaluate() skips compute_metrics() if self.results is empty.
                self.results.append(1)

        def compute_metrics(self, results: list[Any]) -> dict[str, Any]:
            self._reset_needed = True
            computed = dict(self._rf_evaluator.compute())
            if self._rf_ctx.run is not None:
                self._rf_ctx.run.eval_history.append(
                    {
                        "epoch": self._rf_ctx.state.epoch,
                        "step": self._rf_ctx.state.global_step,
                        **computed,
                    }
                )
            return computed

    return _RFEvaluatorMMMetric()


def build_mm_hook(hook: Any, ctx: ExecutionContext) -> Any:
    """Wrap a Research Framework Hook (or plain object) as an ``mmengine.hooks.Hook``."""
    mm_hook_cls = require_mmengine_submodule("hooks").Hook

    def _call(name: str, *args: Any) -> None:
        fn = getattr(hook, name, None)
        if callable(fn):
            fn(*args)

    class _RFHookMMBridge(mm_hook_cls):  # type: ignore[misc,valid-type]
        def before_run(self, runner: Any) -> None:
            _call("on_run_start", ctx)

        def before_train_epoch(self, runner: Any) -> None:
            ctx.state.epoch = getattr(runner, "epoch", ctx.state.epoch) + 1
            ctx.state.step_in_epoch = 0
            _call("on_epoch_start", ctx)

        def before_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
        ) -> None:
            _call("on_step_start", ctx, data_batch)

        def after_train_iter(
            self,
            runner: Any,
            batch_idx: int,
            data_batch: Any = None,
            outputs: Any = None,
        ) -> None:
            result = ctx.state.extra.get("_last_result", outputs)
            _call("on_step_end", ctx, data_batch, result)

        def before_val_epoch(self, runner: Any) -> None:
            _call("on_eval_start", ctx)

        def after_val_epoch(
            self,
            runner: Any,
            metrics: dict[str, Any] | None = None,
        ) -> None:
            _call("on_eval_end", ctx, metrics or {})

        def after_train_epoch(self, runner: Any) -> None:
            summary = {"epoch": ctx.state.epoch, "step": ctx.state.global_step}
            _call("on_epoch_end", ctx, summary)

        def after_run(self, runner: Any) -> None:
            _call("on_run_end", ctx, ctx.run)

    return _RFHookMMBridge()
