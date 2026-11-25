"""Lightweight execution engine orchestrating training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from reframx.execution.context import ExecutionContext
from reframx.execution.hooks.base import dispatch_hooks
from reframx.methods.interface import BaseMethod, MethodResult

if TYPE_CHECKING:
    from reframx.experiments.run import Run


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _average_metrics(records: list[dict[str, Any]], *, skip_keys: set[str]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for record in records:
        for key, value in record.items():
            if key in skip_keys or not _is_number(value):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def _normalize_result(raw: Any) -> MethodResult:
    if isinstance(raw, MethodResult):
        return raw
    if isinstance(raw, Mapping):
        return MethodResult(task_outputs=dict(raw))
    return MethodResult(task_outputs={"output": raw})


def _supports_eval(method: Any) -> bool:
    """Return True if ``method`` provides a concrete ``eval_step``."""
    eval_step = getattr(method, "eval_step", None)
    if not callable(eval_step):
        return False
    if isinstance(method, BaseMethod) and type(method).eval_step is BaseMethod.eval_step:
        return False
    return True


def _resolve_split_iterable(dataset: Any, split: str) -> Iterable[Any] | None:
    if dataset is None:
        return None
    get_split = getattr(dataset, "get_split", None)
    if callable(get_split):
        has_split = getattr(dataset, "has_split", None)
        if callable(has_split) and not has_split(split):
            return None
        try:
            batches = get_split(split)
        except KeyError:
            return None
        if not isinstance(batches, Iterable):
            raise TypeError(
                f"Dataset.get_split({split!r}) must return an Iterable; "
                f"got {type(batches).__name__}."
            )
        return batches
    if split == "train" and isinstance(dataset, Iterable):
        return dataset
    return None


class ExecutionEngine:
    """Orchestrates training, evaluation, and plugin hook lifecycles."""

    def __init__(self, ctx: ExecutionContext) -> None:
        self.ctx = ctx

    def evaluate(self, *, split: str = "val") -> dict[str, Any]:
        """Run one evaluation pass over ``split`` and return aggregated metrics."""
        ctx = self.ctx
        batches = _resolve_split_iterable(ctx.dataset, split)
        if batches is None:
            return {}

        if not _supports_eval(ctx.method):
            return {}
        eval_step = ctx.method.eval_step

        prev_phase = ctx.state.phase
        ctx.state.phase = "eval"
        ctx.strategy.set_model_mode(ctx.model, training=False)
        dispatch_hooks(ctx.hooks, "on_eval_start", ctx)

        evaluator = ctx.evaluator
        if evaluator is not None:
            reset_fn = getattr(evaluator, "reset", None)
            if callable(reset_fn):
                reset_fn()

        step_records: list[dict[str, Any]] = []
        with ctx.strategy.eval_context():
            for batch in batches:
                ctx.step_metrics.clear()
                prepared_batch = ctx.strategy.prepare_batch(batch, ctx)
                result = _normalize_result(eval_step(prepared_batch, ctx))
                metrics = result.resolved_metrics()
                metrics.update(ctx.step_metrics)
                if metrics:
                    step_records.append(metrics)
                if evaluator is not None:
                    evaluator.update(result, prepared_batch)

        aggregated: dict[str, Any] = _average_metrics(step_records, skip_keys=set())
        if evaluator is not None:
            computed = evaluator.compute()
            aggregated.update(dict(computed))

        if ctx.run is not None:
            ctx.run.eval_history.append(
                {
                    "epoch": ctx.state.epoch,
                    "step": ctx.state.global_step,
                    "split": split,
                    **aggregated,
                }
            )

        dispatch_hooks(ctx.hooks, "on_eval_end", ctx, aggregated)
        ctx.state.phase = prev_phase
        return aggregated

    def run(
        self,
        *,
        max_epochs: int = 1,
        max_steps: int | None = None,
        train_split: str = "train",
        eval_split: str | None = "val",
        eval_every_epochs: int = 1,
        catch_interrupt: bool = False,
    ) -> Run:
        """Execute training (and optional evaluation) and return the populated ``Run``."""
        if max_epochs < 1:
            raise ValueError("max_epochs must be >= 1.")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be >= 1 when specified.")
        if eval_every_epochs < 1:
            raise ValueError("eval_every_epochs must be >= 1.")

        ctx = self.ctx
        run: Run = ctx.run
        if ctx.method is None:
            raise ValueError("Cannot execute without a Method.")

        ctx.model = ctx.strategy.prepare_model(ctx.model)
        setup_fn = getattr(ctx.method, "setup", None)
        if callable(setup_fn):
            setup_fn(ctx)

        run.mark_running()
        dispatch_hooks(ctx.hooks, "on_run_start", ctx)

        try:
            for epoch in range(1, max_epochs + 1):
                ctx.state.epoch = epoch
                ctx.state.step_in_epoch = 0
                ctx.state.phase = "train"
                dispatch_hooks(ctx.hooks, "on_epoch_start", ctx)

                ctx.strategy.set_model_mode(ctx.model, training=True)
                train_batches = _resolve_split_iterable(ctx.dataset, train_split)
                if train_batches is None:
                    raise ValueError(
                        f"Dataset does not provide training split '{train_split}'."
                    )

                epoch_step_records: list[dict[str, Any]] = []
                for batch in train_batches:
                    ctx.state.global_step += 1
                    ctx.state.step_in_epoch += 1
                    ctx.step_metrics.clear()

                    prepared_batch = ctx.strategy.prepare_batch(batch, ctx)
                    dispatch_hooks(ctx.hooks, "on_step_start", ctx, prepared_batch)
                    result = _normalize_result(
                        ctx.method.train_step(prepared_batch, ctx)
                    )

                    step_record: dict[str, Any] = {
                        "epoch": epoch,
                        "step": ctx.state.global_step,
                        **result.resolved_metrics(),
                        **ctx.step_metrics,
                    }
                    run.train_history.append(step_record)
                    epoch_step_records.append(step_record)

                    dispatch_hooks(
                        ctx.hooks,
                        "on_step_end",
                        ctx,
                        prepared_batch,
                        result,
                    )

                    if ctx.state.should_stop or (
                        max_steps is not None and ctx.state.global_step >= max_steps
                    ):
                        break

                epoch_summary: dict[str, Any] = {
                    "epoch": epoch,
                    "step": ctx.state.global_step,
                    **{
                        f"train/{k}": v
                        for k, v in _average_metrics(
                            epoch_step_records,
                            skip_keys={"epoch", "step"},
                        ).items()
                    },
                }

                if eval_split is not None and epoch % eval_every_epochs == 0:
                    eval_metrics = self.evaluate(split=eval_split)
                    for k, v in eval_metrics.items():
                        epoch_summary[f"eval/{k}"] = v

                run.epoch_history.append(epoch_summary)
                dispatch_hooks(ctx.hooks, "on_epoch_end", ctx, epoch_summary)

                if ctx.state.should_stop or (
                    max_steps is not None and ctx.state.global_step >= max_steps
                ):
                    break

            ctx.state.phase = "idle"
            stop_reason = ctx.state.extra.get("_stop_reason")
            if ctx.state.should_stop and stop_reason:
                run.mark_interrupted(str(stop_reason))
            else:
                run.mark_completed()
        except KeyboardInterrupt:
            ctx.state.phase = "idle"
            run.mark_interrupted("KeyboardInterrupt")
            dispatch_hooks(ctx.hooks, "on_run_end", ctx, run)
            if not catch_interrupt:
                raise
            return run
        except BaseException as exc:
            ctx.state.phase = "idle"
            run.mark_failed(exc)
            dispatch_hooks(ctx.hooks, "on_run_end", ctx, run)
            raise

        dispatch_hooks(ctx.hooks, "on_run_end", ctx, run)
        return run
