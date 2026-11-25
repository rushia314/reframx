"""MMEngine execution backend implementing ``ExecutionBackend``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from reframx.execution.context import ExecutionContext
from reframx.execution.engine import _supports_eval
from reframx.integrations.mmengine.bridges import (
    _GracefulStopError,
    build_mm_hook,
    build_mm_metric,
    build_mm_model,
)
from reframx.integrations.mmengine.strategy import MMEngineStrategy
from reframx.integrations.mmengine.utils import require_mmengine_submodule

if TYPE_CHECKING:
    from reframx.experiments.run import Run


def _resolve_dataloader(dataset: Any, split: str) -> Any:
    if dataset is None:
        return None
    get_split = getattr(dataset, "get_split", None)
    if callable(get_split):
        has_split = getattr(dataset, "has_split", None)
        if callable(has_split) and not has_split(split):
            return None
        try:
            return get_split(split)
        except KeyError:
            return None
    if split == "train" and isinstance(dataset, Iterable):
        return dataset
    return None


def _build_optim_wrapper(optimizers: Mapping[str, Any]) -> Any:
    if not optimizers:
        return None
    mm_optim = require_mmengine_submodule("optim")
    optim_wrapper_cls = mm_optim.OptimWrapper
    if len(optimizers) == 1:
        opt = next(iter(optimizers.values()))
        if isinstance(opt, optim_wrapper_cls):
            return opt
        return optim_wrapper_cls(optimizer=opt)

    optim_dict_cls = mm_optim.OptimWrapperDict
    wrapped = {
        name: (opt if isinstance(opt, optim_wrapper_cls) else optim_wrapper_cls(optimizer=opt))
        for name, opt in optimizers.items()
    }
    return optim_dict_cls(**wrapped)


class MMEngineBackend:
    """Optional ``ExecutionBackend`` powered by ``mmengine.runner.Runner``.

    Enables DDP/FSDP, ``OptimWrapper``, ``ParamScheduler``, ``CheckpointHook``,
    and TensorBoard/WandB ``Visualizer`` without changing user ``Task``,
    ``Method``, ``Model``, ``Dataset``, ``Objective``, or ``Evaluator`` code.
    """

    def __init__(
        self,
        *,
        work_dir: str = "./work_dirs/reframx",
        launcher: str = "none",
        by_epoch: bool = True,
        optim_wrapper: Any | None = None,
        param_scheduler: Any | None = None,
        default_hooks: Mapping[str, Any] | None = None,
        custom_hooks: Iterable[Any] = (),
        visualizer: Any | None = None,
        resume: bool = False,
        load_from: str | None = None,
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.launcher = launcher
        self.by_epoch = by_epoch
        self.optim_wrapper = optim_wrapper
        self.param_scheduler = param_scheduler
        self.default_hooks = dict(default_hooks) if default_hooks is not None else None
        self.custom_hooks = list(custom_hooks)
        self.visualizer = visualizer
        self.resume = resume
        self.load_from = load_from
        self.cfg = dict(cfg) if cfg is not None else None

    def _create_runner(
        self,
        ctx: ExecutionContext,
        *,
        train_dataloader: Any = None,
        val_dataloader: Any = None,
        train_cfg: dict[str, Any] | None = None,
        val_cfg: dict[str, Any] | None = None,
    ) -> Any:
        mm_runner = require_mmengine_submodule("runner")
        mm_strategy = MMEngineStrategy(base_strategy=ctx.strategy)
        ctx.strategy = mm_strategy
        ctx.model = mm_strategy.prepare_model(ctx.model)

        mm_model = build_mm_model(ctx, mm_strategy)
        optim_wrapper = (
            self.optim_wrapper
            if self.optim_wrapper is not None
            else _build_optim_wrapper(ctx.optimizers)
        )

        val_evaluator = None
        if val_dataloader is not None and ctx.evaluator is not None:
            val_evaluator = [build_mm_metric(ctx.evaluator, ctx)]

        bridged_hooks = [build_mm_hook(hook, ctx) for hook in ctx.hooks]
        all_custom_hooks = [*bridged_hooks, *self.custom_hooks]

        runner_kwargs: dict[str, Any] = {
            "model": mm_model,
            "work_dir": self.work_dir,
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "val_evaluator": val_evaluator,
            "train_cfg": train_cfg,
            "val_cfg": val_cfg,
            "optim_wrapper": optim_wrapper,
            "param_scheduler": self.param_scheduler,
            "default_hooks": self.default_hooks,
            "custom_hooks": all_custom_hooks or None,
            "visualizer": self.visualizer,
            "resume": self.resume,
            "load_from": self.load_from,
            "launcher": self.launcher,
            "cfg": self.cfg,
        }
        return mm_runner.Runner(**runner_kwargs)

    def run(
        self,
        ctx: ExecutionContext,
        *,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        train_split: str = "train",
        eval_split: str | None = "val",
        eval_every_epochs: int | None = None,
        eval_every_steps: int | None = None,
        start_step: int = 0,
        start_epoch: int = 0,
        keep_eval_mode: bool = False,
        max_train_history: int | None = None,
        catch_interrupt: bool = False,
    ) -> Run:
        if ctx.method is None:
            raise ValueError("Cannot execute without a Method.")
        ctx.state.global_step = start_step
        ctx.state.epoch = start_epoch
        ctx.state.extra["_max_train_history"] = max_train_history
        run: Run = ctx.run

        train_dl = _resolve_dataloader(ctx.dataset, train_split)
        if train_dl is None:
            raise ValueError(
                f"Dataset does not provide training split '{train_split}'."
            )

        val_dl = (
            _resolve_dataloader(ctx.dataset, eval_split)
            if eval_split is not None and _supports_eval(ctx.method)
            else None
        )

        if self.by_epoch and max_steps is None:
            train_cfg: dict[str, Any] = {
                "by_epoch": True,
                "max_epochs": max_epochs if max_epochs is not None else 1,
                "val_interval": eval_every_epochs if eval_every_epochs is not None else 1,
            }
        else:
            train_cfg = {
                "by_epoch": False,
                "max_iters": max_steps if max_steps is not None else (max_epochs or 1),
                "val_interval": (
                    eval_every_steps
                    if eval_every_steps is not None
                    else (eval_every_epochs if eval_every_epochs is not None else 1)
                ),
            }

        val_cfg: dict[str, Any] | None = {} if val_dl is not None else None

        setup_fn = getattr(ctx.method, "setup", None)
        if callable(setup_fn):
            setup_fn(ctx)

        run.mark_running()
        try:
            runner = self._create_runner(
                ctx,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                train_cfg=train_cfg,
                val_cfg=val_cfg,
            )
            runner.train()
            ctx.state.phase = "idle"
            run.mark_completed()
        except _GracefulStopError:
            ctx.state.phase = "idle"
            stop_reason = ctx.state.extra.get("_stop_reason")
            if stop_reason:
                run.mark_interrupted(str(stop_reason))
            else:
                run.mark_completed()
        except KeyboardInterrupt:
            ctx.state.phase = "idle"
            run.mark_interrupted("KeyboardInterrupt")
            if not catch_interrupt:
                raise
        except BaseException as exc:
            ctx.state.phase = "idle"
            run.mark_failed(exc)
            from reframx.execution.hooks.base import dispatch_hooks

            dispatch_hooks(ctx.hooks, "on_exception", ctx, exc)
            raise

        return run

    def evaluate(
        self,
        ctx: ExecutionContext,
        *,
        split: str = "val",
    ) -> dict[str, Any]:
        if not _supports_eval(ctx.method):
            return {}
        val_dl = _resolve_dataloader(ctx.dataset, split)
        if val_dl is None:
            return {}
        runner = self._create_runner(
            ctx,
            val_dataloader=val_dl,
            val_cfg={},
        )
        metrics = runner.val()
        return dict(metrics) if isinstance(metrics, Mapping) else {}
