"""Batteries-included, zero-dependency lifecycle hooks for everyday research."""

from __future__ import annotations

import json
import os
import pickle
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from reframx.execution.hooks.base import BaseHook


def _collect_step_metrics(ctx: Any, result: Any) -> dict[str, Any]:
    """Merge step metrics from train_history[-1], result, and ctx.step_metrics."""
    merged: dict[str, Any] = {}
    run = getattr(ctx, "run", None)
    history = getattr(run, "train_history", None)
    if history:
        latest = history[-1]
        if (
            isinstance(latest, Mapping)
            and latest.get("step") == ctx.state.global_step
        ):
            for k, v in latest.items():
                if k not in ("epoch", "step"):
                    merged[k] = v
    if hasattr(result, "resolved_metrics"):
        merged.update(result.resolved_metrics())
    step_metrics = getattr(ctx, "step_metrics", None)
    if isinstance(step_metrics, Mapping):
        merged.update(step_metrics)
    return merged


def _extract_state(obj: Any) -> Any | None:
    if obj is None:
        return None
    get_state = getattr(obj, "get_component_state", None)
    if callable(get_state):
        return get_state()
    state_dict = getattr(obj, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return None


def _atomic_save_state(payload: dict[str, Any], target: Path) -> Path:
    """Atomically write checkpoint payload to ``target`` via temp file + os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        if target.suffix == ".json":
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        else:
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and callable(getattr(torch_mod, "save", None)):
                torch_mod.save(payload, tmp_path)
            else:
                try:
                    tmp_path.write_text(
                        json.dumps(payload, sort_keys=True),
                        encoding="utf-8",
                    )
                except TypeError:
                    with tmp_path.open("wb") as f:
                        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return target


def load_checkpoint(
    path: str | Path,
    *,
    allow_pickle: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint saved by ``CheckpointHook`` (ready for ``resume_from=``)."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {src}")

    raw = src.read_bytes()
    # Try JSON first (safe and human-readable).
    try:
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and callable(getattr(torch_mod, "load", None)):
        loaded = torch_mod.load(src, weights_only=not allow_pickle)
        if isinstance(loaded, dict):
            return loaded

    if not allow_pickle:
        raise ValueError(
            f"Checkpoint '{src}' requires pickle deserialization; "
            "pass allow_pickle=True if you trust this file."
        )
    loaded = pickle.loads(raw)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected checkpoint dict; got {type(loaded).__name__}.")
    return loaded


class CheckpointHook(BaseHook):
    """Atomic checkpoint saver supporting step/epoch intervals, best tracking, and OOM rescue.

    Saves model/method/dataset/strategy/optimizer states along with ``epoch``,
    ``global_step``, and ``metrics`` in a format directly loadable via
    ``experiment.run(resume_from=load_checkpoint(path))``.
    """

    def __init__(
        self,
        save_dir: str | Path,
        *,
        save_every_steps: int | None = None,
        save_every_epochs: int | None = 1,
        monitor: str | None = None,
        mode: Literal["min", "max"] = "min",
        save_best: bool = True,
        save_last: bool = True,
        save_on_exception: bool = True,
        filename_ext: str = ".pt",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        self.save_dir = Path(save_dir)
        self.save_every_steps = save_every_steps
        self.save_every_epochs = save_every_epochs
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_on_exception = save_on_exception
        self.filename_ext = filename_ext if filename_ext.startswith(".") else f".{filename_ext}"

        self.best_value: float | None = None
        self.best_path: Path | None = None
        self.last_path: Path | None = None

    def _build_payload(self, ctx: Any, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        components: dict[str, Any] = {}
        for role, comp in (
            ("model", ctx.model),
            ("method", ctx.method),
            ("dataset", ctx.dataset),
            ("strategy", ctx.strategy),
        ):
            state = _extract_state(comp)
            if state is not None:
                components[role] = state

        optimizers: dict[str, Any] = {}
        for opt_name, opt in getattr(ctx, "optimizers", {}).items():
            state = _extract_state(opt)
            if state is not None:
                optimizers[opt_name] = state

        return {
            "epoch": ctx.state.epoch,
            "global_step": ctx.state.global_step,
            "metrics": dict(metrics or {}),
            "components": components,
            "optimizers": optimizers,
        }

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def _maybe_save_best(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        if not self.save_best or self.monitor is None:
            return
        raw_val = metrics.get(self.monitor)
        if not isinstance(raw_val, (int, float)) or isinstance(raw_val, bool):
            return
        val = float(raw_val)
        if self._is_better(val):
            self.best_value = val
            target = self.save_dir / f"best{self.filename_ext}"
            self.best_path = _atomic_save_state(self._build_payload(ctx, metrics), target)

    def _save_last(self, ctx: Any, metrics: Mapping[str, Any] | None = None) -> None:
        if not self.save_last:
            return
        target = self.save_dir / f"last{self.filename_ext}"
        self.last_path = _atomic_save_state(self._build_payload(ctx, metrics), target)

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        if (
            self.save_every_steps is not None
            and ctx.state.global_step > 0
            and ctx.state.global_step % self.save_every_steps == 0
        ):
            metrics = _collect_step_metrics(ctx, result)
            target = self.save_dir / f"step_{ctx.state.global_step}{self.filename_ext}"
            _atomic_save_state(self._build_payload(ctx, metrics), target)
            self._save_last(ctx, metrics)
            self._maybe_save_best(ctx, metrics)

    def on_eval_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        prefixed = {**metrics, **{f"eval/{k}": v for k, v in metrics.items()}}
        self._maybe_save_best(ctx, prefixed)

    def on_epoch_end(self, ctx: Any, epoch_metrics: Mapping[str, Any]) -> None:
        if (
            self.save_every_epochs is not None
            and ctx.state.epoch > 0
            and ctx.state.epoch % self.save_every_epochs == 0
        ):
            self._save_last(ctx, epoch_metrics)
            self._maybe_save_best(ctx, epoch_metrics)

    def on_exception(self, ctx: Any, exc: BaseException) -> None:
        if self.save_on_exception:
            target = self.save_dir / f"emergency_step_{ctx.state.global_step}{self.filename_ext}"
            _atomic_save_state(
                self._build_payload(ctx, {"exception": f"{type(exc).__name__}: {exc}"}),
                target,
            )


class JSONLLoggerHook(BaseHook):
    """Streams step, evaluation, and epoch metrics to a ``.jsonl`` file on disk."""

    def __init__(self, log_path: str | Path, *, flush_every: int = 1) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be >= 1.")
        self.log_path = Path(log_path)
        self.flush_every = flush_every
        self._fp: Any = None
        self._writes = 0

    def _write_line(self, record: Mapping[str, Any]) -> None:
        if self._fp is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.log_path.open("a", encoding="utf-8")
        self._fp.write(json.dumps(dict(record), default=str) + "\n")
        self._writes += 1
        if self._writes % self.flush_every == 0:
            self._fp.flush()

    def on_run_start(self, ctx: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.log_path.open("a", encoding="utf-8")

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        self._write_line(
            {
                "event": "train",
                "epoch": ctx.state.epoch,
                "step": ctx.state.global_step,
                **_collect_step_metrics(ctx, result),
            }
        )

    def on_eval_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        self._write_line(
            {
                "event": "eval",
                "epoch": ctx.state.epoch,
                "step": ctx.state.global_step,
                **dict(metrics),
            }
        )

    def on_epoch_end(self, ctx: Any, epoch_metrics: Mapping[str, Any]) -> None:
        self._write_line({"event": "epoch", **dict(epoch_metrics)})

    def on_run_end(self, ctx: Any, run: Any) -> None:
        if self._fp is not None:
            self._fp.flush()
            self._fp.close()
            self._fp = None


class ConsoleLoggerHook(BaseHook):
    """Lightweight console progress logger for step-based and epoch-based runs."""

    def __init__(
        self,
        *,
        log_every_steps: int | None = None,
        log_every_epochs: int | None = 1,
        printer: Callable[[str], None] = print,
    ) -> None:
        self.log_every_steps = log_every_steps
        self.log_every_epochs = log_every_epochs
        self.printer = printer

    @staticmethod
    def _fmt(metrics: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for k, v in metrics.items():
            if k in ("epoch", "step"):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        return " | ".join(parts)

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        if (
            self.log_every_steps is not None
            and ctx.state.global_step % self.log_every_steps == 0
        ):
            merged = _collect_step_metrics(ctx, result)
            self.printer(
                f"[Step {ctx.state.global_step}] {self._fmt(merged)}"
            )

    def on_eval_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        if metrics:
            self.printer(
                f"[Eval step={ctx.state.global_step}] {self._fmt(metrics)}"
            )

    def on_epoch_end(self, ctx: Any, epoch_metrics: Mapping[str, Any]) -> None:
        if (
            self.log_every_epochs is not None
            and ctx.state.epoch % self.log_every_epochs == 0
        ):
            self.printer(
                f"[Epoch {ctx.state.epoch}] {self._fmt(epoch_metrics)}"
            )


class FileControlHook(BaseHook):
    """Watches local JSON files for live stop signals and hyperparameter hot-reloading.

    - ``control_path`` (e.g. ``control.json``): ``{"stop": true, "reason": "manual"}``
      triggers a graceful ``ctx.request_stop()``.
    - ``hparams_path`` (e.g. ``live_hparams.json``): when modified, updates
      ``lr`` on registered optimizers (``{"lr": 1e-4}`` or
      ``{"lr": {"opt_name": 1e-4}}``) and invokes optional ``on_hparams(data, ctx)``.
    """

    def __init__(
        self,
        *,
        control_path: str | Path | None = None,
        hparams_path: str | Path | None = None,
        check_every_steps: int = 1,
        on_hparams: Callable[[Mapping[str, Any], Any], None] | None = None,
    ) -> None:
        if check_every_steps < 1:
            raise ValueError("check_every_steps must be >= 1.")
        self.control_path = Path(control_path) if control_path is not None else None
        self.hparams_path = Path(hparams_path) if hparams_path is not None else None
        self.check_every_steps = check_every_steps
        self.on_hparams = on_hparams
        self._last_hparams_mtime: float | None = None

    def _apply_hparams(self, data: Mapping[str, Any], ctx: Any) -> None:
        lr_val = data.get("lr")
        if isinstance(lr_val, (int, float)):
            for opt in getattr(ctx, "optimizers", {}).values():
                for group in getattr(opt, "param_groups", ()):
                    group["lr"] = float(lr_val)
                if hasattr(opt, "lr"):
                    opt.lr = float(lr_val)
        elif isinstance(lr_val, Mapping):
            for opt_name, new_lr in lr_val.items():
                opt = getattr(ctx, "optimizers", {}).get(opt_name)
                if opt is None:
                    continue
                for group in getattr(opt, "param_groups", ()):
                    group["lr"] = float(new_lr)
                if hasattr(opt, "lr"):
                    opt.lr = float(new_lr)

        if self.on_hparams is not None:
            self.on_hparams(data, ctx)

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        if ctx.state.global_step % self.check_every_steps != 0:
            return

        if self.control_path is not None and self.control_path.is_file():
            try:
                ctrl = json.loads(self.control_path.read_text(encoding="utf-8"))
            except Exception:
                ctrl = {}
            if isinstance(ctrl, Mapping) and ctrl.get("stop"):
                ctx.state.extra["_stop_reason"] = str(
                    ctrl.get("reason", "Stopped via control file")
                )
                ctx.request_stop()

        if self.hparams_path is not None and self.hparams_path.is_file():
            mtime = self.hparams_path.stat().st_mtime
            if self._last_hparams_mtime is None or mtime > self._last_hparams_mtime:
                self._last_hparams_mtime = mtime
                try:
                    hp = json.loads(self.hparams_path.read_text(encoding="utf-8"))
                except Exception:
                    hp = None
                if isinstance(hp, Mapping):
                    self._apply_hparams(hp, ctx)
                    ctx.state.extra["live_hparams"] = dict(hp)
