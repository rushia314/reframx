"""Context-aware InstrumentationManager and Execution Hook integration."""

from __future__ import annotations

import contextlib
import contextvars
import fnmatch
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from reframx.execution.hooks.base import BaseHook
from reframx.instrumentation.events import (
    InstrumentationEvent,
    InstrumentationRecord,
    Stage,
    detach_value,
)
from reframx.instrumentation.module_hooks import (
    WatchHandle,
    attach_module_hooks,
)

EventCallback = Callable[[InstrumentationEvent], Mapping[str, Any] | None]
ReduceMode = Literal["last", "mean", "min", "max", "list"]
VALID_REDUCE_MODES: tuple[ReduceMode, ...] = ("last", "mean", "min", "max", "list")

_ACTIVE_MANAGER: contextvars.ContextVar[InstrumentationManager | None] = (
    contextvars.ContextVar("rf_active_instrumentation_manager", default=None)
)


def get_active_manager() -> InstrumentationManager | None:
    """Return the currently active ``InstrumentationManager``, if any."""
    return _ACTIVE_MANAGER.get()


@dataclass(slots=True)
class _ProbeSubscription:
    pattern: str
    callback: EventCallback
    every_n_steps: int
    phases: frozenset[str] | None
    detach: bool
    reduce: ReduceMode = "last"
    active: bool = True


def _reduce_values(values: list[Any], mode: ReduceMode) -> Any:
    if not values:
        return None
    if mode == "last":
        return values[-1]
    if mode == "list":
        return list(values)
    nums = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not nums:
        return values[-1]
    if mode == "mean":
        return sum(nums) / len(nums)
    if mode == "min":
        return min(nums)
    if mode == "max":
        return max(nums)
    return values[-1]


class InstrumentationManager:
    """Coordinates named probes, module watches, sampling, and recorded metrics."""

    def __init__(self, *, max_records: int | None = None) -> None:
        self._probe_subs: list[_ProbeSubscription] = []
        self._handles: list[WatchHandle] = []
        self._records: list[InstrumentationRecord] = []
        self._step_metrics_buffer: dict[str, tuple[ReduceMode, list[Any]]] = {}
        self._max_records = max_records

        self._epoch = 0
        self._step = 0
        self._phase = "idle"
        self._call_counts: dict[tuple[str, Stage], int] = {}

    @property
    def records(self) -> list[InstrumentationRecord]:
        return self._records

    def clear_records(self) -> None:
        self._records.clear()

    def pop_step_metrics(self) -> dict[str, Any]:
        """Reduce, return, and clear metrics captured during the current step."""
        reduced = {
            metric_key: _reduce_values(values, mode)
            for metric_key, (mode, values) in self._step_metrics_buffer.items()
        }
        self._step_metrics_buffer.clear()
        return reduced

    def set_step_context(
        self,
        *,
        epoch: int | None = None,
        step: int | None = None,
        phase: str | None = None,
    ) -> None:
        if epoch is not None:
            self._epoch = epoch
        if step is not None and step != self._step:
            self._step = step
            self._call_counts.clear()
        if phase is not None:
            self._phase = phase

    def _get_step_context(self) -> tuple[int, int, str]:
        return self._epoch, self._step, self._phase

    def _next_call_index(self, name: str, stage: Stage) -> int:
        key = (name, stage)
        idx = self._call_counts.get(key, 0)
        self._call_counts[key] = idx + 1
        return idx

    def _should_sample(
        self,
        *,
        every_n_steps: int,
        phases: frozenset[str] | None,
    ) -> bool:
        if phases is not None and self._phase not in phases:
            return False
        if every_n_steps > 1 and self._step > 0 and (self._step % every_n_steps != 0):
            return False
        return True

    def _record_result(
        self,
        event: InstrumentationEvent,
        metrics: Mapping[str, Any] | None,
        *,
        reduce: ReduceMode = "last",
    ) -> None:
        if not metrics:
            return
        metric_dict = dict(metrics)
        record = InstrumentationRecord(
            name=event.name,
            stage=event.stage,
            epoch=event.epoch,
            step=event.step,
            phase=event.phase,
            call_index=event.call_index,
            metrics=metric_dict,
        )
        self._records.append(record)
        if self._max_records is not None and len(self._records) > self._max_records:
            del self._records[: len(self._records) - self._max_records]
        for key, value in metric_dict.items():
            full_key = f"{event.name}/{key}"
            if full_key not in self._step_metrics_buffer:
                self._step_metrics_buffer[full_key] = (reduce, [value])
            else:
                self._step_metrics_buffer[full_key][1].append(value)

    def watch_probe(
        self,
        pattern: str,
        callback: EventCallback,
        *,
        every_n_steps: int = 1,
        phases: Iterable[str] | None = None,
        detach: bool = True,
        reduce: ReduceMode = "last",
    ) -> WatchHandle:
        """Subscribe ``callback`` to probes matching ``pattern`` (exact or glob)."""
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1.")
        if reduce not in VALID_REDUCE_MODES:
            raise ValueError(
                f"Unsupported reduce mode {reduce!r}; expected one of {VALID_REDUCE_MODES}."
            )
        sub = _ProbeSubscription(
            pattern=pattern,
            callback=callback,
            every_n_steps=every_n_steps,
            phases=frozenset(phases) if phases is not None else None,
            detach=detach,
            reduce=reduce,
        )
        self._probe_subs.append(sub)

        def _remove() -> None:
            sub.active = False
            if sub in self._probe_subs:
                self._probe_subs.remove(sub)

        handle = WatchHandle(_remove)
        self._handles.append(handle)
        return handle

    def emit_probe(
        self,
        name: str,
        value: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Emit a probe event if listeners match and always return ``value`` untouched."""
        if not self._probe_subs:
            return value

        matching = [
            sub
            for sub in self._probe_subs
            if sub.active
            and fnmatch.fnmatchcase(name, sub.pattern)
            and self._should_sample(
                every_n_steps=sub.every_n_steps,
                phases=sub.phases,
            )
        ]
        if not matching:
            return value

        call_index = self._next_call_index(name, "probe")
        detached_cache: Any = None
        has_detached = False

        for sub in matching:
            if sub.detach:
                if not has_detached:
                    detached_cache = detach_value(value)
                    has_detached = True
                event_value = detached_cache
            else:
                event_value = value

            event = InstrumentationEvent(
                name=name,
                stage="probe",
                value=event_value,
                epoch=self._epoch,
                step=self._step,
                phase=self._phase,
                call_index=call_index,
                metadata=dict(metadata or {}),
            )
            self._record_result(event, sub.callback(event), reduce=sub.reduce)

        return value

    def watch_module(
        self,
        module: Any,
        *,
        callback: EventCallback,
        stages: Iterable[Stage] = ("post_forward",),
        name: str | None = None,
        every_n_steps: int = 1,
        phases: Iterable[str] | None = None,
        detach: bool = True,
        reduce: ReduceMode = "last",
    ) -> WatchHandle:
        """Attach hooks to a single module and return its removable ``WatchHandle``."""
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1.")
        if reduce not in VALID_REDUCE_MODES:
            raise ValueError(
                f"Unsupported reduce mode {reduce!r}; expected one of {VALID_REDUCE_MODES}."
            )
        phase_set = frozenset(phases) if phases is not None else None
        target = getattr(module, "module", module)
        resolved_name = name or type(target).__name__

        handle = attach_module_hooks(
            target,
            name=resolved_name,
            stages=stages,
            emit_fn=lambda event: self._record_result(
                event, callback(event), reduce=reduce
            ),
            should_sample=lambda: self._should_sample(
                every_n_steps=every_n_steps,
                phases=phase_set,
            ),
            next_call_index=self._next_call_index,
            get_step_context=self._get_step_context,
            detach=detach,
        )
        self._handles.append(handle)
        return handle

    def watch_modules(
        self,
        root_module: Any,
        *,
        pattern: str = "*",
        callback: EventCallback,
        stages: Iterable[Stage] = ("post_forward",),
        every_n_steps: int = 1,
        phases: Iterable[str] | None = None,
        detach: bool = True,
        reduce: ReduceMode = "last",
    ) -> list[WatchHandle]:
        """Attach hooks to every submodule whose name matches glob ``pattern``."""
        target = getattr(root_module, "module", root_module)
        named_modules = getattr(target, "named_modules", None)
        if not callable(named_modules):
            raise TypeError(
                f"Object {type(target).__name__} does not define named_modules()."
            )
        handles: list[WatchHandle] = []
        for mod_name, submodule in named_modules():
            if not mod_name:
                continue
            if fnmatch.fnmatchcase(mod_name, pattern):
                handles.append(
                    self.watch_module(
                        submodule,
                        callback=callback,
                        stages=stages,
                        name=mod_name,
                        every_n_steps=every_n_steps,
                        phases=phases,
                        detach=detach,
                        reduce=reduce,
                    )
                )
        return handles

    @contextlib.contextmanager
    def activate(
        self,
        *,
        epoch: int | None = None,
        step: int | None = None,
        phase: str | None = None,
    ) -> Generator[InstrumentationManager, None, None]:
        """Activate this manager in the current context so ``rf.probe()`` finds it."""
        self.set_step_context(epoch=epoch, step=step, phase=phase)
        token = _ACTIVE_MANAGER.set(self)
        try:
            yield self
        finally:
            _ACTIVE_MANAGER.reset(token)

    def close(self) -> None:
        """Remove all registered module hooks and probe subscriptions."""
        for handle in list(self._handles):
            handle.remove()
        self._handles.clear()
        self._probe_subs.clear()

    def __enter__(self) -> InstrumentationManager:
        self._ctx_token: contextvars.Token[InstrumentationManager | None] | None = (
            _ACTIVE_MANAGER.set(self)
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        token = getattr(self, "_ctx_token", None)
        if token is not None:
            _ACTIVE_MANAGER.reset(token)
            self._ctx_token = None
        self.close()


class InstrumentationHook(BaseHook):
    """Connects an ``InstrumentationManager`` to an ``Experiment`` run lifecycle."""

    def __init__(
        self,
        manager: InstrumentationManager,
        *,
        log_to_run: bool = True,
        auto_close: bool = True,
    ) -> None:
        self.manager = manager
        self.log_to_run = log_to_run
        self.auto_close = auto_close
        self._token: contextvars.Token[InstrumentationManager | None] | None = None

    def on_run_start(self, ctx: Any) -> None:
        self._token = _ACTIVE_MANAGER.set(self.manager)
        if ctx.run is not None:
            ctx.run.artifacts["instrumentation"] = self.manager.records

    def on_step_start(self, ctx: Any, batch: Any) -> None:
        self.manager.set_step_context(
            epoch=ctx.state.epoch,
            step=ctx.state.global_step,
            phase="train",
        )

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        step_metrics = self.manager.pop_step_metrics()
        if self.log_to_run and step_metrics and ctx.run is not None and ctx.run.train_history:
            ctx.run.train_history[-1].update(step_metrics)

    def on_eval_start(self, ctx: Any) -> None:
        self.manager.set_step_context(
            epoch=ctx.state.epoch,
            step=ctx.state.global_step,
            phase="eval",
        )

    def on_run_end(self, ctx: Any, run: Any) -> None:
        if self._token is not None:
            _ACTIVE_MANAGER.reset(self._token)
            self._token = None
        if self.auto_close:
            self.manager.close()
