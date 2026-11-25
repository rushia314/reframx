"""Non-invasive module hook attachment and lifecycle handles."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from reframx.instrumentation.events import InstrumentationEvent, Stage, detach_value

VALID_MODULE_STAGES: tuple[Stage, ...] = ("pre_forward", "post_forward", "backward")


class WatchHandle:
    """Removable handle for a registered probe or module watch."""

    def __init__(self, remover: Callable[[], None]) -> None:
        self._remover = remover
        self._removed = False

    @property
    def is_active(self) -> bool:
        return not self._removed

    def remove(self) -> None:
        if not self._removed:
            self._removed = True
            self._remover()

    def __enter__(self) -> WatchHandle:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.remove()


def attach_module_hooks(
    module: Any,
    *,
    name: str,
    stages: Iterable[Stage],
    emit_fn: Callable[[InstrumentationEvent], None],
    should_sample: Callable[[], bool],
    next_call_index: Callable[[str, Stage], int],
    get_step_context: Callable[[], tuple[int, int, str]],
    detach: bool = True,
) -> WatchHandle:
    """Attach forward/backward hooks to ``module`` and return a ``WatchHandle``."""
    stage_tuple = tuple(stages)
    for stage in stage_tuple:
        if stage not in VALID_MODULE_STAGES:
            raise ValueError(
                f"Unsupported module watch stage {stage!r}; "
                f"expected one of {VALID_MODULE_STAGES}."
            )

    raw_handles: list[Any] = []

    def _maybe_detach(val: Any) -> Any:
        return detach_value(val) if detach else val

    if "pre_forward" in stage_tuple:
        reg_pre = getattr(module, "register_forward_pre_hook", None)
        if not callable(reg_pre):
            raise TypeError(
                f"Module {type(module).__name__} does not support "
                "register_forward_pre_hook."
            )

        def _pre_hook(mod: Any, args: Any, kwargs: Mapping[str, Any] | None = None) -> None:
            if not should_sample():
                return
            epoch, step, phase = get_step_context()
            event = InstrumentationEvent(
                name=name,
                stage="pre_forward",
                value=_maybe_detach(args),
                inputs=_maybe_detach(args),
                kwargs=_maybe_detach(kwargs or {}),
                module=mod,
                epoch=epoch,
                step=step,
                phase=phase,
                call_index=next_call_index(name, "pre_forward"),
            )
            emit_fn(event)

        try:
            raw_handles.append(reg_pre(_pre_hook, with_kwargs=True))
        except TypeError:
            raw_handles.append(reg_pre(lambda mod, args: _pre_hook(mod, args, {})))

    if "post_forward" in stage_tuple:
        reg_post = getattr(module, "register_forward_hook", None)
        if not callable(reg_post):
            raise TypeError(
                f"Module {type(module).__name__} does not support "
                "register_forward_hook."
            )

        def _post_hook(
            mod: Any,
            args: Any,
            kwargs_or_out: Any,
            maybe_out: Any = None,
            *,
            has_kwargs: bool = False,
        ) -> None:
            if not should_sample():
                return
            kwargs = kwargs_or_out if has_kwargs else {}
            output = maybe_out if has_kwargs else kwargs_or_out
            epoch, step, phase = get_step_context()
            detached_out = _maybe_detach(output)
            event = InstrumentationEvent(
                name=name,
                stage="post_forward",
                value=detached_out,
                inputs=_maybe_detach(args),
                kwargs=_maybe_detach(kwargs or {}),
                output=detached_out,
                module=mod,
                epoch=epoch,
                step=step,
                phase=phase,
                call_index=next_call_index(name, "post_forward"),
            )
            emit_fn(event)

        try:
            raw_handles.append(
                reg_post(
                    lambda mod, args, kwargs, out: _post_hook(
                        mod, args, kwargs, out, has_kwargs=True
                    ),
                    with_kwargs=True,
                )
            )
        except TypeError:
            raw_handles.append(
                reg_post(
                    lambda mod, args, out: _post_hook(
                        mod, args, out, has_kwargs=False
                    )
                )
            )

    if "backward" in stage_tuple:
        reg_bwd = getattr(module, "register_full_backward_hook", None) or getattr(
            module, "register_backward_hook", None
        )
        if not callable(reg_bwd):
            raise TypeError(
                f"Module {type(module).__name__} does not support "
                "register_full_backward_hook."
            )

        def _bwd_hook(mod: Any, grad_input: Any, grad_output: Any) -> None:
            if not should_sample():
                return
            epoch, step, phase = get_step_context()
            detached_grad_out = _maybe_detach(grad_output)
            event = InstrumentationEvent(
                name=name,
                stage="backward",
                value=detached_grad_out,
                inputs=_maybe_detach(grad_input),
                output=detached_grad_out,
                module=mod,
                epoch=epoch,
                step=step,
                phase=phase,
                call_index=next_call_index(name, "backward"),
            )
            emit_fn(event)

        raw_handles.append(reg_bwd(_bwd_hook))

    def _remove_all() -> None:
        for handle in raw_handles:
            remove_fn = getattr(handle, "remove", None)
            if callable(remove_fn):
                remove_fn()
        raw_handles.clear()

    return WatchHandle(_remove_all)
