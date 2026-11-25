"""Event payloads and safe tensor detachment for model instrumentation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Stage = Literal["pre_forward", "post_forward", "backward", "probe"]


def detach_value(value: Any) -> Any:
    """Recursively detach tensor-like objects without mutating containers.

    Preserves nested ``dict``, ``list``, ``tuple``, and ``namedtuple``
    structures while calling ``.detach()`` on any object that provides it.
    """
    detach_fn = getattr(value, "detach", None)
    if callable(detach_fn):
        return detach_fn()
    if isinstance(value, Mapping):
        return {k: detach_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        if hasattr(value, "_fields"):  # namedtuple
            return type(value)(*(detach_value(v) for v in value))
        return tuple(detach_value(v) for v in value)
    if isinstance(value, list):
        return [detach_value(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class InstrumentationEvent:
    """Structured event passed to user instrumentation callbacks."""

    name: str
    stage: Stage
    value: Any = None
    inputs: Any = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    output: Any = None
    module: Any = None
    epoch: int = 0
    step: int = 0
    phase: str = "idle"
    call_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InstrumentationRecord:
    """One recorded measurement produced by an instrumentation callback."""

    name: str
    stage: Stage
    epoch: int
    step: int
    phase: str
    call_index: int
    metrics: Mapping[str, Any] = field(default_factory=dict)
