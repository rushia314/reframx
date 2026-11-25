"""Named Probes and built-in tensor/scalar measurement reducers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from reframx.instrumentation.events import InstrumentationEvent
from reframx.instrumentation.manager import get_active_manager

T = TypeVar("T")


def probe(
    name: str,
    value: T,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> T:
    """Emit a named probe to the active ``InstrumentationManager`` and return ``value``.

    Zero-overhead pass-through when no manager is active or no listener matches.
    """
    manager = get_active_manager()
    if manager is None:
        return value
    manager.emit_probe(name, value, metadata=metadata)
    return value


class Probe:
    """Reusable callable probe for exposing intermediate model tensors/states.

    Example::

        self.attn_probe = Probe("attn/scores")
        scores = self.attn_probe(scores)
    """

    def __init__(self, name: str) -> None:
        if not name or not name.strip():
            raise ValueError("Probe.name must be a non-empty string.")
        self.name = name.strip()

    def __call__(
        self,
        value: T,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> T:
        return probe(self.name, value, metadata=metadata)


def _flatten_numbers(value: Any) -> list[float]:
    """Fallback numeric extractor for Python scalars, lists, tuples, or `.item()`."""
    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return [float(item_fn())]
        except Exception:
            pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        out: list[float] = []
        for elem in value:
            out.extend(_flatten_numbers(elem))
        return out
    return []


def tensor_stats(event: InstrumentationEvent) -> dict[str, float]:
    """Compute basic summary statistics (mean, std, min, max, l2_norm).

    Uses native tensor ops (e.g. PyTorch) directly on device when available,
    falling back to pure Python for scalars and sequences.
    """
    val = event.value
    if (
        hasattr(val, "float")
        and hasattr(val, "mean")
        and hasattr(val, "min")
        and hasattr(val, "max")
        and hasattr(val, "norm")
    ):
        t = val.float()
        std_val = float(t.std().item()) if getattr(t, "numel", lambda: 2)() > 1 else 0.0
        return {
            "mean": float(t.mean().item()),
            "std": std_val,
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "l2_norm": float(t.norm().item()),
        }

    nums = _flatten_numbers(val)
    if not nums:
        return {}
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / n
    l2 = math.sqrt(sum(x * x for x in nums))
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(nums),
        "max": max(nums),
        "l2_norm": l2,
    }


def check_nan_inf(
    event: InstrumentationEvent,
    *,
    raise_on_error: bool = False,
) -> dict[str, bool]:
    """Detect NaN or Inf values in a probed tensor/scalar."""
    val = event.value
    has_nan = False
    has_inf = False

    isnan_fn = getattr(val, "isnan", None)
    isinf_fn = getattr(val, "isinf", None)
    if callable(isnan_fn) and callable(isinf_fn):
        has_nan = bool(isnan_fn().any().item())
        has_inf = bool(isinf_fn().any().item())
    else:
        for num in _flatten_numbers(val):
            if math.isnan(num):
                has_nan = True
            if math.isinf(num):
                has_inf = True

    if raise_on_error and (has_nan or has_inf):
        raise FloatingPointError(
            f"Non-finite value detected at '{event.name}' "
            f"(stage={event.stage}, epoch={event.epoch}, step={event.step}): "
            f"has_nan={has_nan}, has_inf={has_inf}."
        )

    return {"has_nan": has_nan, "has_inf": has_inf}
