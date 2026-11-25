"""Lazy PyTorch import and device/tensor utility helpers.

PyTorch is intentionally never declared as a hard dependency of the core
package so that installing ``reframx`` never overwrites a user's
platform-specific (CUDA / ROCm / CPU) PyTorch build.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any


def is_torch_available() -> bool:
    """Return True if ``torch`` can be imported in the current environment."""
    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    """Import and return ``torch``, raising a helpful error if missing."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError(
            "PyTorch integration requires 'torch', which is not installed in "
            "the active environment. Install the build matching your hardware "
            "from https://pytorch.org/get-started/locally/."
        ) from exc


def resolve_device(device: str | Any | None = "auto") -> Any:
    """Resolve ``"auto"`` or an explicit device string into a ``torch.device``."""
    torch = require_torch()
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def move_to_device(data: Any, device: Any, *, non_blocking: bool = False) -> Any:
    """Recursively move tensors inside nested containers onto ``device``."""
    torch = require_torch()
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    if isinstance(data, Mapping):
        return {
            key: move_to_device(value, device, non_blocking=non_blocking)
            for key, value in data.items()
        }
    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(
                *(move_to_device(v, device, non_blocking=non_blocking) for v in data)
            )
        return tuple(
            move_to_device(value, device, non_blocking=non_blocking) for value in data
        )
    if isinstance(data, list):
        return [
            move_to_device(value, device, non_blocking=non_blocking) for value in data
        ]
    if is_dataclass(data) and not isinstance(data, type):
        kwargs = {
            f.name: move_to_device(
                getattr(data, f.name), device, non_blocking=non_blocking
            )
            for f in fields(data)
            if f.init
        }
        return type(data)(**kwargs)
    return data
