"""Lazy MMEngine import helpers.

MMEngine is an optional execution backend and is never declared as a hard
dependency of the core package.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def is_mmengine_available() -> bool:
    """Return True if ``mmengine`` can be imported in the active environment."""
    return importlib.util.find_spec("mmengine") is not None


def require_mmengine() -> Any:
    """Import and return ``mmengine``, raising a helpful error if missing."""
    try:
        return importlib.import_module("mmengine")
    except ImportError as exc:
        raise ImportError(
            "MMEngine backend requires 'mmengine', which is not installed in "
            "the active environment. Install it matching your PyTorch setup "
            "(e.g. `pip install mmengine`)."
        ) from exc


def require_mmengine_submodule(submodule: str) -> Any:
    """Import and return ``mmengine.<submodule>`` safely."""
    mmengine = require_mmengine()
    existing = getattr(mmengine, submodule, None)
    if existing is not None:
        return existing
    try:
        return importlib.import_module(f"mmengine.{submodule}")
    except ImportError as exc:
        raise ImportError(
            f"Could not import 'mmengine.{submodule}': {exc}"
        ) from exc
