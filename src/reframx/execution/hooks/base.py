"""Base hook interface for user plugins and lifecycle observers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class BaseHook:
    """Optional convenience base class for lifecycle plugins/hooks.

    Plain Python objects that implement any subset of these methods are also
    accepted without inheriting from ``BaseHook``.
    """

    def on_run_start(self, ctx: Any) -> None:
        """Called once when a run begins."""

    def on_epoch_start(self, ctx: Any) -> None:
        """Called at the beginning of each training epoch."""

    def on_step_start(self, ctx: Any, batch: Any) -> None:
        """Called before ``method.train_step``."""

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        """Called after ``method.train_step``."""

    def on_eval_start(self, ctx: Any) -> None:
        """Called before an evaluation pass begins."""

    def on_eval_end(self, ctx: Any, metrics: Mapping[str, Any]) -> None:
        """Called after an evaluation pass completes."""

    def on_epoch_end(self, ctx: Any, epoch_metrics: Mapping[str, Any]) -> None:
        """Called at the end of each epoch."""

    def on_run_end(self, ctx: Any, run: Any) -> None:
        """Called once when a run finishes (completed or failed)."""


def dispatch_hooks(hooks: list[Any], event_name: str, *args: Any) -> None:
    """Invoke ``event_name`` on every hook that defines it."""
    for hook in hooks:
        callback = getattr(hook, event_name, None)
        if callable(callback):
            callback(*args)
