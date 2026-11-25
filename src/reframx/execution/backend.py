"""Pluggable execution backend protocol and default native backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from reframx.execution.context import ExecutionContext
from reframx.execution.engine import ExecutionEngine

if TYPE_CHECKING:
    from reframx.experiments.run import Run


@runtime_checkable
class ExecutionBackend(Protocol):
    """Minimal contract for pluggable execution backends (native, MMEngine, ...)."""

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
        """Execute a full training (and optional evaluation) run."""
        ...

    def evaluate(
        self,
        ctx: ExecutionContext,
        *,
        split: str = "val",
    ) -> dict[str, Any]:
        """Run a standalone evaluation pass over ``split``."""
        ...


class NativeBackend:
    """Default zero-dependency execution backend backed by ``ExecutionEngine``."""

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
        engine = ExecutionEngine(ctx)
        return engine.run(
            max_epochs=max_epochs,
            max_steps=max_steps,
            train_split=train_split,
            eval_split=eval_split,
            eval_every_epochs=eval_every_epochs,
            eval_every_steps=eval_every_steps,
            start_step=start_step,
            start_epoch=start_epoch,
            keep_eval_mode=keep_eval_mode,
            max_train_history=max_train_history,
            catch_interrupt=catch_interrupt,
        )

    def evaluate(
        self,
        ctx: ExecutionContext,
        *,
        split: str = "val",
    ) -> dict[str, Any]:
        engine = ExecutionEngine(ctx)
        return engine.evaluate(split=split)
