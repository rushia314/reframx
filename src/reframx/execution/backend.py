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
        max_epochs: int = 1,
        max_steps: int | None = None,
        train_split: str = "train",
        eval_split: str | None = "val",
        eval_every_epochs: int = 1,
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
        max_epochs: int = 1,
        max_steps: int | None = None,
        train_split: str = "train",
        eval_split: str | None = "val",
        eval_every_epochs: int = 1,
        catch_interrupt: bool = False,
    ) -> Run:
        engine = ExecutionEngine(ctx)
        return engine.run(
            max_epochs=max_epochs,
            max_steps=max_steps,
            train_split=train_split,
            eval_split=eval_split,
            eval_every_epochs=eval_every_epochs,
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
