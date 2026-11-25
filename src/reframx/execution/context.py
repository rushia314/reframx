"""Runtime execution state and context passed to Methods and Hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Phase = Literal["idle", "train", "eval"]


@dataclass(slots=True)
class ExecutionState:
    """Mutable runtime counters and control flags for one execution."""

    epoch: int = 0
    global_step: int = 0
    step_in_epoch: int = 0
    phase: Phase = "idle"
    should_stop: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionContext:
    """Runtime services exposed to Methods and Hooks during execution."""

    model: Any
    strategy: Any
    task: Any = None
    method: Any = None
    dataset: Any = None
    objectives: dict[str, Any] = field(default_factory=dict)
    optimizers: dict[str, Any] = field(default_factory=dict)
    evaluator: Any = None
    run: Any = None
    hooks: list[Any] = field(default_factory=list)
    state: ExecutionState = field(default_factory=ExecutionState)
    step_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def objective(self) -> Any:
        """Return the default objective when a single objective is registered."""
        if "default" in self.objectives:
            return self.objectives["default"]
        if len(self.objectives) == 1:
            return next(iter(self.objectives.values()))
        raise KeyError(
            "Ambiguous or missing default objective; access `ctx.objectives[name]`."
        )

    def get_optimizer(self, name: str | None = None) -> Any:
        """Look up an optimizer by name, or return the sole registered optimizer."""
        if name is not None:
            if name not in self.optimizers:
                raise KeyError(f"Optimizer '{name}' is not registered.")
            return self.optimizers[name]
        if "default" in self.optimizers:
            return self.optimizers["default"]
        if len(self.optimizers) == 1:
            return next(iter(self.optimizers.values()))
        raise KeyError(
            "Ambiguous or missing default optimizer; pass an explicit optimizer name."
        )

    def backward(self, loss: Any) -> None:
        """Run backward pass on ``loss`` through the active Strategy."""
        self.strategy.backward(loss, self)

    def step(
        self,
        optimizer_name: str | None = None,
        *,
        zero_grad: bool = True,
    ) -> bool:
        """Step the selected optimizer through the active Strategy."""
        optimizer = self.get_optimizer(optimizer_name)
        return bool(
            self.strategy.optimizer_step(optimizer, self, zero_grad=zero_grad)
        )

    def zero_grad(self, optimizer_name: str | None = None) -> None:
        """Zero gradients on the selected optimizer through the active Strategy."""
        optimizer = self.get_optimizer(optimizer_name)
        self.strategy.zero_grad(optimizer, self)

    def log_metric(self, name: str, value: Any) -> None:
        """Attach an extra metric to the current step record."""
        self.step_metrics[name] = value

    def request_stop(self) -> None:
        """Signal the execution engine to stop gracefully after the current step."""
        self.state.should_stop = True
