"""Strategy interface and default backward/optimization strategy."""

from __future__ import annotations

import contextlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    Capability,
    coerce_capabilities,
)
from reframx.core.contracts.requirements import (
    RequirementsInput,
    coerce_requirements,
)


def _default_strategy_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Strategy", component_type="strategy")


@dataclass(frozen=True, slots=True)
class StrategySpec(ComponentSpec):
    """Descriptive specification for an execution/optimization Strategy."""

    metadata: ComponentMetadata = field(default_factory=_default_strategy_metadata)


class BaseStrategy(ABC):
    """Base class controlling backward, optimizer updates, and eval contexts."""

    def __init__(
        self,
        *,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        strategy_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="strategy",
        )
        self._spec = StrategySpec(
            metadata=strategy_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
        )

    def get_spec(self) -> StrategySpec:
        return self._spec

    @property
    def spec(self) -> StrategySpec:
        return self._spec

    def prepare_model(self, model: Any) -> Any:
        """Optionally wrap or move a model before execution."""
        return model

    def prepare_batch(self, batch: Any, ctx: Any) -> Any:
        """Optionally transform or move a batch (e.g. to a device) before a step."""
        return batch

    def set_model_mode(self, model: Any, *, training: bool) -> None:
        """Toggle train/eval mode when supported by the model."""
        target = getattr(model, "module", model)
        train_fn = getattr(target, "train", None)
        if callable(train_fn):
            train_fn(training)

    @contextlib.contextmanager
    def eval_context(self) -> Generator[None, None, None]:
        """Context manager entered during evaluation steps (e.g. ``no_grad``)."""
        torch_mod = sys.modules.get("torch")
        no_grad = getattr(torch_mod, "no_grad", None) if torch_mod is not None else None
        if callable(no_grad):
            with no_grad():
                yield
        else:
            yield

    @abstractmethod
    def backward(self, loss: Any, ctx: Any) -> None:
        """Run backward pass on ``loss``."""

    @abstractmethod
    def optimizer_step(self, optimizer: Any, ctx: Any, *, zero_grad: bool = True) -> bool:
        """Step ``optimizer`` and return whether an update occurred."""

    @abstractmethod
    def zero_grad(self, optimizer: Any, ctx: Any) -> None:
        """Zero gradients on ``optimizer``."""


class StandardBackwardStrategy(BaseStrategy):
    """Backend-agnostic strategy using duck-typed ``loss.backward()`` and optimizers.

    Works with PyTorch tensors/optimizers as well as custom Python/NumPy
    autograd implementations without requiring a hard ``torch`` dependency.
    """

    def __init__(
        self,
        *,
        grad_accumulation_steps: int = 1,
        on_before_step: Callable[[Any, Any], None] | None = None,
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if grad_accumulation_steps < 1:
            raise ValueError("grad_accumulation_steps must be >= 1.")
        self.grad_accumulation_steps = grad_accumulation_steps
        self.on_before_step = on_before_step
        self._accum_counts: dict[int, int] = {}
        super().__init__(
            capabilities=(
                Capability("backward"),
                Capability("optimizer_step"),
            ),
            metadata=metadata,
            attributes={
                "grad_accumulation_steps": grad_accumulation_steps,
                **dict(attributes or {}),
            },
        )

    def backward(self, loss: Any, ctx: Any) -> None:
        scaled_loss = (
            loss / self.grad_accumulation_steps
            if self.grad_accumulation_steps > 1
            else loss
        )
        backward_fn = getattr(scaled_loss, "backward", None)
        if not callable(backward_fn):
            raise TypeError(
                f"Loss object of type {type(loss).__name__} does not define a "
                "callable .backward() method."
            )
        backward_fn()

    def optimizer_step(
        self,
        optimizer: Any,
        ctx: Any,
        *,
        zero_grad: bool = True,
    ) -> bool:
        opt_id = id(optimizer)
        count = self._accum_counts.get(opt_id, 0) + 1
        self._accum_counts[opt_id] = count

        if count % self.grad_accumulation_steps != 0:
            return False

        if self.on_before_step is not None:
            self.on_before_step(optimizer, ctx)

        step_fn = getattr(optimizer, "step", None)
        if not callable(step_fn):
            raise TypeError(
                f"Optimizer of type {type(optimizer).__name__} does not define "
                "a callable .step() method."
            )
        step_fn()

        if zero_grad:
            self.zero_grad(optimizer, ctx)
        return True

    def zero_grad(self, optimizer: Any, ctx: Any) -> None:
        zero_fn = getattr(optimizer, "zero_grad", None)
        if not callable(zero_fn):
            raise TypeError(
                f"Optimizer of type {type(optimizer).__name__} does not define "
                "a callable .zero_grad() method."
            )
        zero_fn()
