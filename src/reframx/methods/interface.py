"""Method result envelope, BaseMethod (full freedom), and SupervisedMethod."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    coerce_capabilities,
)
from reframx.core.contracts.requirements import (
    RequirementsInput,
    coerce_requirements,
)
from reframx.methods.specification import MethodSpec


def _to_scalar(value: Any) -> Any:
    """Convert scalar-like tensors (with ``.item()``) to Python numbers for metrics."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


@dataclass(slots=True)
class MethodResult:
    """Structured output returned by one Method step.

    ``loss`` is optional: methods with custom multi-loss or non-standard flows
    can store individual objectives in ``objectives`` and manage backward calls
    directly in ``train_step``.
    """

    task_outputs: Mapping[str, Any] = field(default_factory=dict)
    loss: Any | None = None
    objectives: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_outputs = dict(self.task_outputs)
        self.objectives = dict(self.objectives)
        self.metrics = dict(self.metrics)
        self.artifacts = dict(self.artifacts)

    def resolved_metrics(self) -> dict[str, Any]:
        """Return step metrics, automatically including scalar ``loss`` if absent."""
        merged = {k: _to_scalar(v) for k, v in self.metrics.items()}
        if self.loss is not None and "loss" not in merged:
            merged["loss"] = _to_scalar(self.loss)
        return merged


class BaseMethod(ABC):
    """Base class giving researchers complete control over their training algorithm.

    Override ``train_step(batch, ctx)`` to implement any custom procedure:
    multiple forwards, multiple ``ctx.backward(loss)`` calls, multiple
    optimizers, custom gradient manipulation, or custom state updates.
    """

    def __init__(
        self,
        *,
        requirements: RequirementsInput = (),
        capabilities: CapabilitiesInput = (),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        method_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="method",
        )
        self._spec = MethodSpec(
            metadata=method_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
        )

    def get_spec(self) -> MethodSpec:
        return self._spec

    @property
    def spec(self) -> MethodSpec:
        return self._spec

    def setup(self, ctx: Any) -> None:  # noqa: B027
        """Optional hook called once before execution begins."""

    @abstractmethod
    def train_step(self, batch: Any, ctx: Any) -> MethodResult:
        """Execute one training step and return a ``MethodResult``."""

    def eval_step(self, batch: Any, ctx: Any) -> MethodResult:
        """Optional evaluation step; override to support evaluation splits."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement eval_step(batch, ctx) "
            "to support evaluation."
        )


class SupervisedMethod(BaseMethod):
    """Convenience Method for standard ``forward -> loss -> backward -> step`` training.

    Subclasses only need to implement ``compute_step(batch, ctx) -> MethodResult``
    returning ``loss`` (and optional ``task_outputs`` / ``metrics``). The
    strategy-managed ``zero_grad``, ``backward``, and ``step`` calls are handled
    automatically during training, while ``eval_step`` reuses ``compute_step``.
    """

    optimizer_name: str | None = None

    @abstractmethod
    def compute_step(self, batch: Any, ctx: Any) -> MethodResult:
        """Compute forward pass, loss, and outputs for one batch."""

    def train_step(self, batch: Any, ctx: Any) -> MethodResult:
        result = self.compute_step(batch, ctx)
        if result.loss is None:
            raise ValueError(
                f"{type(self).__name__}.compute_step() must return a MethodResult "
                "with a non-None `loss` during training."
            )
        ctx.backward(result.loss)
        ctx.step(self.optimizer_name)
        return result

    def eval_step(self, batch: Any, ctx: Any) -> MethodResult:
        return self.compute_step(batch, ctx)
