"""Specification, protocol, and convenience base class for Evaluator components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    coerce_capabilities,
)
from reframx.core.contracts.requirements import (
    RequirementsInput,
    coerce_requirements,
)


def _default_evaluator_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Evaluator", component_type="evaluator")


@dataclass(frozen=True, slots=True)
class EvaluatorSpec(ComponentSpec):
    """Descriptive specification for an Evaluator component."""

    metadata: ComponentMetadata = field(default_factory=_default_evaluator_metadata)


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Minimal protocol for stateful evaluation across batches."""

    def reset(self) -> None:
        """Reset accumulated state before an evaluation pass."""
        ...

    def update(self, result: Any, batch: Any) -> None:
        """Ingest one evaluation step result and its corresponding batch."""
        ...

    def compute(self) -> Mapping[str, Any]:
        """Compute and return aggregated evaluation metrics."""
        ...


class BaseEvaluator(ABC):
    """Optional convenience base class for user-defined evaluators."""

    def __init__(
        self,
        *,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        evaluator_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="evaluator",
        )
        self._spec = EvaluatorSpec(
            metadata=evaluator_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
        )

    def get_spec(self) -> EvaluatorSpec:
        return self._spec

    @property
    def spec(self) -> EvaluatorSpec:
        return self._spec

    def reset(self) -> None:  # noqa: B027
        """Optional hook to reset internal metric accumulators."""

    @abstractmethod
    def update(self, result: Any, batch: Any) -> None:
        """Ingest one evaluation step result and its batch."""

    @abstractmethod
    def compute(self) -> Mapping[str, Any]:
        """Compute and return aggregated metrics."""
