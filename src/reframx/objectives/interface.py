"""Specification and convenience base class for Objective (loss) components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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


def _default_objective_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Objective", component_type="objective")


@dataclass(frozen=True, slots=True)
class ObjectiveSpec(ComponentSpec):
    """Descriptive specification for an Objective / loss component."""

    metadata: ComponentMetadata = field(default_factory=_default_objective_metadata)


class BaseObjective(ABC):
    """Optional convenience base class for custom loss/objective functions.

    Subclasses implement ``compute(...)`` and can be invoked directly via
    ``objective(...)`` or ``objective.compute(...)``.
    """

    def __init__(
        self,
        *,
        capabilities: CapabilitiesInput | None = None,
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        objective_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="objective",
        )
        resolved_caps = (
            coerce_capabilities("loss")
            if capabilities is None
            else coerce_capabilities(capabilities)
        )
        self._spec = ObjectiveSpec(
            metadata=objective_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=resolved_caps,
            attributes=attributes or {},
        )

    def get_spec(self) -> ObjectiveSpec:
        return self._spec

    @property
    def spec(self) -> ObjectiveSpec:
        return self._spec

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """Compute and return the objective/loss value."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute(*args, **kwargs)
