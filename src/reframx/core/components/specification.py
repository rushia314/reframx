"""Serializable description of a component's public contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    Capability,
    coerce_capability,
)
from reframx.core.contracts.requirements import (
    Requirement,
    RequirementGroup,
    coerce_requirement,
)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Common specification shared by all component types.

    Specialised specifications (TaskSpec, ModelSpec, ...) should compose or
    extend this structure when their interfaces are designed. `attributes`
    holds additional descriptive data; it is not a place for runtime state.
    """
    metadata: ComponentMetadata
    requirements: tuple[Requirement | RequirementGroup, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ComponentMetadata):
            raise TypeError("ComponentSpec.metadata must be ComponentMetadata.")
        raw_reqs = (
            (self.requirements,)
            if isinstance(self.requirements, (str, Mapping, Requirement, RequirementGroup))
            else self.requirements
        )
        raw_caps = (
            (self.capabilities,)
            if isinstance(self.capabilities, (str, Mapping, Capability))
            else self.capabilities
        )
        normalized_reqs = tuple(coerce_requirement(r) for r in raw_reqs)
        normalized_caps = tuple(coerce_capability(c) for c in raw_caps)
        object.__setattr__(self, "requirements", normalized_reqs)
        object.__setattr__(self, "capabilities", normalized_caps)
        object.__setattr__(self, "attributes", dict(self.attributes))