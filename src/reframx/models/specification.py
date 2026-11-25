"""Specification for neural network model components."""

from __future__ import annotations

from dataclasses import dataclass, field

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec


def _default_model_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Model", component_type="model")


@dataclass(frozen=True, slots=True)
class ModelSpec(ComponentSpec):
    """Descriptive specification for a Model component."""

    metadata: ComponentMetadata = field(default_factory=_default_model_metadata)
    architecture: str | None = None
    public_apis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ComponentSpec.__post_init__(self)
        apis = tuple(self.public_apis)
        if not all(isinstance(api, str) and api.strip() for api in apis):
            raise ValueError("ModelSpec.public_apis must contain non-empty strings.")
        object.__setattr__(self, "public_apis", tuple(api.strip() for api in apis))
