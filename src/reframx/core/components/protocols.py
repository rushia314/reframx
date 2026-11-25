"""Structural Protocols for common and optional component behaviours."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from reframx.core.components.specification import ComponentSpec
from reframx.core.validation.context import ValidationContext
from reframx.core.validation.report import ValidationReport


@runtime_checkable
class SpecProvider(Protocol):
    """Optional protocol for objects that expose their own specification."""

    def get_spec(self) -> ComponentSpec:
        """Return the component's public specification."""
        ...

@runtime_checkable
class ConfigProvider(Protocol):
    """Optional protocol for components that expose a configuration schema."""

    @classmethod
    def get_config_schema(cls) -> Any:
        """Return a schema understood by the chosen configuration integration."""
        ...

@runtime_checkable
class ComponentValidator(Protocol):
    """Optional protocol for component-specific validation."""

    def validate_component(self, context: ValidationContext) -> ValidationReport:
        """Validate this component in its current integration context."""
        ...

@runtime_checkable
class StateProvider(Protocol):
    """Optional protocol for components with restorable custom runtime state."""

    def get_component_state(self) -> Mapping[str, Any]:
        """Return the component-owned state needed for restoration."""
        ...

    def load_component_state(self, state: Mapping[str, Any]) -> None:
        """Restore state previously returned by `get_component_state`."""
        ...
