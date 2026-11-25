"""Capabilities a component declares that it provides."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .io import IOContract, coerce_io_contract


@dataclass(frozen=True, slots=True)
class Capability:
    """A named ability exposed by a component.

    The framework does not infer that a capability is true merely from the
    component's architecture. It records the declaration and may run optional
    user-provided checks to test it.
    """

    name: str
    description: str | None = None
    contract: IOContract = field(default_factory=IOContract)
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Capability.name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "contract", coerce_io_contract(self.contract))
        object.__setattr__(self, "properties", dict(self.properties))


CapabilityLike: TypeAlias = Capability | str | Mapping[str, Any]


def coerce_capability(item: CapabilityLike) -> Capability:
    """Coerce a ``str``, ``Mapping``, or ``Capability`` into a ``Capability``."""
    if isinstance(item, Capability):
        return item
    if isinstance(item, str):
        return Capability(item)
    if isinstance(item, Mapping):
        return Capability(**item)
    raise TypeError(
        "ComponentSpec.capabilities must contain Capability objects, strings, "
        f"or mappings; got {type(item).__name__}."
    )


CapabilitiesInput: TypeAlias = CapabilityLike | Iterable[CapabilityLike]


def coerce_capabilities(items: CapabilitiesInput) -> tuple[Capability, ...]:
    """Normalize a single capability or iterable of capabilities into a tuple."""
    if isinstance(items, (Capability, str, Mapping)):
        return (coerce_capability(items),)
    return tuple(coerce_capability(item) for item in items)
