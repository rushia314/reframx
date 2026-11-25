"""Context passed to optional component-specific validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from reframx.core.components.specification import ComponentSpec


@dataclass(frozen = True, slots = True)
class ValidationContext:
    """Read-only view of the objects relevant to a validation call."""

    component: Any
    component_spec: ComponentSpec
    role: str
    peers: Mapping[str, Any] = field(default_factory = dict)
    experiment: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "peers", dict(self.peers))