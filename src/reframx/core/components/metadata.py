"""Identity and descriptive metadata for framework components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen = True, slots = True)
class ComponentMetadata:
    """Human-readable identity for a component.

    `component_type` is intentionally a string rather than a closed enum so that
    users and plugins can introduce new component categories without modifying
    the framework's core package.
    """

    name: str
    component_type: str = "component"
    version: str | None = None
    api_version: str | None = None
    description: str | None = None
    author: str | None = None
    tags: tuple[str, ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ComponentMetadata.name must be a non-empty string.")
        if not self.component_type or not self.component_type.strip():
            raise ValueError("ComponentMetadata.component_type must be non-empty.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "component_type", self.component_type.strip())
        object.__setattr__(self, "tags", tuple(self.tags))