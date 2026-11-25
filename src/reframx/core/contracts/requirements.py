"""Requirements declared by a component."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from .io import IOContract, coerce_io_contract


@dataclass(frozen=True, slots=True)
class Requirement:
    name: str
    provider_type: str | None = None
    optional: bool = False
    description: str | None = None
    contract: IOContract = field(default_factory=IOContract)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Requirement.name must be a non-empty string.")
        if self.provider_type is not None:
            if not isinstance(self.provider_type, str) or not self.provider_type.strip():
                raise ValueError(
                    "Requirement.provider_type must be None or a non-empty string."
                )
            object.__setattr__(self, "provider_type", self.provider_type.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "contract", coerce_io_contract(self.contract))
        object.__setattr__(self, "constraints", dict(self.constraints))


def coerce_single_requirement(item: Requirement | str | Mapping[str, Any]) -> Requirement:
    """Coerce a ``str``, ``Mapping``, or ``Requirement`` into a ``Requirement``."""
    if isinstance(item, Requirement):
        return item
    if isinstance(item, str):
        return Requirement(item)
    if isinstance(item, Mapping):
        return Requirement(**item)
    raise TypeError(
        "All members must be Requirement instances, strings, or mappings; "
        f"got {type(item).__name__}."
    )


@dataclass(frozen=True, slots=True)
class RequirementGroup:
    """A logical group of alternative or combined requirements."""

    name: str | None = None
    requirements: tuple[Requirement, ...] = ()
    mode: Literal["all_of", "any_of"] = "all_of"
    optional: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        raw_requirements = tuple(self.requirements)
        if not raw_requirements:
            raise ValueError("RequirementGroup cannot be empty.")

        if self.mode not in ("all_of", "any_of"):
            raise ValueError("mode must be 'all_of' or 'any_of'.")

        requirements = tuple(coerce_single_requirement(r) for r in raw_requirements)

        if self.mode == "any_of" and any(r.optional for r in requirements):
            raise ValueError(
                "Individual requirements in any_of groups "
                "cannot be optional. Use group.optional instead."
            )

        if self.name is None:
            member_names = ", ".join(r.name for r in requirements)
            resolved_name = f"{self.mode}({member_names})"
        else:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("RequirementGroup.name must be non-empty.")
            resolved_name = self.name.strip()

        object.__setattr__(self, "name", resolved_name)
        object.__setattr__(self, "requirements", requirements)


RequirementLike: TypeAlias = Requirement | RequirementGroup | str | Mapping[str, Any]


def coerce_requirement(item: RequirementLike) -> Requirement | RequirementGroup:
    """Coerce a ``str``, ``Mapping``, ``Requirement``, or ``RequirementGroup``."""
    if isinstance(item, (Requirement, RequirementGroup)):
        return item
    if isinstance(item, str):
        return Requirement(item)
    if isinstance(item, Mapping):
        if "requirements" in item:
            return RequirementGroup(**item)
        return Requirement(**item)
    raise TypeError(
        "ComponentSpec.requirements must contain Requirement, RequirementGroup, "
        f"str, or Mapping objects; got {type(item).__name__}."
    )


RequirementsInput: TypeAlias = RequirementLike | Iterable[RequirementLike]


def coerce_requirements(
    items: RequirementsInput,
) -> tuple[Requirement | RequirementGroup, ...]:
    """Normalize a single requirement/group or iterable into a tuple."""
    if isinstance(items, (Requirement, RequirementGroup, str, Mapping)):
        return (coerce_requirement(items),)
    return tuple(coerce_requirement(item) for item in items)