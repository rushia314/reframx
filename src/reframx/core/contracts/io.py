from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

ShapeDim: TypeAlias = int | str | None


@dataclass(frozen=True, slots=True)
class IOContract:
    """Describe the structure and semantics of an input or output."""

    kind: str | None = None
    value_type: str | None = None
    semantics: str | None = None

    dtype: str | None = None
    shape: tuple[ShapeDim, ...] | None = None

    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", dict(self.attributes))
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))


IOContractLike: TypeAlias = IOContract | Mapping[str, Any] | None


def coerce_io_contract(value: IOContractLike) -> IOContract:
    """Coerce ``None``, ``Mapping``, or ``IOContract`` into an ``IOContract``."""
    if value is None:
        return IOContract()
    if isinstance(value, IOContract):
        return value
    if isinstance(value, Mapping):
        return IOContract(**value)
    raise TypeError(
        f"contract must be IOContract, Mapping, or None; got {type(value).__name__}."
    )