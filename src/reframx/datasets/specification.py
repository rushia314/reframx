"""Specification for Dataset components."""

from __future__ import annotations

from dataclasses import dataclass, field

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec


def _default_dataset_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Dataset", component_type="dataset")


@dataclass(frozen=True, slots=True)
class DatasetSpec(ComponentSpec):
    """Descriptive specification for a Dataset component."""

    metadata: ComponentMetadata = field(default_factory=_default_dataset_metadata)
    splits: tuple[str, ...] = ("train",)

    def __post_init__(self) -> None:
        ComponentSpec.__post_init__(self)
        normalized_splits = tuple(self.splits)
        if not all(isinstance(s, str) and s.strip() for s in normalized_splits):
            raise ValueError("DatasetSpec.splits must contain non-empty strings.")
        object.__setattr__(
            self,
            "splits",
            tuple(s.strip() for s in normalized_splits),
        )
