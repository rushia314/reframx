"""Protocol and convenience base class for Dataset components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    coerce_capabilities,
)
from reframx.core.contracts.requirements import (
    RequirementsInput,
    coerce_requirements,
)
from reframx.datasets.specification import DatasetSpec


@runtime_checkable
class DatasetProtocol(Protocol):
    """Minimal runtime protocol for datasets supplying split iterables."""

    def get_split(self, split: str) -> Iterable[Any]:
        """Return an iterable of batches for the requested split."""
        ...


class BaseDataset:
    """Optional convenience base class for user-defined datasets.

    Users can either pass ``splits_data`` directly (mapping split names like
    ``"train"`` / ``"val"`` to iterables of batches) or override ``get_split``.
    """

    def __init__(
        self,
        *,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        splits: Iterable[str] | None = None,
        splits_data: Mapping[str, Iterable[Any]] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        self._splits_data: dict[str, Iterable[Any]] = dict(splits_data or {})
        resolved_splits = (
            tuple(splits)
            if splits is not None
            else (tuple(self._splits_data.keys()) or ("train",))
        )
        dataset_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="dataset",
        )
        self._spec = DatasetSpec(
            metadata=dataset_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
            splits=resolved_splits,
        )

    def get_spec(self) -> DatasetSpec:
        return self._spec

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def has_split(self, split: str) -> bool:
        return split in self._splits_data or split in self._spec.splits

    def get_split(self, split: str) -> Iterable[Any]:
        if split in self._splits_data:
            return self._splits_data[split]
        raise KeyError(
            f"Dataset '{self._spec.metadata.name}' does not provide split '{split}'."
        )
