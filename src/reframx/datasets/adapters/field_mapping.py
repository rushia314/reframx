"""Non-invasive Dataset adapter for field renaming and batch transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from reframx.core.components.handle import ComponentHandle
from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    Capability,
    coerce_capabilities,
)
from reframx.datasets.specification import DatasetSpec


class _MappedSplitIterable(Iterable[Any]):
    """Re-iterable view over a dataset split applying field mapping & transform."""

    def __init__(
        self,
        source_iterable: Iterable[Any],
        *,
        field_map: dict[str, str],
        keep_unmapped: bool,
        transform: Callable[[Any], Any] | None,
    ) -> None:
        self._source = source_iterable
        self._field_map = field_map
        self._keep_unmapped = keep_unmapped
        self._transform = transform

    def __iter__(self) -> Iterator[Any]:
        for batch in self._source:
            mapped = self._map_batch(batch)
            if self._transform is not None:
                mapped = self._transform(mapped)
            yield mapped

    def _map_batch(self, batch: Any) -> Any:
        if not self._field_map or not isinstance(batch, Mapping):
            return batch
        result: dict[str, Any] = {}
        for key, value in batch.items():
            if key in self._field_map:
                result[self._field_map[key]] = value
            elif self._keep_unmapped:
                result[key] = value
        return result


class DatasetFieldAdapter:
    """Translates a Dataset's native schema to match a Method's requirements.

    Does not mutate the wrapped dataset. Automatically rewrites matching
    ``Capability`` names in ``DatasetSpec`` while preserving their
    ``IOContract``, ``description``, and ``properties``.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        field_map: Mapping[str, str] | None = None,
        keep_unmapped: bool = True,
        transform: Callable[[Any], Any] | None = None,
        capabilities: CapabilitiesInput | None = None,
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if dataset is None:
            raise ValueError("DatasetFieldAdapter.dataset cannot be None.")

        self._dataset = dataset
        self._field_map: dict[str, str] = {}
        for src, dst in (field_map or {}).items():
            if not isinstance(src, str) or not src.strip():
                raise ValueError("field_map source keys must be non-empty strings.")
            if not isinstance(dst, str) or not dst.strip():
                raise ValueError("field_map target keys must be non-empty strings.")
            self._field_map[src.strip()] = dst.strip()

        self._keep_unmapped = keep_unmapped
        self._transform = transform

        source_handle = ComponentHandle.wrap(dataset, role="dataset")
        source_spec = source_handle.spec
        source_splits = getattr(source_spec, "splits", ("train",))

        resolved_caps = (
            coerce_capabilities(capabilities)
            if capabilities is not None
            else self._derive_capabilities(source_spec.capabilities)
        )
        adapter_metadata = metadata or ComponentMetadata(
            name=f"{source_spec.metadata.name}Adapter",
            component_type="dataset",
            version=source_spec.metadata.version,
            description=source_spec.metadata.description,
        )
        self._spec = DatasetSpec(
            metadata=adapter_metadata,
            requirements=source_spec.requirements,
            capabilities=resolved_caps,
            splits=tuple(source_splits),
            attributes={
                **dict(source_spec.attributes),
                "adapted_from": source_spec.metadata.name,
                "field_map": dict(self._field_map),
                **dict(attributes or {}),
            },
        )

    def _derive_capabilities(
        self,
        source_capabilities: tuple[Capability, ...],
    ) -> tuple[Capability, ...]:
        derived: list[Capability] = []
        seen_names: set[str] = set()

        for cap in source_capabilities:
            if cap.name in self._field_map:
                new_name = self._field_map[cap.name]
                if new_name not in seen_names:
                    derived.append(
                        Capability(
                            name=new_name,
                            description=cap.description,
                            contract=cap.contract,
                            properties=cap.properties,
                        )
                    )
                    seen_names.add(new_name)
            elif self._keep_unmapped and cap.name not in seen_names:
                derived.append(cap)
                seen_names.add(cap.name)

        for dst in self._field_map.values():
            if dst not in seen_names:
                derived.append(Capability(name=dst))
                seen_names.add(dst)

        return tuple(derived)

    @property
    def dataset(self) -> Any:
        return self._dataset

    def get_spec(self) -> DatasetSpec:
        return self._spec

    @property
    def spec(self) -> DatasetSpec:
        return self._spec

    def has_split(self, split: str) -> bool:
        has_split_fn = getattr(self._dataset, "has_split", None)
        if callable(has_split_fn):
            return bool(has_split_fn(split))
        return split in self._spec.splits

    def get_split(self, split: str) -> Iterable[Any]:
        get_split_fn = getattr(self._dataset, "get_split", None)
        if callable(get_split_fn):
            source_iterable = get_split_fn(split)
        elif split == "train" and isinstance(self._dataset, Iterable):
            source_iterable = self._dataset
        else:
            raise KeyError(
                f"Adapted dataset '{self._spec.metadata.name}' does not provide "
                f"split '{split}'."
            )
        return _MappedSplitIterable(
            source_iterable,
            field_map=self._field_map,
            keep_unmapped=self._keep_unmapped,
            transform=self._transform,
        )
