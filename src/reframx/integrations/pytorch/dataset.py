"""Adapter for PyTorch DataLoaders and Datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import CapabilitiesInput
from reframx.core.contracts.requirements import RequirementsInput
from reframx.datasets.interface import BaseDataset
from reframx.integrations.pytorch.utils import require_torch


class TorchDatasetAdapter(BaseDataset):
    """Wraps PyTorch ``DataLoader`` instances or ``torch.utils.data.Dataset``s.

    Pass either pre-built ``loaders={"train": train_loader, "val": val_loader}``
    or raw ``datasets={"train": train_ds, ...}`` along with ``dataloader_kwargs``
    (e.g. ``batch_size``, ``shuffle``, ``num_workers``, ``collate_fn``).
    """

    def __init__(
        self,
        *,
        loaders: Mapping[str, Iterable[Any]] | None = None,
        datasets: Mapping[str, Any] | None = None,
        dataloader_kwargs: Mapping[str, Any] | None = None,
        split_dataloader_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if not loaders and not datasets:
            raise ValueError(
                "TorchDatasetAdapter requires at least one of `loaders` or `datasets`."
            )

        self._loaders: dict[str, Iterable[Any]] = dict(loaders or {})
        self._datasets: dict[str, Any] = dict(datasets or {})
        self._dataloader_kwargs: dict[str, Any] = dict(dataloader_kwargs or {})
        self._split_dataloader_kwargs: dict[str, dict[str, Any]] = {
            split: dict(cfg)
            for split, cfg in (split_dataloader_kwargs or {}).items()
        }

        all_splits = tuple(dict.fromkeys([*self._loaders.keys(), *self._datasets.keys()]))
        super().__init__(
            capabilities=capabilities,
            requirements=requirements,
            metadata=metadata,
            splits=all_splits,
            attributes={"framework": "pytorch", **dict(attributes or {})},
        )

    def has_split(self, split: str) -> bool:
        return split in self._loaders or split in self._datasets

    def get_split(self, split: str) -> Iterable[Any]:
        if split in self._loaders:
            return self._loaders[split]
        if split in self._datasets:
            torch = require_torch()
            kwargs = {
                **self._dataloader_kwargs,
                **self._split_dataloader_kwargs.get(split, {}),
            }
            if "shuffle" not in kwargs:
                kwargs["shuffle"] = split == "train"
            loader: Iterable[Any] = torch.utils.data.DataLoader(
                self._datasets[split], **kwargs
            )
            return loader
        raise KeyError(
            f"TorchDatasetAdapter '{self.spec.metadata.name}' does not provide "
            f"split '{split}'."
        )
