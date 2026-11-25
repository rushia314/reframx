"""Convenience base class and non-invasive adapter for Model components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    coerce_capabilities,
)
from reframx.core.contracts.requirements import (
    RequirementsInput,
    coerce_requirements,
)
from reframx.models.specification import ModelSpec


class BaseModel:
    """Optional convenience base class for user-defined models.

    Researchers do not have to inherit from this class; plain Python objects
    or ``torch.nn.Module`` instances can be wrapped with ``ModelAdapter`` or
    ``ComponentHandle.wrap()``.
    """

    def __init__(
        self,
        *,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        architecture: str | None = None,
        public_apis: Iterable[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        model_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="model",
        )
        self._spec = ModelSpec(
            metadata=model_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
            architecture=architecture or type(self).__name__,
            public_apis=tuple(public_apis),
        )

    def get_spec(self) -> ModelSpec:
        return self._spec

    @property
    def spec(self) -> ModelSpec:
        return self._spec


class ModelAdapter:
    """Non-invasive wrapper that attaches a ``ModelSpec`` to an arbitrary model.

    Useful for integrating a raw ``torch.nn.Module`` or third-party model
    without modifying its class definition.
    """

    def __init__(
        self,
        module: Any,
        *,
        spec: ModelSpec | None = None,
        capabilities: CapabilitiesInput = (),
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        architecture: str | None = None,
        public_apis: Iterable[str] = (),
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if module is None:
            raise ValueError("ModelAdapter.module cannot be None.")
        self._module = module
        if spec is not None:
            if not isinstance(spec, ModelSpec):
                raise TypeError(f"spec must be ModelSpec; got {type(spec).__name__}.")
            self._spec = spec
        else:
            model_metadata = metadata or ComponentMetadata(
                name=type(module).__name__,
                component_type="model",
            )
            self._spec = ModelSpec(
                metadata=model_metadata,
                requirements=coerce_requirements(requirements),
                capabilities=coerce_capabilities(capabilities),
                attributes=attributes or {},
                architecture=architecture or type(module).__name__,
                public_apis=tuple(public_apis),
            )

    @property
    def module(self) -> Any:
        return self._module

    def get_spec(self) -> ModelSpec:
        return self._spec

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)
