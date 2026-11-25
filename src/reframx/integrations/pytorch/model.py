"""Non-invasive PyTorch ``nn.Module`` adapter with StateProvider support."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import (
    CapabilitiesInput,
    Capability,
    coerce_capabilities,
)
from reframx.core.contracts.io import IOContract
from reframx.core.contracts.requirements import RequirementsInput
from reframx.integrations.pytorch.utils import require_torch
from reframx.models.interface import ModelAdapter
from reframx.models.specification import ModelSpec


def _count_parameters(module: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        numel = int(param.numel())
        total += numel
        if getattr(param, "requires_grad", False):
            trainable += numel
    return total, trainable


class TorchModelAdapter(ModelAdapter):
    """Wraps any ``torch.nn.Module`` without mutating or subclassing it.

    - Preserves the user's original module untouched (``adapter.module``).
    - Records parameter counts and ``framework="pytorch"`` in ``ModelSpec``.
    - Defaults ``capabilities`` to ``(Capability("forward"),)`` if omitted.
    - Implements ``StateProvider`` via ``state_dict`` / ``load_state_dict``.
    """

    def __init__(
        self,
        module: Any,
        *,
        spec: ModelSpec | None = None,
        capabilities: CapabilitiesInput | None = None,
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        architecture: str | None = None,
        public_apis: Iterable[str] = ("forward",),
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        torch = require_torch()
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                "TorchModelAdapter expects a torch.nn.Module; "
                f"got {type(module).__name__}."
            )

        total_params, trainable_params = _count_parameters(module)
        merged_attrs = {
            "framework": "pytorch",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            **dict(attributes or {}),
        }
        resolved_caps = (
            coerce_capabilities(capabilities)
            if capabilities is not None
            else (
                Capability(
                    "forward",
                    contract=IOContract(kind="api", value_type="callable"),
                ),
            )
        )

        super().__init__(
            module=module,
            spec=spec,
            capabilities=resolved_caps,
            requirements=requirements,
            metadata=metadata,
            architecture=architecture or type(module).__name__,
            public_apis=public_apis,
            attributes=merged_attrs,
        )

    def get_component_state(self) -> Mapping[str, Any]:
        """Return the module's ``state_dict`` for checkpointing."""
        return dict(self._module.state_dict())

    def load_component_state(self, state: Mapping[str, Any]) -> None:
        """Restore the module's parameters/buffers from ``state``."""
        self._module.load_state_dict(state)
