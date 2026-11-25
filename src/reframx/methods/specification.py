"""Specification for Method components."""

from __future__ import annotations

from dataclasses import dataclass, field

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec


def _default_method_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Method", component_type="method")


@dataclass(frozen=True, slots=True)
class MethodSpec(ComponentSpec):
    """Descriptive specification for a research Method.

    A Method declares its data/model/objective/service needs in ``requirements``
    and the outputs it produces in ``capabilities``. Compatibility with a Task
    is determined by matching Method capabilities against Task accepted outputs.
    """

    metadata: ComponentMetadata = field(default_factory=_default_method_metadata)
