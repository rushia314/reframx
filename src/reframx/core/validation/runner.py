"""Invocation helpers for optional user-defined component validators."""

from __future__ import annotations

from typing import Any, Mapping

from reframx.core.components.handle import ComponentHandle
from reframx.core.exceptions import ValidatorContractError
from reframx.core.validation.context import ValidationContext
from reframx.core.validation.report import ValidationReport


def run_component_validator(
    handle: ComponentHandle,
    *,
    peers: Mapping[str, Any] | None = None,
    experiment: Any | None = None,
) -> ValidationReport:
    """Run a component's optional validator and return its report.
    
    Exceptions raised inside user validation code intentionally propagate. A
    researcher should see programming errors in their validator rather than
    having them silently converted into compatibility warnings.
    """
    validator = handle.get_custom_validator()
    if validator is None:
        return ValidationReport()
    context = ValidationContext(
        component = handle.instance,
        component_spec = handle.spec,
        role = handle.role,
        peers = peers or {},
        experiment = experiment,
    )
    result = validator(context)
    if not isinstance(result, ValidationReport):
        raise ValidatorContractError(
            f"{type(handle.instance).__name__}.validate_component() must return "
            f"ValidationReport; got {type(result).__name__}."
        )
    return result