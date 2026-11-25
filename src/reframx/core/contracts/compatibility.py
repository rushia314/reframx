"""Generic compatibility checks between component requirements and capabilities.

A component declares what it requires through ``Requirement`` objects and
``RequirementGroup`` objects. Other components declare what they provide
through ``Capability`` objects. This module only matches these contracts; it
does not encode fixed relationships such as Task -> Method or Method -> Model.

Supported requirement forms
---------------------------
* ``Requirement``: one capability must be provided.
* ``RequirementGroup(mode="all_of")``: every child requirement must pass.
* ``RequirementGroup(mode="any_of")``: at least one child requirement must pass.

If a requirement has an ``IOContract`` in its optional ``contract`` attribute,
the checker compares its declared fields with the provider capability's
contract. The attribute is accessed dynamically so this module remains usable
while the IOContract model is being introduced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from reframx.core.components.specification import ComponentSpec
from reframx.core.contracts.requirements import Requirement, RequirementGroup
from reframx.core.validation.report import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)


@dataclass(frozen=True, slots=True)
class _ContractComparison:
    """Result of comparing one required contract with one provided contract."""

    mismatches: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()

    @property
    def is_compatible(self) -> bool:
        """Whether no definite contradiction was found."""
        return not self.mismatches

    @property
    def is_verified(self) -> bool:
        """Whether the contracts match and every requested field was declared."""
        return self.is_compatible and not self.unknowns


@dataclass(frozen=True, slots=True)
class _RequirementResult:
    """Internal result used when evaluating requirement groups."""

    status: Literal["satisfied", "unverified", "unsatisfied"]
    report: ValidationReport
    details: Mapping[str, Any]


def _read_contract_value(contract: Any, field_name: str) -> Any:
    """Read a contract field from either a mapping or an object."""
    if contract is None:
        return None
    if isinstance(contract, Mapping):
        return contract.get(field_name)
    return getattr(contract, field_name, None)


def _compare_shape(
    required_shape: Any,
    provided_shape: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Compare shapes and return ``(mismatches, unknowns)``.

    Integer dimensions are concrete and must match. String dimensions are
    symbolic (for example ``"B"`` or ``"T"``) and act as wildcards. A
    provider-side ``None`` or symbolic dimension cannot prove a concrete
    required dimension, so that comparison is marked unknown.
    """
    try:
        required_dims = tuple(required_shape)
        provided_dims = tuple(provided_shape)
    except TypeError:
        return (("shape must be an iterable of dimensions",), ())

    if len(required_dims) != len(provided_dims):
        return (
            (
                f"rank mismatch: required rank {len(required_dims)}, "
                f"provided rank {len(provided_dims)}",
            ),
            (),
        )

    mismatches: list[str] = []
    unknowns: list[str] = []

    for index, (required_dim, provided_dim) in enumerate(
        zip(required_dims, provided_dims, strict=True)
    ):
        required_is_concrete = isinstance(required_dim, int) and not isinstance(
            required_dim, bool
        )
        provided_is_concrete = isinstance(provided_dim, int) and not isinstance(
            provided_dim, bool
        )

        if required_is_concrete:
            if provided_is_concrete:
                if required_dim != provided_dim:
                    mismatches.append(
                        f"dimension {index} mismatch: required {required_dim}, "
                        f"provided {provided_dim}"
                    )
            elif provided_dim is None or isinstance(provided_dim, str):
                unknowns.append(
                    f"provider dimension {index} is symbolic or unspecified; "
                    f"cannot verify required size {required_dim}"
                )
            else:
                mismatches.append(
                    f"provider dimension {index} has unsupported value "
                    f"{provided_dim!r}"
                )

    return tuple(mismatches), tuple(unknowns)


def _compare_io_contracts(
    requirement_contract: Any,
    capability_contract: Any,
) -> _ContractComparison:
    """Compare fields understood by the generic IO contract.

    Fields omitted by the requirement impose no constraint. If the requirement
    specifies a field but the provider does not declare it, the comparison is
    unknown rather than a definite mismatch. ``strict_contract=True`` turns
    such unknowns into validation errors at the call site.
    """
    if requirement_contract is None:
        return _ContractComparison()

    mismatches: list[str] = []
    unknowns: list[str] = []

    for field_name in ("kind", "value_type", "semantics", "dtype"):
        required_value = _read_contract_value(requirement_contract, field_name)
        if required_value is None:
            continue

        provided_value = _read_contract_value(capability_contract, field_name)
        if provided_value is None:
            unknowns.append(f"provider did not declare '{field_name}'")
        elif provided_value != required_value:
            mismatches.append(
                f"{field_name} mismatch: required {required_value!r}, "
                f"provided {provided_value!r}"
            )

    required_shape = _read_contract_value(requirement_contract, "shape")
    if required_shape is not None:
        provided_shape = _read_contract_value(capability_contract, "shape")
        if provided_shape is None:
            unknowns.append("provider did not declare 'shape'")
        else:
            shape_mismatches, shape_unknowns = _compare_shape(
                required_shape,
                provided_shape,
            )
            mismatches.extend(shape_mismatches)
            unknowns.extend(shape_unknowns)

    required_attributes = _read_contract_value(
        requirement_contract,
        "attributes",
    ) or {}
    provided_attributes = _read_contract_value(
        capability_contract,
        "attributes",
    ) or {}

    if not isinstance(required_attributes, Mapping):
        mismatches.append("required contract 'attributes' must be a mapping")
    elif not isinstance(provided_attributes, Mapping):
        unknowns.append("provider contract 'attributes' is not a mapping")
    else:
        for key, required_value in required_attributes.items():
            if key not in provided_attributes:
                unknowns.append(f"provider did not declare attribute '{key}'")
            elif provided_attributes[key] != required_value:
                mismatches.append(
                    f"attribute '{key}' mismatch: required {required_value!r}, "
                    f"provided {provided_attributes[key]!r}"
                )

    return _ContractComparison(
        mismatches=tuple(mismatches),
        unknowns=tuple(unknowns),
    )


def _make_issue(
    *,
    consumer: ComponentSpec,
    code: str,
    message: str,
    severity: ValidationSeverity,
    location: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> ValidationIssue:
    """Create a standardized issue associated with the consuming component."""
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        component=consumer.metadata.name,
        location=location,
        details=dict(details or {}),
    )


def _evaluate_requirement(
    *,
    consumer: ComponentSpec,
    requirement: Requirement,
    providers: tuple[ComponentSpec, ...],
    strict_contract: bool,
    optional_override: bool = False,
) -> _RequirementResult:
    """Evaluate one Requirement and return its report plus an internal status."""
    requirement_name = requirement.name
    expected_provider_type = getattr(requirement, "provider_type", None)
    is_optional = bool(getattr(requirement, "optional", False) or optional_override)
    severity_for_failure = (
        ValidationSeverity.WARNING if is_optional else ValidationSeverity.ERROR
    )

    named_candidates = [
        (provider, capability)
        for provider in providers
        for capability in provider.capabilities
        if capability.name == requirement_name
    ]

    candidates = [
        (provider, capability)
        for provider, capability in named_candidates
        if expected_provider_type is None
        or provider.metadata.component_type == expected_provider_type
    ]

    if not candidates:
        if named_candidates and expected_provider_type is not None:
            code = "CAPABILITY_PROVIDER_TYPE_MISMATCH"
            reason = (
                f"Capability '{requirement_name}' exists, but not on a provider "
                f"of type '{expected_provider_type}'."
            )
        else:
            code = (
                "OPTIONAL_CAPABILITY_MISSING"
                if is_optional
                else "REQUIRED_CAPABILITY_MISSING"
            )
            if expected_provider_type is None:
                reason = f"No provider offers capability '{requirement_name}'."
            else:
                reason = (
                    f"No provider of type '{expected_provider_type}' offers "
                    f"capability '{requirement_name}'."
                )

        if is_optional and code == "CAPABILITY_PROVIDER_TYPE_MISMATCH":
            code = f"OPTIONAL_{code}"

        report = ValidationReport(
            issues=[
                _make_issue(
                    consumer=consumer,
                    code=code,
                    message=(
                        f"Optional requirement '{requirement_name}' is not satisfied. "
                        f"{reason}"
                        if is_optional
                        else f"Required requirement '{requirement_name}' is not satisfied. "
                        f"{reason}"
                    ),
                    severity=severity_for_failure,
                    location=expected_provider_type,
                    details={
                        "requirement": requirement_name,
                        "provider_type": expected_provider_type,
                        "optional": is_optional,
                        "constraints": dict(getattr(requirement, "constraints", {})),
                    },
                )
            ]
        )
        return _RequirementResult(
            status="unsatisfied",
            report=report,
            details={"code": code, "reason": reason},
        )

    requirement_contract = getattr(requirement, "contract", None)
    comparisons: list[tuple[ComponentSpec, _ContractComparison]] = []
    for provider, capability in candidates:
        comparison = _compare_io_contracts(
            requirement_contract,
            getattr(capability, "contract", None),
        )
        comparisons.append((provider, comparison))

    verified_candidates = [
        (provider, comparison)
        for provider, comparison in comparisons
        if comparison.is_verified
    ]
    unverified_candidates = [
        (provider, comparison)
        for provider, comparison in comparisons
        if comparison.is_compatible and comparison.unknowns
    ]

    if verified_candidates:
        return _RequirementResult(
            status="satisfied",
            report=ValidationReport(),
            details={
                "requirement": requirement_name,
                "providers": [provider.metadata.name for provider, _ in verified_candidates],
            },
        )

    if unverified_candidates:
        unknown_details = [
            {
                "provider": provider.metadata.name,
                "unknowns": list(comparison.unknowns),
            }
            for provider, comparison in unverified_candidates
        ]
        report_severity = (
            ValidationSeverity.WARNING
            if is_optional or not strict_contract
            else ValidationSeverity.ERROR
        )
        report = ValidationReport(
            issues=[
                _make_issue(
                    consumer=consumer,
                    code="CAPABILITY_CONTRACT_UNVERIFIED",
                    message=(
                        f"Requirement '{requirement_name}' may be satisfied, but "
                        "some contract fields could not be verified."
                    ),
                    severity=report_severity,
                    location=expected_provider_type,
                    details={
                        "requirement": requirement_name,
                        "providers": unknown_details,
                        "strict_contract": strict_contract,
                    },
                )
            ]
        )
        return _RequirementResult(
            status="unverified",
            report=report,
            details={
                "requirement": requirement_name,
                "unknowns": unknown_details,
            },
        )

    mismatch_details = [
        {
            "provider": provider.metadata.name,
            "mismatches": list(comparison.mismatches),
        }
        for provider, comparison in comparisons
    ]
    report = ValidationReport(
        issues=[
            _make_issue(
                consumer=consumer,
                code=(
                    "OPTIONAL_CAPABILITY_CONTRACT_MISMATCH"
                    if is_optional
                    else "CAPABILITY_CONTRACT_MISMATCH"
                ),
                message=(
                    f"Capability '{requirement_name}' was found, but no provider "
                    "satisfies its declared contract."
                ),
                severity=severity_for_failure,
                location=expected_provider_type,
                details={
                    "requirement": requirement_name,
                    "candidates": mismatch_details,
                },
            )
        ]
    )
    return _RequirementResult(
        status="unsatisfied",
        report=report,
        details={
            "requirement": requirement_name,
            "candidates": mismatch_details,
        },
    )


def _validate_requirement_group(
    *,
    consumer: ComponentSpec,
    group: RequirementGroup,
    providers: tuple[ComponentSpec, ...],
    strict_contract: bool,
) -> ValidationReport:
    """Validate an all-of or any-of RequirementGroup."""
    group_name = group.name
    mode = group.mode
    group_is_optional = bool(group.optional)

    results = [
        _evaluate_requirement(
            consumer=consumer,
            requirement=requirement,
            providers=providers,
            strict_contract=strict_contract,
            optional_override=(group_is_optional if mode == "all_of" else False),
        )
        for requirement in group.requirements
    ]

    if mode == "all_of":
        report = ValidationReport()
        for result in results:
            report.extend(result.report)
        return report

    if mode != "any_of":
        # RequirementGroup should already validate this in __post_init__, but
        # keep the checker defensive against custom or malformed objects.
        raise ValueError(
            f"Unsupported RequirementGroup mode {mode!r}; "
            "expected 'all_of' or 'any_of'."
        )

    if any(result.status == "satisfied" for result in results):
        # One fully verified alternative is enough. Failed alternatives are not
        # validation errors because they were not selected.
        return ValidationReport()

    unverified_results = [
        result for result in results if result.status == "unverified"
    ]
    if unverified_results:
        severity = (
            ValidationSeverity.WARNING
            if group_is_optional or not strict_contract
            else ValidationSeverity.ERROR
        )
        code = "REQUIREMENT_GROUP_UNVERIFIED"
        message = (
            f"Requirement group '{group_name}' has no fully verified alternative; "
            "at least one alternative may be satisfied but has incomplete contract data."
        )
        details = {
            "group": group_name,
            "mode": mode,
            "alternatives": [dict(result.details) for result in results],
            "strict_contract": strict_contract,
        }
    else:
        severity = (
            ValidationSeverity.WARNING if group_is_optional else ValidationSeverity.ERROR
        )
        code = "OPTIONAL_REQUIREMENT_GROUP_UNSATISFIED" if group_is_optional else (
            "REQUIREMENT_GROUP_UNSATISFIED"
        )
        message = (
            f"No alternative in requirement group '{group_name}' was satisfied."
        )
        details = {
            "group": group_name,
            "mode": mode,
            "alternatives": [dict(result.details) for result in results],
        }

    return ValidationReport(
        issues=[
            _make_issue(
                consumer=consumer,
                code=code,
                message=message,
                severity=severity,
                location=group_name,
                details=details,
            )
        ]
    )


def validate_requirements(
    consumer: ComponentSpec,
    providers: Iterable[ComponentSpec],
    *,
    strict_contract: bool = False,
) -> ValidationReport:
    """Validate a component's requirements against available providers.

    A component may declare individual ``Requirement`` objects or grouped
    requirements using ``RequirementGroup``.

    Matching rules
    --------------
    1. Requirement and Capability names must match.
    2. ``provider_type`` is an optional provider filter when set to ``None``.
    3. If a requirement has an IO contract, its declared fields are compared
       with the matching capability's optional ``contract`` attribute.
    4. Missing provider-side contract details are warnings by default, or
       errors with ``strict_contract=True``.
    5. Optional individual requirements and optional groups produce warnings
       instead of errors when they are not satisfied.
    6. ``all_of`` groups require every member; ``any_of`` groups require at
       least one member. Failed alternatives in a successful ``any_of`` group
       do not create validation issues.

    The function always returns a ValidationReport, including when the
    consumer declares no requirements.
    """
    if not isinstance(consumer, ComponentSpec):
        raise TypeError("consumer must be a ComponentSpec.")

    provider_specs = tuple(providers)
    for provider in provider_specs:
        if not isinstance(provider, ComponentSpec):
            raise TypeError(
                "providers must contain only ComponentSpec objects; "
                f"got {type(provider).__name__}."
            )

    report = ValidationReport()

    for item in consumer.requirements:
        if isinstance(item, RequirementGroup):
            report.extend(
                _validate_requirement_group(
                    consumer=consumer,
                    group=item,
                    providers=provider_specs,
                    strict_contract=strict_contract,
                )
            )
        elif isinstance(item, Requirement):
            result = _evaluate_requirement(
                consumer=consumer,
                requirement=item,
                providers=provider_specs,
                strict_contract=strict_contract,
            )
            report.extend(result.report)
        else:
            raise TypeError(
                "ComponentSpec.requirements must contain Requirement or "
                f"RequirementGroup objects; got {type(item).__name__}."
            )

    return report
