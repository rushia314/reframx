import pytest

from reframx import (
    ComponentHandle,
    ComponentMetadata,
    ComponentSpec,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    run_component_validator,
)
from reframx.core.exceptions import ValidatorContractError


def test_report_validity_and_serialization():
    report = ValidationReport()
    report.add(
        ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code="OPTIONAL_MISSING",
            message="An optional capability is missing.",
        )
    )

    assert report.is_valid
    assert len(report.warnings) == 1
    assert report.to_dict()["is_valid"] is True

    report.add(
        ValidationIssue(
            severity="error",
            code="INVALID_MODEL",
            message="Model does not satisfy the contract.",
        )
    )
    assert not report.is_valid


def test_custom_validator_is_called():
    class UserComponent:
        def validate_component(self, context):
            assert context.role == "model"
            return ValidationReport(
                issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        code="CUSTOM_CHECK_PASSED",
                        message="User validation completed.",
                    )
                ]
            )

    spec = ComponentSpec(
        metadata=ComponentMetadata("UserComponent", "model")
    )
    handle = ComponentHandle.wrap(UserComponent(), spec=spec)
    report = run_component_validator(handle)

    assert report.is_valid
    assert report.infos[0].code == "CUSTOM_CHECK_PASSED"


def test_invalid_custom_validator_result_is_rejected():
    class BadComponent:
        def validate_component(self, context):
            return True

    handle = ComponentHandle.wrap(BadComponent(), role="model")
    with pytest.raises(ValidatorContractError):
        run_component_validator(handle)
