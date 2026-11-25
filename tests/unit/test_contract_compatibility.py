"""Regression tests for generic requirement/capability compatibility."""

from __future__ import annotations

import pytest

from reframx import (
    Capability,
    ComponentMetadata,
    ComponentSpec,
    Requirement,
    RequirementGroup,
    ValidationSeverity,
    validate_requirements,
)
from reframx.core.contracts.io import IOContract


def make_spec(
    name: str,
    component_type: str,
    *,
    requirements=(),
    capabilities=(),
) -> ComponentSpec:
    return ComponentSpec(
        metadata=ComponentMetadata(name, component_type),
        requirements=tuple(requirements),
        capabilities=tuple(capabilities),
    )


def tensor_contract(
    *,
    semantics: str = "token_prediction_scores",
    dtype: str = "float32",
    shape: tuple[int | str | None, ...] = ("B", "T", "V"),
    attributes=None,
) -> IOContract:
    return IOContract(
        kind="data",
        value_type="tensor",
        semantics=semantics,
        dtype=dtype,
        shape=shape,
        attributes=attributes or {},
    )


def test_all_of_group_passes_when_every_requirement_is_satisfied():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            Requirement("input_ids", "dataset"),
            Requirement("labels", "dataset"),
            Requirement("token_logits", "model"),
        ),
    )
    providers = [
        make_spec(
            "TrainDataset",
            "dataset",
            capabilities=(Capability("input_ids"), Capability("labels")),
        ),
        make_spec(
            "Transformer",
            "model",
            capabilities=(Capability("token_logits"),),
        ),
    ]

    report = validate_requirements(consumer, providers)

    assert report.is_valid
    assert report.issues == []


def test_all_of_group_reports_each_missing_requirement():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            RequirementGroup(
                name="training_inputs",
                mode="all_of",
                requirements=(
                    Requirement("input_ids", "dataset"),
                    Requirement("labels", "dataset"),
                ),
            ),
        ),
    )
    providers = [
        make_spec(
            "PartialDataset",
            "dataset",
            capabilities=(Capability("input_ids"),),
        )
    ]

    report = validate_requirements(consumer, providers)

    assert not report.is_valid
    assert [issue.code for issue in report.errors] == [
        "REQUIRED_CAPABILITY_MISSING"
    ]
    assert report.errors[0].details["requirement"] == "labels"


def test_any_of_group_passes_when_one_alternative_is_satisfied():
    consumer = make_spec(
        "Classification",
        "task",
        requirements=(
            RequirementGroup(
                name="prediction_output",
                mode="any_of",
                requirements=(
                    Requirement("class_logits", "method"),
                    Requirement("class_probabilities", "method"),
                ),
            ),
        ),
    )
    provider = make_spec(
        "LogitMethod",
        "method",
        capabilities=(Capability("class_logits"),),
    )

    report = validate_requirements(consumer, [provider])

    assert report.is_valid
    assert report.issues == []


def test_any_of_group_fails_when_no_alternative_is_satisfied():
    consumer = make_spec(
        "Classification",
        "task",
        requirements=(
            RequirementGroup(
                name="prediction_output",
                mode="any_of",
                requirements=(
                    Requirement("class_logits", "method"),
                    Requirement("class_probabilities", "method"),
                ),
            ),
        ),
    )

    report = validate_requirements(consumer, [])

    assert not report.is_valid
    assert len(report.errors) == 1
    assert report.errors[0].code == "REQUIREMENT_GROUP_UNSATISFIED"
    assert report.errors[0].details["group"] == "prediction_output"


def test_optional_group_missing_is_warning_not_error():
    consumer = make_spec(
        "OptionalTask",
        "task",
        requirements=(
            RequirementGroup(
                name="optional_diagnostics",
                mode="any_of",
                optional=True,
                requirements=(
                    Requirement("attention_maps", "model"),
                    Requirement("hidden_states", "model"),
                ),
            ),
        ),
    )

    report = validate_requirements(consumer, [])

    assert report.is_valid
    assert len(report.warnings) == 1
    assert report.warnings[0].severity is ValidationSeverity.WARNING
    assert report.warnings[0].code == "OPTIONAL_REQUIREMENT_GROUP_UNSATISFIED"


def test_any_of_group_suppresses_failed_alternatives_when_one_passes():
    consumer = make_spec(
        "Task",
        "task",
        requirements=(
            RequirementGroup(
                name="prediction",
                mode="any_of",
                requirements=(
                    Requirement("class_logits", "method"),
                    Requirement("class_probabilities", "method"),
                ),
            ),
        ),
    )
    provider = make_spec(
        "Method",
        "method",
        capabilities=(Capability("class_probabilities"),),
    )

    report = validate_requirements(consumer, [provider])

    assert report.is_valid
    assert report.issues == []


def test_io_contract_dtype_mismatch_is_reported():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            Requirement(
                "token_logits",
                "model",
                contract=tensor_contract(dtype="float32"),
            ),
        ),
    )
    provider = make_spec(
        "Transformer",
        "model",
        capabilities=(
            Capability(
                "token_logits",
                contract=tensor_contract(dtype="float16"),
            ),
        ),
    )

    report = validate_requirements(consumer, [provider])

    assert not report.is_valid
    assert report.errors[0].code == "CAPABILITY_CONTRACT_MISMATCH"
    mismatches = report.errors[0].details["candidates"][0]["mismatches"]
    assert any("dtype mismatch" in item for item in mismatches)


def test_io_contract_shape_rank_mismatch_is_reported():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            Requirement(
                "token_logits",
                "model",
                contract=tensor_contract(shape=("B", "T", "V")),
            ),
        ),
    )
    provider = make_spec(
        "Transformer",
        "model",
        capabilities=(
            Capability(
                "token_logits",
                contract=tensor_contract(shape=("B", "V")),
            ),
        ),
    )

    report = validate_requirements(consumer, [provider])

    assert not report.is_valid
    assert report.errors[0].code == "CAPABILITY_CONTRACT_MISMATCH"
    mismatches = report.errors[0].details["candidates"][0]["mismatches"]
    assert any("rank mismatch" in item for item in mismatches)


def test_contract_unknown_is_warning_by_default_and_error_in_strict_mode():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            Requirement(
                "token_logits",
                "model",
                contract=tensor_contract(dtype="float32"),
            ),
        ),
    )
    provider = make_spec(
        "Transformer",
        "model",
        capabilities=(Capability("token_logits", contract=IOContract()),),
    )

    permissive_report = validate_requirements(consumer, [provider])
    strict_report = validate_requirements(
        consumer,
        [provider],
        strict_contract=True,
    )

    assert permissive_report.is_valid
    assert len(permissive_report.warnings) == 1
    assert permissive_report.warnings[0].code == "CAPABILITY_CONTRACT_UNVERIFIED"

    assert not strict_report.is_valid
    assert len(strict_report.errors) == 1
    assert strict_report.errors[0].code == "CAPABILITY_CONTRACT_UNVERIFIED"


def test_multiple_providers_succeed_when_at_least_one_matches_contract():
    consumer = make_spec(
        "SFT",
        "method",
        requirements=(
            Requirement(
                "token_logits",
                provider_type=None,
                contract=tensor_contract(dtype="float32"),
            ),
        ),
    )
    providers = [
        make_spec(
            "HalfPrecisionModel",
            "model",
            capabilities=(
                Capability(
                    "token_logits",
                    contract=tensor_contract(dtype="float16"),
                ),
            ),
        ),
        make_spec(
            "FullPrecisionModel",
            "model",
            capabilities=(
                Capability(
                    "token_logits",
                    contract=tensor_contract(dtype="float32"),
                ),
            ),
        ),
    ]

    report = validate_requirements(consumer, providers)

    assert report.is_valid
    assert report.issues == []


def test_requirement_without_provider_type_accepts_any_provider_type():
    token_contract = IOContract(
        kind="data",
        value_type="tensor",
        dtype="int64",
        shape=("B", "T"),
    )
    consumer = make_spec(
        "SequenceTask",
        "task",
        requirements=(
            Requirement(
                "input_ids",
                provider_type=None,
                contract=token_contract,
            ),
        ),
    )
    dataset = make_spec(
        "TokenDataset",
        "dataset",
        capabilities=(
            Capability(
                "input_ids",
                contract=token_contract,
            ),
        ),
    )

    report = validate_requirements(consumer, [dataset])

    assert report.is_valid
    assert report.issues == []


def test_requirement_group_rejects_optional_children_in_any_of_mode():
    with pytest.raises(ValueError, match="cannot be optional"):
        RequirementGroup(
            name="prediction",
            mode="any_of",
            requirements=(
                Requirement("class_logits", "method", optional=True),
                Requirement("class_probabilities", "method"),
            ),
        )


def test_io_contract_defaults_and_normalization():
    empty = IOContract()
    assert empty.shape is None
    assert empty.attributes == {}

    attrs = {"normalized": True}
    contract = IOContract(shape=["B", "T"], attributes=attrs)  # type: ignore[arg-type]
    attrs["normalized"] = False

    assert contract.shape == ("B", "T")
    assert contract.attributes == {"normalized": True}
