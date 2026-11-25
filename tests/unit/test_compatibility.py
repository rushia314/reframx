from reframx import (
    Capability,
    ComponentMetadata,
    ComponentSpec,
    Requirement,
    ValidationSeverity,
    validate_requirements,
)


def test_required_capability_is_satisfied():
    method = ComponentSpec(
        metadata=ComponentMetadata("SFT", "method"),
        requirements=(Requirement("token_logits", "model"),),
    )
    model = ComponentSpec(
        metadata=ComponentMetadata("Transformer", "model"),
        capabilities=(Capability("token_logits"),),
    )

    report = validate_requirements(method, [model])
    assert report.is_valid
    assert report.issues == []


def test_missing_required_capability_is_error():
    method = ComponentSpec(
        metadata=ComponentMetadata("SFT", "method"),
        requirements=(Requirement("sequence_log_probs", "model"),),
    )
    model = ComponentSpec(metadata=ComponentMetadata("Transformer", "model"))

    report = validate_requirements(method, [model])
    assert not report.is_valid
    assert report.errors[0].code == "REQUIRED_CAPABILITY_MISSING"


def test_missing_optional_capability_is_warning_not_error():
    method = ComponentSpec(
        metadata=ComponentMetadata("SFT", "method"),
        requirements=(Requirement("hidden_states", "model", optional=True),),
    )
    model = ComponentSpec(metadata=ComponentMetadata("Transformer", "model"))

    report = validate_requirements(method, [model])
    assert report.is_valid
    assert len(report.warnings) == 1
    assert report.warnings[0].severity is ValidationSeverity.WARNING


def test_capability_from_wrong_provider_type_does_not_match():
    method = ComponentSpec(
        metadata=ComponentMetadata("SFT", "method"),
        requirements=(Requirement("token_logits", "model"),),
    )
    dataset = ComponentSpec(
        metadata=ComponentMetadata("MyDataset", "dataset"),
        capabilities=(Capability("token_logits"),),
    )

    report = validate_requirements(method, [dataset])
    assert not report.is_valid
