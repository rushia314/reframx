"""Minimal demonstration of the Common Component Interface."""

from reframx import (
    Capability,
    ComponentHandle,
    ComponentMetadata,
    ComponentSpec,
    Requirement,
    validate_requirements,
)


class CustomTransformer:
    """Example user-defined model; it does not inherit from the framework."""

    def get_spec(self) -> ComponentSpec:
        return ComponentSpec(
            metadata=ComponentMetadata(
                name="CustomTransformer",
                component_type="model",
                version="0.1.0",
                description="A user-created transformer model.",
            ),
            capabilities=(
                Capability("causal_forward"),
                Capability("token_logits"),
                Capability("hidden_states"),
            ),
            attributes={"num_layers": 12, "hidden_size": 768},
        )


class CustomSFTMethod:
    """Example method declaring what it needs from a model."""

    def get_spec(self) -> ComponentSpec:
        return ComponentSpec(
            metadata=ComponentMetadata(
                name="CustomSFTMethod",
                component_type="method",
                version="0.1.0",
            ),
            requirements=(
                Requirement("causal_forward", provider_type="model"),
                Requirement("token_logits", provider_type="model"),
                Requirement(
                    "sequence_log_probs",
                    provider_type="model",
                    optional=True,
                ),
            ),
        )


if __name__ == "__main__":
    model = CustomTransformer()
    method = CustomSFTMethod()

    model_handle = ComponentHandle.wrap(model)
    method_handle = ComponentHandle.wrap(method)

    report = validate_requirements(
        consumer=method_handle.spec,
        providers=[model_handle.spec],
    )

    print(f"Compatible: {report.is_valid}")
    for issue in report.issues:
        print(f"[{issue.severity.value.upper()}] {issue.code}: {issue.message}")
