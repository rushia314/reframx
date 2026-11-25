"""Built-in Task definitions for common neural network research problems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.requirements import Requirement, RequirementGroup
from reframx.tasks.definitions.evaluation import EvaluationDefinition
from reframx.tasks.definitions.objectives import ObjectiveGoal
from reframx.tasks.definitions.problem import ProblemDefinition
from reframx.tasks.interface import BaseTask


class ClassificationTask(BaseTask):
    """Standard classification task accepting either ``logits`` or ``probabilities``."""

    def __init__(
        self,
        *,
        num_classes: int | None = None,
        metadata: ComponentMetadata | None = None,
        primary_metrics: Iterable[str] = ("accuracy",),
        secondary_metrics: Iterable[str] = (),
        constraints: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        attrs = dict(attributes or {})
        if num_classes is not None:
            attrs["num_classes"] = num_classes
        super().__init__(
            requirements=(
                RequirementGroup(
                    name="classification_output",
                    mode="any_of",
                    requirements=(
                        Requirement("logits"),
                        Requirement("probabilities"),
                    ),
                ),
            ),
            metadata=metadata
            or ComponentMetadata(name="ClassificationTask", component_type="task"),
            problem=ProblemDefinition(
                name="Classification",
                description="Assign discrete class predictions to input samples.",
                output_semantics="class_scores_or_probabilities",
            ),
            objective_goals=(
                ObjectiveGoal(
                    name="classification_error",
                    semantics="prediction_divergence",
                    direction="minimize",
                ),
            ),
            evaluation=EvaluationDefinition(
                primary_metrics=tuple(primary_metrics),
                secondary_metrics=tuple(secondary_metrics),
            ),
            constraints=constraints,
            attributes=attrs,
        )


class RegressionTask(BaseTask):
    """Standard regression task accepting continuous ``predictions``."""

    def __init__(
        self,
        *,
        metadata: ComponentMetadata | None = None,
        primary_metrics: Iterable[str] = ("mse", "mae"),
        secondary_metrics: Iterable[str] = (),
        constraints: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            requirements=(Requirement("predictions"),),
            metadata=metadata
            or ComponentMetadata(name="RegressionTask", component_type="task"),
            problem=ProblemDefinition(
                name="Regression",
                description="Predict continuous target values for input samples.",
                output_semantics="continuous_predictions",
            ),
            objective_goals=(
                ObjectiveGoal(
                    name="prediction_error",
                    semantics="continuous_error",
                    direction="minimize",
                ),
            ),
            evaluation=EvaluationDefinition(
                primary_metrics=tuple(primary_metrics),
                secondary_metrics=tuple(secondary_metrics),
            ),
            constraints=constraints,
            attributes=attributes,
        )


class CausalLMTask(BaseTask):
    """Autoregressive language modeling task accepting ``token_logits``."""

    def __init__(
        self,
        *,
        metadata: ComponentMetadata | None = None,
        primary_metrics: Iterable[str] = ("perplexity",),
        secondary_metrics: Iterable[str] = (),
        constraints: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            requirements=(Requirement("token_logits"),),
            metadata=metadata
            or ComponentMetadata(name="CausalLMTask", component_type="task"),
            problem=ProblemDefinition(
                name="Causal Language Modeling",
                description="Predict next-token distributions given preceding context.",
                output_semantics="token_prediction_scores",
            ),
            objective_goals=(
                ObjectiveGoal(
                    name="next_token_nll",
                    semantics="token_negative_log_likelihood",
                    direction="minimize",
                ),
            ),
            evaluation=EvaluationDefinition(
                primary_metrics=tuple(primary_metrics),
                secondary_metrics=tuple(secondary_metrics),
            ),
            constraints=constraints,
            attributes=attributes,
        )
