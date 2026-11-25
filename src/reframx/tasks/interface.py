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
from reframx.tasks.definitions.evaluation import EvaluationDefinition
from reframx.tasks.definitions.objectives import ObjectiveGoal
from reframx.tasks.definitions.problem import ProblemDefinition
from reframx.tasks.specification import TaskSpec


class BaseTask:
    def __init__(
        self,
        *,
        requirements: RequirementsInput = (),
        metadata: ComponentMetadata | None = None,
        capabilities: CapabilitiesInput = (),
        problem: ProblemDefinition | Mapping[str, Any] | None = None,
        objective_goals: Iterable[ObjectiveGoal | Mapping[str, Any]] = (),
        evaluation: EvaluationDefinition | Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:

        task_metadata = metadata or ComponentMetadata(
            name=type(self).__name__,
            component_type="task",
        )
        self._spec = TaskSpec(
            metadata=task_metadata,
            requirements=coerce_requirements(requirements),
            capabilities=coerce_capabilities(capabilities),
            attributes=attributes or {},
            problem=problem if problem is not None else ProblemDefinition(),  # type: ignore[arg-type]
            objective_goals=tuple(objective_goals),  # type: ignore[arg-type]
            evaluation=(
                evaluation if evaluation is not None else EvaluationDefinition()  # type: ignore[arg-type]
            ),
            constraints=constraints or {},
        )

    def get_spec(self) -> TaskSpec:
        return self._spec   

    @property
    def spec(self) -> TaskSpec:
        return self._spec