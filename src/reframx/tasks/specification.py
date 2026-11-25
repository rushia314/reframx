from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec
from reframx.core.contracts.requirements import Requirement, RequirementGroup
from reframx.tasks.definitions.evaluation import EvaluationDefinition
from reframx.tasks.definitions.objectives import ObjectiveGoal
from reframx.tasks.definitions.problem import ProblemDefinition


def _default_task_metadata() -> ComponentMetadata:
    return ComponentMetadata(name="Task", component_type="task")


@dataclass(frozen=True, slots=True)
class TaskSpec(ComponentSpec):
    metadata: ComponentMetadata = field(default_factory=_default_task_metadata)
    problem: ProblemDefinition = field(default_factory=ProblemDefinition)

    objective_goals: tuple[ObjectiveGoal, ...] = ()

    evaluation: EvaluationDefinition = field(default_factory=EvaluationDefinition)

    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ComponentSpec.__post_init__(self)

        problem = self.problem
        if isinstance(problem, Mapping):
            problem = ProblemDefinition(**problem)
        elif not isinstance(problem, ProblemDefinition):
            raise TypeError("TaskSpec.problem must be ProblemDefinition or Mapping.")

        evaluation = self.evaluation
        if isinstance(evaluation, Mapping):
            evaluation = EvaluationDefinition(**evaluation)
        elif not isinstance(evaluation, EvaluationDefinition):
            raise TypeError(
                "TaskSpec.evaluation must be EvaluationDefinition or Mapping."
            )

        normalized_goals: list[ObjectiveGoal] = []
        for item in self.objective_goals:
            if isinstance(item, ObjectiveGoal):
                normalized_goals.append(item)
            elif isinstance(item, Mapping):
                normalized_goals.append(ObjectiveGoal(**item))
            else:
                raise TypeError(
                    "TaskSpec.objective_goals must contain ObjectiveGoal or Mapping objects."
                )

        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "objective_goals", tuple(normalized_goals))
        object.__setattr__(self, "constraints", dict(self.constraints))

    @property
    def accepted_outputs(self) -> tuple[Requirement | RequirementGroup, ...]:
        """Outputs accepted by this task from external components."""
        return self.requirements