import pytest

from reframx import (
    BaseTask,
    Capability,
    ComponentHandle,
    ComponentMetadata,
    ComponentSpec,
    EvaluationDefinition,
    ObjectiveGoal,
    ProblemDefinition,
    Requirement,
    RequirementGroup,
    TaskSpec,
    validate_requirements,
)


def test_task_spec_defaults():
    spec = TaskSpec()

    assert spec.metadata.component_type == "task"
    assert spec.requirements == ()
    assert spec.objective_goals == ()
    assert isinstance(spec.problem, ProblemDefinition)
    assert isinstance(spec.evaluation, EvaluationDefinition)


def test_task_spec_requires_only_requirements():
    requirements = (
        Requirement("task_output"),
    )

    spec = TaskSpec(requirements=requirements)

    assert spec.requirements == requirements
    assert spec.accepted_outputs == requirements


def test_task_spec_accepts_requirement_group():
    group = RequirementGroup(
        name="prediction",
        mode="any_of",
        requirements=(
            Requirement("logits"),
            Requirement("probabilities"),
        ),
    )

    spec = TaskSpec(requirements=(group,))

    assert spec.accepted_outputs == (group,)


def test_base_task_generates_metadata():
    class ClassificationTask(BaseTask):
        def __init__(self):
            super().__init__(
                requirements=(
                    Requirement("class_logits"),
                )
            )

    task = ClassificationTask()
    spec = task.get_spec()

    assert spec.metadata.name == "ClassificationTask"
    assert spec.metadata.component_type == "task"


def test_base_task_custom_definitions():
    task = BaseTask(
        requirements=(Requirement("prediction"),),
        problem=ProblemDefinition(
            name="Classification",
            description="Classify samples.",
        ),
        objective_goals=(
            ObjectiveGoal(
                name="accuracy",
                semantics="correct_prediction",
                direction="maximize",
            ),
        ),
        evaluation=EvaluationDefinition(
            primary_metrics=("accuracy",),
        ),
    )

    spec = task.get_spec()

    assert spec.problem.name == "Classification"
    assert spec.objective_goals[0].direction == "maximize"
    assert spec.evaluation.primary_metrics == ("accuracy",)


def test_base_task_works_with_component_handle_and_compatibility():
    task = BaseTask(
        requirements=(
            RequirementGroup(
                name="prediction",
                mode="any_of",
                requirements=(
                    Requirement("class_logits"),
                    Requirement("class_probabilities"),
                ),
            ),
        )
    )
    method_spec = ComponentSpec(
        metadata=ComponentMetadata(name="LogitMethod", component_type="method"),
        capabilities=(Capability("class_logits"),),
    )

    handle = ComponentHandle.wrap(task)
    assert handle.role == "task"
    assert handle.spec is task.spec

    report = validate_requirements(handle.spec, [method_spec])
    assert report.is_valid
    assert report.issues == []


def test_task_spec_and_definitions_validate_inputs():
    with pytest.raises(TypeError):
        TaskSpec(problem="not_a_problem")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        TaskSpec(evaluation="not_an_eval")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        TaskSpec(objective_goals=("not_a_goal",))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        ObjectiveGoal(name=" ", semantics="loss")

    with pytest.raises(ValueError):
        ObjectiveGoal(name="loss", semantics="nll", direction="sideways")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        EvaluationDefinition(primary_metrics=(" ",))