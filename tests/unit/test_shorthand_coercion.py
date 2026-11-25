"""Tests for ergonomic str/dict shorthand coercion across contracts and specs."""

from __future__ import annotations

import pytest

import reframx as rf


def test_task_and_requirement_group_shorthand_coercion():
    task = rf.BaseTask(
        requirements=[
            rf.RequirementGroup(
                name="prediction",
                mode="any_of",
                requirements=[
                    rf.Requirement(
                        "logits",
                        contract={
                            "kind": "data",
                            "value_type": "tensor",
                            "semantics": "multilabel_logits",
                            "shape": ("B", 4),
                        },
                    ),
                    "probabilities",
                ],
            ),
        ],
        problem={
            "name": "Multi-Label Classification",
            "output_semantics": "independent_binary_scores",
        },
        objective_goals=[
            {"name": "bce", "semantics": "binary_error", "direction": "minimize"}
        ],
        evaluation={"primary_metrics": ("macro_f1",)},
    )

    spec = task.spec
    assert isinstance(spec.problem, rf.ProblemDefinition)
    assert spec.problem.name == "Multi-Label Classification"
    assert isinstance(spec.evaluation, rf.EvaluationDefinition)
    assert spec.evaluation.primary_metrics == ("macro_f1",)
    assert isinstance(spec.objective_goals[0], rf.ObjectiveGoal)

    group = spec.accepted_outputs[0]
    assert isinstance(group, rf.RequirementGroup)
    assert group.requirements[0].name == "logits"
    assert isinstance(group.requirements[0].contract, rf.IOContract)
    assert group.requirements[0].contract.shape == ("B", 4)
    assert group.requirements[1] == rf.Requirement("probabilities")


def test_component_spec_accepts_str_and_dict_shorthands():
    consumer = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("MyMethod", "method"),
        requirements=(
            "inputs",
            {
                "name": "logits",
                "provider_type": "model",
                "contract": {"dtype": "float32", "shape": ("B", "C")},
            },
        ),
        capabilities=("predictions",),
    )
    dataset = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("MyDataset", "dataset"),
        capabilities=("inputs",),
    )
    model = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("MyModel", "model"),
        capabilities=(
            {
                "name": "logits",
                "contract": {"dtype": "float32", "shape": ("B", "C")},
            },
        ),
    )

    report = rf.validate_requirements(consumer, [dataset, model], strict_contract=True)
    assert report.is_valid
    assert report.issues == []


def test_invalid_shorthand_dict_key_raises_type_error():
    with pytest.raises(TypeError):
        rf.Requirement("logits", contract={"unknown_field": 123})  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        rf.TaskSpec(problem={"not_a_field": "x"})  # type: ignore[arg-type]


def test_base_task_accepts_single_dict_or_str_directly_for_requirements():
    task_from_dict = rf.BaseTask(
        requirements={
            "name": "prediction",
            "mode": "any_of",
            "requirements": [
                {"name": "logits", "contract": {"shape": ("B", 4)}},
                "probabilities",
            ],
        },
        capabilities="task_cap",
    )
    group = task_from_dict.spec.accepted_outputs[0]
    assert isinstance(group, rf.RequirementGroup)
    assert group.name == "prediction"
    assert group.mode == "any_of"
    assert group.requirements[0].contract.shape == ("B", 4)
    assert group.requirements[1] == rf.Requirement("probabilities")
    assert task_from_dict.spec.capabilities == (rf.Capability("task_cap"),)

    task_from_str = rf.BaseTask(requirements="logits")
    assert task_from_str.spec.accepted_outputs == (rf.Requirement("logits"),)


def test_requirement_group_auto_generates_name_when_omitted():
    task = rf.BaseTask(
        requirements={
            "mode": "any_of",
            "requirements": ["logits", "probabilities"],
        }
    )
    group = task.spec.accepted_outputs[0]
    assert isinstance(group, rf.RequirementGroup)
    assert group.name == "any_of(logits, probabilities)"

    with pytest.raises(ValueError, match="non-empty"):
        rf.RequirementGroup(name="   ", requirements=(rf.Requirement("logits"),))


def test_model_adapter_and_base_components_accept_single_str_and_dict_shorthand():
    m1 = rf.ModelAdapter(object(), capabilities="forward")
    assert m1.spec.capabilities == (rf.Capability("forward"),)

    m2 = rf.ModelAdapter(
        object(),
        capabilities={
            "name": "forward",
            "contract": {"kind": "api", "value_type": "callable"},
        },
    )
    assert m2.spec.capabilities[0].contract == rf.IOContract(
        kind="api", value_type="callable"
    )

    ds = rf.BaseDataset(capabilities=["inputs", "targets"], splits_data={"train": []})
    assert [c.name for c in ds.spec.capabilities] == ["inputs", "targets"]

    class MinimalLoss(rf.BaseObjective):
        def compute(self, pred, target):
            return 0.0

    # Default capability for BaseObjective is ("loss",)
    assert MinimalLoss().spec.capabilities == (rf.Capability("loss"),)

    class CustomLoss(rf.BaseObjective):
        def __init__(self) -> None:
            super().__init__(capabilities=["loss", "huber_loss"])

        def compute(self, pred, target):
            return 0.0

    assert [c.name for c in CustomLoss().spec.capabilities] == ["loss", "huber_loss"]
