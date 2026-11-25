"""Unit and integration tests for DatasetFieldAdapter."""

from __future__ import annotations

import reframx as rf


def test_dataset_field_adapter_renames_capabilities_and_preserves_io_contract():
    image_contract = rf.IOContract(
        kind="data",
        value_type="tensor",
        semantics="rgb_image",
        dtype="float32",
        shape=("B", 3, 32, 32),
    )
    source = rf.BaseDataset(
        capabilities=(
            rf.Capability("image", contract=image_contract),
            rf.Capability("label"),
            rf.Capability("sample_id"),
        ),
        splits_data={
            "train": [{"image": [1, 2], "label": 1, "sample_id": "a"}],
            "val": [{"image": [3, 4], "label": 0, "sample_id": "b"}],
        },
    )

    adapted = rf.DatasetFieldAdapter(
        source,
        field_map={"image": "inputs", "label": "targets"},
        keep_unmapped=True,
    )

    cap_by_name = {c.name: c for c in adapted.spec.capabilities}
    assert set(cap_by_name.keys()) == {"inputs", "targets", "sample_id"}
    assert cap_by_name["inputs"].contract == image_contract

    # Re-iterable across multiple epochs.
    for _ in range(2):
        train_batches = list(adapted.get_split("train"))
        assert train_batches == [{"inputs": [1, 2], "targets": 1, "sample_id": "a"}]

    assert list(adapted.get_split("val")) == [
        {"inputs": [3, 4], "targets": 0, "sample_id": "b"}
    ]

    # Original dataset is not mutated.
    assert [c.name for c in source.spec.capabilities] == ["image", "label", "sample_id"]


def test_dataset_field_adapter_drop_unmapped_and_custom_transform():
    source = rf.BaseDataset(
        capabilities=(rf.Capability("raw_x"), rf.Capability("raw_y"), rf.Capability("extra")),
        splits_data={"train": [{"raw_x": 2.0, "raw_y": 5.0, "extra": "drop_me"}]},
    )

    adapted = rf.DatasetFieldAdapter(
        source,
        field_map={"raw_x": "x", "raw_y": "y"},
        keep_unmapped=False,
        transform=lambda b: {"x": b["x"] * 10.0, "y": b["y"]},
    )

    assert [c.name for c in adapted.spec.capabilities] == ["x", "y"]
    assert list(adapted.get_split("train")) == [{"x": 20.0, "y": 5.0}]


def test_dataset_field_adapter_satisfies_method_requirements_in_experiment():
    raw_dataset = rf.BaseDataset(
        capabilities=(rf.Capability("img"), rf.Capability("cls")),
        splits_data={"train": [{"img": 3.0, "cls": 6.0}]},
    )

    class ExpectsInputsTargetsMethod(rf.BaseMethod):
        def __init__(self) -> None:
            super().__init__(
                requirements=(
                    rf.Requirement("inputs", provider_type="dataset"),
                    rf.Requirement("targets", provider_type="dataset"),
                ),
                capabilities=(rf.Capability("predictions"),),
            )
            self.seen: list[dict[str, float]] = []

        def train_step(
            self,
            batch: dict[str, float],
            ctx: rf.ExecutionContext,
        ) -> rf.MethodResult:
            self.seen.append(batch)
            return rf.MethodResult(
                task_outputs={"predictions": batch["inputs"] * 2.0},
                metrics={"loss": 0.0},
            )

    # Without adapter, validation fails because 'inputs'/'targets' are missing.
    bad_exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=ExpectsInputsTargetsMethod(),
        dataset=raw_dataset,
    )
    assert not bad_exp.validate().is_valid

    # Wrapping with DatasetFieldAdapter satisfies contracts and runs cleanly.
    method = ExpectsInputsTargetsMethod()
    adapted_dataset = rf.DatasetFieldAdapter(
        raw_dataset,
        field_map={"img": "inputs", "cls": "targets"},
    )
    good_exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=method,
        dataset=adapted_dataset,
    )
    assert good_exp.validate().is_valid
    run = good_exp.run(max_epochs=1, eval_split=None)
    assert run.status is rf.RunStatus.COMPLETED
    assert method.seen == [{"inputs": 3.0, "targets": 6.0}]
