"""Custom Task example: contract-driven Task with shorthand RequirementGroup & IOContract.

Demonstrates that a Task only defines accepted outputs — multiple different
Methods (emitting either `logits` or `probabilities`) satisfy the same Task
without modifying the Task itself.
"""

from __future__ import annotations

import reframx as rf


class MultiLabelTask(rf.BaseTask):
    def __init__(self, num_labels: int) -> None:
        super().__init__(
            requirements={
                "mode": "any_of",
                "requirements": [
                    {
                        "name": "logits",
                        "contract": {
                            "kind": "data",
                            "value_type": "tensor",
                            "semantics": "multilabel_logits",
                            "shape": ("B", num_labels),
                        },
                    },
                    "probabilities",
                ],
            },
            problem={
                "name": "Multi-Label Classification",
                "output_semantics": "independent_binary_scores",
            },
            evaluation={"primary_metrics": ("macro_f1",)},
            attributes={"num_labels": num_labels},
        )


def main() -> None:
    task = MultiLabelTask(num_labels=4)

    logit_method = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("LogitMethod", "method"),
        capabilities={
            "name": "logits",
            "contract": {
                "kind": "data",
                "value_type": "tensor",
                "semantics": "multilabel_logits",
                "shape": ("B", 4),
            },
        },
    )
    prob_method = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("ProbMethod", "method"),
        capabilities="probabilities",
    )
    incompatible_method = rf.ComponentSpec(
        metadata=rf.ComponentMetadata("WrongShapeMethod", "method"),
        capabilities={
            "name": "logits",
            "contract": {
                "kind": "data",
                "value_type": "tensor",
                "semantics": "multilabel_logits",
                "shape": ("B", 10),
            },
        },
    )

    for method in (logit_method, prob_method, incompatible_method):
        report = rf.validate_requirements(task.spec, [method], strict_contract=True)
        print(f"{method.metadata.name:18s} -> compatible={report.is_valid}")
        for issue in report.issues:
            print(f"   [{issue.severity.value.upper()}] {issue.code}: {issue.message}")


if __name__ == "__main__":
    main()
