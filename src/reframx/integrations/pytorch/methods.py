"""Built-in PyTorch Methods for common supervised training workflows."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import Capability
from reframx.core.contracts.requirements import Requirement
from reframx.methods.interface import MethodResult, SupervisedMethod


def extract_inputs_and_targets(
    batch: Any,
    *,
    input_key: str = "inputs",
    target_key: str = "targets",
) -> tuple[Any, Any]:
    """Extract ``(inputs, targets)`` from either a dict batch or ``(x, y)`` pair."""
    if isinstance(batch, Mapping):
        if input_key not in batch or target_key not in batch:
            raise KeyError(
                f"Batch mapping must contain '{input_key}' and '{target_key}'; "
                f"got keys {list(batch.keys())}."
            )
        return batch[input_key], batch[target_key]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        "Unsupported batch format; expected a mapping with input/target keys "
        f"or an (inputs, targets) sequence, got {type(batch).__name__}."
    )


class TorchSupervisedMethod(SupervisedMethod):
    """Ready-to-use supervised Method supporting dict and ``(x, y)`` batches.

    Works seamlessly with ``TorchModelAdapter``, ``TorchDatasetAdapter``,
    ``TorchStrategy`` (including AMP ``autocast_context``), and any built-in
    or custom ``BaseObjective``.
    """

    def __init__(
        self,
        *,
        input_key: str = "inputs",
        target_key: str = "targets",
        output_key: str = "logits",
        model_capability: str = "forward",
        objective_capability: str = "loss",
        require_dataset_fields: bool = False,
        use_autocast: bool = True,
        optimizer_name: str | None = None,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.output_key = output_key
        self.use_autocast = use_autocast
        self.optimizer_name = optimizer_name

        reqs: list[Requirement] = [
            Requirement(model_capability, provider_type="model"),
            Requirement(objective_capability, provider_type="objective"),
        ]
        if require_dataset_fields:
            reqs.extend(
                [
                    Requirement(input_key, provider_type="dataset"),
                    Requirement(target_key, provider_type="dataset"),
                ]
            )

        super().__init__(
            requirements=tuple(reqs),
            capabilities=(Capability(output_key),),
            metadata=metadata
            or ComponentMetadata(
                name=type(self).__name__,
                component_type="method",
            ),
            attributes={
                "framework": "pytorch",
                "input_key": input_key,
                "target_key": target_key,
                "output_key": output_key,
            },
        )

    def compute_step(self, batch: Any, ctx: Any) -> MethodResult:
        inputs, targets = extract_inputs_and_targets(
            batch,
            input_key=self.input_key,
            target_key=self.target_key,
        )
        autocast = getattr(ctx.strategy, "autocast_context", None)
        cm = autocast() if (self.use_autocast and callable(autocast)) else contextlib.nullcontext()
        with cm:
            outputs = ctx.model(inputs)
            loss = ctx.objective(outputs, targets)
        return MethodResult(
            task_outputs={self.output_key: outputs},
            loss=loss,
        )


class TorchClassificationMethod(TorchSupervisedMethod):
    """Preset ``TorchSupervisedMethod`` emitting ``logits`` for ``ClassificationTask``."""

    def __init__(
        self,
        *,
        input_key: str = "inputs",
        target_key: str = "targets",
        output_key: str = "logits",
        objective_capability: str = "loss",
        require_dataset_fields: bool = False,
        use_autocast: bool = True,
        optimizer_name: str | None = None,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        super().__init__(
            input_key=input_key,
            target_key=target_key,
            output_key=output_key,
            objective_capability=objective_capability,
            require_dataset_fields=require_dataset_fields,
            use_autocast=use_autocast,
            optimizer_name=optimizer_name,
            metadata=metadata,
        )


class TorchRegressionMethod(TorchSupervisedMethod):
    """Preset ``TorchSupervisedMethod`` emitting ``predictions`` for ``RegressionTask``."""

    def __init__(
        self,
        *,
        input_key: str = "inputs",
        target_key: str = "targets",
        output_key: str = "predictions",
        objective_capability: str = "loss",
        require_dataset_fields: bool = False,
        use_autocast: bool = True,
        optimizer_name: str | None = None,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        super().__init__(
            input_key=input_key,
            target_key=target_key,
            output_key=output_key,
            objective_capability=objective_capability,
            require_dataset_fields=require_dataset_fields,
            use_autocast=use_autocast,
            optimizer_name=optimizer_name,
            metadata=metadata,
        )
