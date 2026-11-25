"""Built-in PyTorch Evaluators for classification and regression."""

from __future__ import annotations

import math
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.requirements import Requirement
from reframx.evaluators.interface import BaseEvaluator
from reframx.integrations.pytorch.methods import extract_inputs_and_targets


class TorchAccuracyEvaluator(BaseEvaluator):
    """Computes accuracy for both multiclass (argmax) and binary (sigmoid) logits."""

    def __init__(
        self,
        *,
        output_key: str = "logits",
        input_key: str = "inputs",
        target_key: str = "targets",
        threshold: float = 0.5,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self.output_key = output_key
        self.input_key = input_key
        self.target_key = target_key
        self.threshold = threshold
        self._correct = 0
        self._total = 0
        super().__init__(
            requirements=(Requirement(output_key),),
            metadata=metadata
            or ComponentMetadata(
                name="TorchAccuracyEvaluator",
                component_type="evaluator",
            ),
            attributes={"framework": "pytorch", "output_key": output_key},
        )

    def reset(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, result: Any, batch: Any) -> None:
        _, targets = extract_inputs_and_targets(
            batch,
            input_key=self.input_key,
            target_key=self.target_key,
        )
        logits = result.task_outputs[self.output_key]
        if getattr(logits, "ndim", 1) > 1 and logits.shape[-1] > 1:
            preds = logits.argmax(dim=-1)
        else:
            preds = (logits.sigmoid() >= self.threshold).to(dtype=targets.dtype)
        self._correct += int((preds == targets).sum().item())
        self._total += int(targets.numel())

    def compute(self) -> dict[str, float]:
        return {"accuracy": self._correct / max(self._total, 1)}


class TorchRegressionEvaluator(BaseEvaluator):
    """Computes MSE, RMSE, and MAE for continuous regression predictions."""

    def __init__(
        self,
        *,
        output_key: str = "predictions",
        input_key: str = "inputs",
        target_key: str = "targets",
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self.output_key = output_key
        self.input_key = input_key
        self.target_key = target_key
        self._sum_sq_err = 0.0
        self._sum_abs_err = 0.0
        self._total = 0
        super().__init__(
            requirements=(Requirement(output_key),),
            metadata=metadata
            or ComponentMetadata(
                name="TorchRegressionEvaluator",
                component_type="evaluator",
            ),
            attributes={"framework": "pytorch", "output_key": output_key},
        )

    def reset(self) -> None:
        self._sum_sq_err = 0.0
        self._sum_abs_err = 0.0
        self._total = 0

    def update(self, result: Any, batch: Any) -> None:
        _, targets = extract_inputs_and_targets(
            batch,
            input_key=self.input_key,
            target_key=self.target_key,
        )
        preds = result.task_outputs[self.output_key]
        diff = preds - targets
        self._sum_sq_err += float((diff * diff).sum().item())
        self._sum_abs_err += float(diff.abs().sum().item())
        self._total += int(targets.numel())

    def compute(self) -> dict[str, float]:
        denom = max(self._total, 1)
        mse = self._sum_sq_err / denom
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self._sum_abs_err / denom,
        }
