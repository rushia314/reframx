"""Built-in PyTorch Objectives (losses) and a generic ``nn.Module`` loss adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import CapabilitiesInput
from reframx.integrations.pytorch.utils import require_torch
from reframx.objectives.interface import BaseObjective


class TorchLossAdapter(BaseObjective):
    """Wrap any PyTorch loss module or callable (e.g. ``nn.L1Loss()``)."""

    def __init__(
        self,
        loss_fn: Any,
        *,
        capabilities: CapabilitiesInput = ("loss",),
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if not callable(loss_fn):
            raise TypeError(
                f"TorchLossAdapter expects a callable loss; got {type(loss_fn).__name__}."
            )
        self._loss_fn = loss_fn
        super().__init__(
            capabilities=capabilities,
            metadata=metadata
            or ComponentMetadata(
                name=type(loss_fn).__name__,
                component_type="objective",
            ),
            attributes={"framework": "pytorch", **dict(attributes or {})},
        )

    @property
    def loss_fn(self) -> Any:
        return self._loss_fn

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        return self._loss_fn(*args, **kwargs)


class CrossEntropyObjective(BaseObjective):
    """Built-in wrapper around ``torch.nn.functional.cross_entropy``."""

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        weight: Any | None = None,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self._torch = require_torch()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.weight = weight
        super().__init__(
            capabilities=("loss", "cross_entropy_loss"),
            metadata=metadata
            or ComponentMetadata(
                name="CrossEntropyObjective",
                component_type="objective",
            ),
            attributes={
                "framework": "pytorch",
                "ignore_index": ignore_index,
                "label_smoothing": label_smoothing,
                "reduction": reduction,
            },
        )

    def compute(self, logits: Any, targets: Any) -> Any:
        return self._torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class MSEObjective(BaseObjective):
    """Built-in wrapper around ``torch.nn.functional.mse_loss``."""

    def __init__(
        self,
        *,
        reduction: str = "mean",
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self._torch = require_torch()
        self.reduction = reduction
        super().__init__(
            capabilities=("loss", "mse_loss"),
            metadata=metadata
            or ComponentMetadata(
                name="MSEObjective",
                component_type="objective",
            ),
            attributes={"framework": "pytorch", "reduction": reduction},
        )

    def compute(self, predictions: Any, targets: Any) -> Any:
        return self._torch.nn.functional.mse_loss(
            predictions,
            targets,
            reduction=self.reduction,
        )


class BCEWithLogitsObjective(BaseObjective):
    """Built-in wrapper around ``torch.nn.functional.binary_cross_entropy_with_logits``."""

    def __init__(
        self,
        *,
        reduction: str = "mean",
        pos_weight: Any | None = None,
        metadata: ComponentMetadata | None = None,
    ) -> None:
        self._torch = require_torch()
        self.reduction = reduction
        self.pos_weight = pos_weight
        super().__init__(
            capabilities=("loss", "bce_loss"),
            metadata=metadata
            or ComponentMetadata(
                name="BCEWithLogitsObjective",
                component_type="objective",
            ),
            attributes={"framework": "pytorch", "reduction": reduction},
        )

    def compute(self, logits: Any, targets: Any) -> Any:
        return self._torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )
