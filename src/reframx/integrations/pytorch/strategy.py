"""PyTorch execution/optimization Strategy (device, AMP, grad clipping, accum)."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Mapping
from typing import Any, Literal

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import Capability
from reframx.integrations.pytorch.utils import (
    move_to_device,
    require_torch,
    resolve_device,
)
from reframx.strategies.interface import StandardBackwardStrategy

PrecisionMode = Literal["fp32", "fp16", "bf16"]


class TorchStrategy(StandardBackwardStrategy):
    """PyTorch-native Strategy handling device placement, AMP, and grad clipping.

    Supports:
    - Automatic or explicit device placement (``"auto"``, ``"cpu"``, ``"cuda"``, ...).
    - Automatic batch transfer via ``prepare_batch`` (disable with
      ``auto_move_batch=False`` for custom batch handling).
    - Mixed precision (``"fp32"``, ``"fp16"``, ``"bf16"``) with ``GradScaler``
      automatically enabled for ``fp16`` on CUDA.
    - Gradient accumulation (``grad_accumulation_steps``) and gradient norm
      clipping (``max_grad_norm``).
    """

    def __init__(
        self,
        *,
        device: str | Any = "auto",
        precision: PrecisionMode = "fp32",
        grad_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        auto_move_batch: bool = True,
        non_blocking: bool = False,
        on_before_step: Callable[[Any, Any], None] | None = None,
        metadata: ComponentMetadata | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(
                f"precision must be 'fp32', 'fp16', or 'bf16'; got {precision!r}."
            )
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when specified.")

        self._torch = require_torch()
        self.device = resolve_device(device)
        self.precision: PrecisionMode = precision
        self.max_grad_norm = max_grad_norm
        self.auto_move_batch = auto_move_batch
        self.non_blocking = non_blocking

        self._use_amp = precision in ("fp16", "bf16")
        self._amp_dtype = (
            self._torch.float16
            if precision == "fp16"
            else (self._torch.bfloat16 if precision == "bf16" else None)
        )
        self._scaler: Any | None = None
        if precision == "fp16" and self.device.type == "cuda":
            self._scaler = self._torch.amp.GradScaler("cuda")

        super().__init__(
            grad_accumulation_steps=grad_accumulation_steps,
            on_before_step=on_before_step,
            metadata=metadata
            or ComponentMetadata(name="TorchStrategy", component_type="strategy"),
            attributes={
                "framework": "pytorch",
                "device": str(self.device),
                "precision": precision,
                "max_grad_norm": max_grad_norm,
                **dict(attributes or {}),
            },
        )
        # Re-export PyTorch-specific capabilities alongside base capabilities.
        self._spec = type(self._spec)(
            metadata=self._spec.metadata,
            requirements=self._spec.requirements,
            capabilities=(
                *self._spec.capabilities,
                Capability("torch_device"),
                Capability("amp" if self._use_amp else "fp32"),
            ),
            attributes=self._spec.attributes,
        )

    @property
    def scaler(self) -> Any | None:
        return self._scaler

    def prepare_model(self, model: Any) -> Any:
        if model is None:
            return None
        target = getattr(model, "module", model)
        to_fn = getattr(target, "to", None)
        if callable(to_fn):
            to_fn(self.device)
        return model

    def prepare_batch(self, batch: Any, ctx: Any) -> Any:
        if not self.auto_move_batch:
            return batch
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)

    @contextlib.contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """Enter PyTorch AMP autocast when ``precision`` is ``fp16`` or ``bf16``."""
        if not self._use_amp:
            yield
            return
        with self._torch.amp.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
        ):
            yield

    @contextlib.contextmanager
    def eval_context(self) -> Generator[None, None, None]:
        """Use ``torch.inference_mode()`` (and AMP if enabled) during evaluation."""
        with self._torch.inference_mode(), self.autocast_context():
            yield

    def backward(self, loss: Any, ctx: Any) -> None:
        scaled_loss = (
            loss / self.grad_accumulation_steps
            if self.grad_accumulation_steps > 1
            else loss
        )
        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def _clip_gradients(self, optimizer: Any) -> None:
        if self.max_grad_norm is None:
            return
        params: list[Any] = []
        for group in getattr(optimizer, "param_groups", ()):
            for param in group.get("params", ()):
                if getattr(param, "grad", None) is not None:
                    params.append(param)
        if params:
            self._torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    def optimizer_step(
        self,
        optimizer: Any,
        ctx: Any,
        *,
        zero_grad: bool = True,
    ) -> bool:
        opt_id = id(optimizer)
        count = self._accum_counts.get(opt_id, 0) + 1
        self._accum_counts[opt_id] = count

        if count % self.grad_accumulation_steps != 0:
            return False

        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

        self._clip_gradients(optimizer)

        if self.on_before_step is not None:
            self.on_before_step(optimizer, ctx)

        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

        if zero_grad:
            self.zero_grad(optimizer, ctx)
        return True

    def zero_grad(self, optimizer: Any, ctx: Any) -> None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
