"""PyTorch execution/optimization Strategy (device, AMP, grad clipping, schedulers)."""

from __future__ import annotations

import contextlib
import math
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
SchedulerStepOn = Literal["step", "epoch"]


def _normalize_schedulers(value: Any | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"default": value}


class TorchStrategy(StandardBackwardStrategy):
    """PyTorch-native Strategy handling device placement, AMP, grad clipping, and schedulers.

    Supports:
    - Automatic or explicit device placement (``"auto"``, ``"cpu"``, ``"cuda"``, ...).
    - ``auto_move_model=False`` for quantized (bitsandbytes 4-bit/8-bit),
      ``device_map="auto"``, or mixed-dtype models that must not have ``.to(device)`` called.
    - Automatic batch transfer via ``prepare_batch`` (disable with ``auto_move_batch=False``).
    - Mixed precision (``"fp32"``, ``"fp16"``, ``"bf16"``) with ``GradScaler``
      automatically enabled for ``fp16`` on CUDA.
    - Gradient accumulation (``grad_accumulation_steps``), gradient norm clipping
      (``max_grad_norm``, automatically logging ``grad_norm``), and
      ``skip_nonfinite_grads=True`` to skip optimizer updates on NaN/Inf gradients.
    - Optional LR ``schedulers`` stepped per optimizer step or per epoch.
    """

    def __init__(
        self,
        *,
        device: str | Any = "auto",
        precision: PrecisionMode = "fp32",
        grad_accumulation_steps: int = 1,
        max_grad_norm: float | None = None,
        skip_nonfinite_grads: bool = False,
        auto_move_model: bool = True,
        auto_move_batch: bool = True,
        non_blocking: bool = False,
        schedulers: Any | Mapping[str, Any] | None = None,
        scheduler_step_on: SchedulerStepOn = "step",
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
        if scheduler_step_on not in ("step", "epoch"):
            raise ValueError("scheduler_step_on must be 'step' or 'epoch'.")

        self._torch = require_torch()
        self.device = resolve_device(device)
        self.precision: PrecisionMode = precision
        self.max_grad_norm = max_grad_norm
        self.skip_nonfinite_grads = skip_nonfinite_grads
        self.auto_move_model = auto_move_model
        self.auto_move_batch = auto_move_batch
        self.non_blocking = non_blocking
        self.schedulers: dict[str, Any] = _normalize_schedulers(schedulers)
        self.scheduler_step_on: SchedulerStepOn = scheduler_step_on

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
                "skip_nonfinite_grads": skip_nonfinite_grads,
                "auto_move_model": auto_move_model,
                **dict(attributes or {}),
            },
        )
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
        if model is None or not self.auto_move_model:
            return model
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

    def _collect_grad_params(self, optimizer: Any) -> list[Any]:
        params: list[Any] = []
        for group in getattr(optimizer, "param_groups", ()):
            for param in group.get("params", ()):
                if getattr(param, "grad", None) is not None:
                    params.append(param)
        return params

    def _has_nonfinite_grads(self, params: list[Any]) -> bool:
        isfinite = getattr(self._torch, "isfinite", None)
        for param in params:
            grad = param.grad
            if callable(isfinite) and isinstance(grad, self._torch.Tensor):
                if not bool(isfinite(grad).all().item()):
                    return True
            else:
                item = getattr(grad, "item", None)
                if callable(item) and not math.isfinite(float(item())):
                    return True
        return False

    def _clip_gradients(self, params: list[Any], ctx: Any) -> float | None:
        if self.max_grad_norm is None or not params:
            return None
        raw_norm = self._torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        norm = float(raw_norm.item() if hasattr(raw_norm, "item") else raw_norm)
        if ctx is not None:
            log_fn = getattr(ctx, "log_metric", None)
            if callable(log_fn):
                log_fn("grad_norm", norm)
            state = getattr(ctx, "state", None)
            if state is not None:
                state.extra["last_grad_norm"] = norm
        return norm

    def _step_matching_schedulers(self, optimizer: Any) -> None:
        if not self.schedulers:
            return
        matched = [
            sched
            for sched in self.schedulers.values()
            if getattr(sched, "optimizer", None) is optimizer
        ]
        targets = matched if matched else (
            [self.schedulers["default"]]
            if "default" in self.schedulers
            else list(self.schedulers.values())
        )
        for sched in targets:
            step_fn = getattr(sched, "step", None)
            if callable(step_fn):
                step_fn()

    def on_epoch_end(self, ctx: Any, epoch_summary: Mapping[str, Any]) -> None:
        """Step epoch-level LR schedulers when ``scheduler_step_on == 'epoch'``."""
        if self.scheduler_step_on != "epoch" or not self.schedulers:
            return
        for sched in self.schedulers.values():
            step_fn = getattr(sched, "step", None)
            if callable(step_fn):
                step_fn()

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

        params = self._collect_grad_params(optimizer)
        grad_norm = self._clip_gradients(params, ctx)

        if self.skip_nonfinite_grads:
            nonfinite = (
                grad_norm is not None and not math.isfinite(grad_norm)
            ) or self._has_nonfinite_grads(params)
            if nonfinite:
                if ctx is not None and callable(getattr(ctx, "log_metric", None)):
                    ctx.log_metric("skipped_nonfinite_step", 1)
                if zero_grad:
                    self.zero_grad(optimizer, ctx)
                if self._scaler is not None:
                    self._scaler.update()
                return False

        if self.on_before_step is not None:
            self.on_before_step(optimizer, ctx)

        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

        if self.scheduler_step_on == "step":
            self._step_matching_schedulers(optimizer)

        if zero_grad:
            self.zero_grad(optimizer, ctx)
        return True

    def zero_grad(self, optimizer: Any, ctx: Any) -> None:
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
