"""PyTorch-specific lifecycle hooks (CUDA VRAM monitoring & cache management)."""

from __future__ import annotations

from typing import Any

from reframx.execution.hooks.base import BaseHook
from reframx.integrations.pytorch.utils import require_torch

_BYTES_PER_GIB = float(1024**3)


class CudaMemoryHook(BaseHook):
    """Logs CUDA VRAM usage (GiB) and optionally calls ``torch.cuda.empty_cache()``."""

    def __init__(
        self,
        *,
        log_every_steps: int = 1,
        empty_cache_every_steps: int | None = None,
        reset_peak_each_log: bool = False,
    ) -> None:
        if log_every_steps < 1:
            raise ValueError("log_every_steps must be >= 1.")
        if empty_cache_every_steps is not None and empty_cache_every_steps < 1:
            raise ValueError("empty_cache_every_steps must be >= 1.")
        self._torch = require_torch()
        self.log_every_steps = log_every_steps
        self.empty_cache_every_steps = empty_cache_every_steps
        self.reset_peak_each_log = reset_peak_each_log

    def on_step_end(self, ctx: Any, batch: Any, result: Any) -> None:
        if not self._torch.cuda.is_available():
            return

        step = ctx.state.global_step
        if step % self.log_every_steps == 0:
            peak_gib = self._torch.cuda.max_memory_allocated() / _BYTES_PER_GIB
            alloc_gib = self._torch.cuda.memory_allocated() / _BYTES_PER_GIB
            vram_metrics = {
                "peak_vram_gib": peak_gib,
                "allocated_vram_gib": alloc_gib,
            }
            if hasattr(ctx, "step_metrics") and isinstance(ctx.step_metrics, dict):
                ctx.step_metrics.update(vram_metrics)
            if ctx.run is not None and ctx.run.train_history:
                ctx.run.train_history[-1].update(vram_metrics)
            ctx.state.extra["peak_vram_gib"] = peak_gib
            if self.reset_peak_each_log:
                self._torch.cuda.reset_peak_memory_stats()

        if (
            self.empty_cache_every_steps is not None
            and step % self.empty_cache_every_steps == 0
        ):
            self._torch.cuda.empty_cache()
