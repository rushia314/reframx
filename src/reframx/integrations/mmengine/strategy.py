"""Strategy delegating backward/step/zero_grad to MMEngine's OptimWrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.contracts.capabilities import Capability
from reframx.strategies.interface import BaseStrategy


class MMEngineStrategy(BaseStrategy):
    """Bridges ``ctx.backward()`` / ``ctx.step()`` to an MMEngine ``OptimWrapper``.

    Supports both a single ``OptimWrapper`` and an ``OptimWrapperDict`` (for
    multi-optimizer research methods).
    """

    def __init__(self, *, base_strategy: Any | None = None) -> None:
        self._base_strategy = base_strategy
        self._active_optim_wrapper: Any = None
        super().__init__(
            capabilities=(
                Capability("backward"),
                Capability("optimizer_step"),
                Capability("mmengine_optim_wrapper"),
            ),
            metadata=ComponentMetadata(
                name="MMEngineStrategy",
                component_type="strategy",
            ),
            attributes={"framework": "mmengine"},
        )

    def bind_optim_wrapper(self, optim_wrapper: Any) -> None:
        """Bind the ``OptimWrapper`` (or dict of wrappers) passed by MMEngine."""
        self._active_optim_wrapper = optim_wrapper

    def _resolve_wrapper(self, optimizer: Any) -> Any:
        # When ctx.step(name) resolves an optimizer from ctx.optimizers, it may
        # already be the OptimWrapper itself (when bound by MMEngineBackend).
        if hasattr(optimizer, "backward") and hasattr(optimizer, "step"):
            return optimizer
        if isinstance(self._active_optim_wrapper, Mapping):
            for wrapper in self._active_optim_wrapper.values():
                if getattr(wrapper, "optimizer", None) is optimizer:
                    return wrapper
            if len(self._active_optim_wrapper) == 1:
                return next(iter(self._active_optim_wrapper.values()))
        if self._active_optim_wrapper is not None:
            return self._active_optim_wrapper
        return optimizer

    def prepare_model(self, model: Any) -> Any:
        if self._base_strategy is not None:
            prepare_fn = getattr(self._base_strategy, "prepare_model", None)
            if callable(prepare_fn):
                return prepare_fn(model)
        return model

    def prepare_batch(self, batch: Any, ctx: Any) -> Any:
        if self._base_strategy is not None:
            prepare_fn = getattr(self._base_strategy, "prepare_batch", None)
            if callable(prepare_fn):
                return prepare_fn(batch, ctx)
        return batch

    def backward(self, loss: Any, ctx: Any) -> None:
        wrapper = self._active_optim_wrapper
        if isinstance(wrapper, Mapping):
            # Multi-optimizer dict: fall back to loss.backward() unless user
            # called a specific wrapper directly.
            loss.backward()
            return
        backward_fn = getattr(wrapper, "backward", None)
        if callable(backward_fn):
            backward_fn(loss)
        else:
            loss.backward()

    def optimizer_step(
        self,
        optimizer: Any,
        ctx: Any,
        *,
        zero_grad: bool = True,
    ) -> bool:
        wrapper = self._resolve_wrapper(optimizer)
        should_update = getattr(wrapper, "should_update", None)
        if callable(should_update) and not should_update():
            return False

        wrapper.step()
        if zero_grad:
            self.zero_grad(wrapper, ctx)
        return True

    def zero_grad(self, optimizer: Any, ctx: Any) -> None:
        wrapper = self._resolve_wrapper(optimizer)
        wrapper.zero_grad()
