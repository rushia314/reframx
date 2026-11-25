"""Experiment composition, validation, and execution entry point."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from reframx.core.components.handle import ComponentHandle
from reframx.core.contracts.compatibility import validate_requirements
from reframx.core.exceptions import ValidationFailedError
from reframx.core.validation.report import ValidationReport
from reframx.core.validation.runner import run_component_validator
from reframx.execution.backend import ExecutionBackend, NativeBackend
from reframx.execution.context import ExecutionContext
from reframx.experiments.run import Run
from reframx.strategies.interface import StandardBackwardStrategy


def _normalize_named_mapping(
    value: Any | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {"default": value}


class Experiment:
    """Composes research components, validates contracts, and executes Runs."""

    def __init__(
        self,
        *,
        task: Any = None,
        method: Any = None,
        model: Any = None,
        dataset: Any = None,
        objectives: Any | Mapping[str, Any] | None = None,
        optimizers: Any | Mapping[str, Any] | None = None,
        evaluator: Any = None,
        strategy: Any = None,
        backend: ExecutionBackend | None = None,
        hooks: Iterable[Any] = (),
        name: str = "Experiment",
        strict_contract: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.task = task
        self.method = method
        self.model = model
        self.dataset = dataset
        self.objectives: dict[str, Any] = _normalize_named_mapping(objectives)
        self.optimizers: dict[str, Any] = _normalize_named_mapping(optimizers)
        self.evaluator = evaluator
        self.strategy = strategy if strategy is not None else StandardBackwardStrategy()
        self.backend: ExecutionBackend = backend if backend is not None else NativeBackend()
        self.hooks: list[Any] = list(hooks)
        self.strict_contract = strict_contract
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.runs: list[Run] = []
        self._active_ctx: ExecutionContext | None = None

    def stop(self, reason: str = "Stopped by user") -> bool:
        """Request the currently running training loop to stop gracefully.

        Safe to call from a Hook, Method, or another thread. Returns ``True``
        if an active run was signaled, or ``False`` if no run is active.
        """
        ctx = self._active_ctx
        if ctx is None:
            return False
        ctx.state.extra["_stop_reason"] = reason
        ctx.request_stop()
        return True

    def get_component_handles(self) -> list[ComponentHandle]:
        """Wrap all attached components into non-invasive ``ComponentHandle``s."""
        handles: list[ComponentHandle] = []
        singletons = (
            ("task", self.task),
            ("method", self.method),
            ("model", self.model),
            ("dataset", self.dataset),
            ("evaluator", self.evaluator),
            ("strategy", self.strategy),
        )
        for role, component in singletons:
            if component is not None:
                handles.append(ComponentHandle.wrap(component, role=role))

        for objective in self.objectives.values():
            handles.append(ComponentHandle.wrap(objective, role="objective"))

        return handles

    def validate(self) -> ValidationReport:
        """Validate all declared requirements/capabilities and custom validators."""
        handles = self.get_component_handles()
        report = ValidationReport()

        for index, handle in enumerate(handles):
            provider_specs = [
                other.spec for i, other in enumerate(handles) if i != index
            ]
            report.extend(
                validate_requirements(
                    consumer=handle.spec,
                    providers=provider_specs,
                    strict_contract=self.strict_contract,
                )
            )

        peers: dict[str, Any] = {
            "task": self.task,
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "objectives": dict(self.objectives),
            "optimizers": dict(self.optimizers),
            "evaluator": self.evaluator,
            "strategy": self.strategy,
        }
        for handle in handles:
            report.extend(
                run_component_validator(handle, peers=peers, experiment=self)
            )

        return report

    def _build_context(self, run: Run | None = None) -> ExecutionContext:
        return ExecutionContext(
            model=self.model,
            strategy=self.strategy,
            task=self.task,
            method=self.method,
            dataset=self.dataset,
            objectives=dict(self.objectives),
            optimizers=dict(self.optimizers),
            evaluator=self.evaluator,
            run=run,
            hooks=list(self.hooks),
        )

    def run(
        self,
        *,
        max_epochs: int = 1,
        max_steps: int | None = None,
        train_split: str = "train",
        eval_split: str | None = "val",
        eval_every_epochs: int = 1,
        validate: bool = True,
        catch_interrupt: bool = False,
    ) -> Run:
        """Validate (optionally) and execute a training/evaluation Run."""
        validation_report: ValidationReport | None = None
        if validate:
            validation_report = self.validate()
            if not validation_report.is_valid:
                error_summary = "; ".join(
                    f"[{issue.code}] {issue.message}"
                    for issue in validation_report.errors
                )
                raise ValidationFailedError(
                    f"Experiment '{self.name}' failed validation: {error_summary}",
                    report=validation_report,
                )

        run = Run(
            experiment_name=self.name,
            validation_report=validation_report,
            metadata=dict(self.metadata),
        )
        self.runs.append(run)

        ctx = self._build_context(run=run)
        self._active_ctx = ctx
        try:
            return self.backend.run(
                ctx,
                max_epochs=max_epochs,
                max_steps=max_steps,
                train_split=train_split,
                eval_split=eval_split,
                eval_every_epochs=eval_every_epochs,
                catch_interrupt=catch_interrupt,
            )
        finally:
            self._active_ctx = None

    def evaluate(
        self,
        *,
        split: str = "val",
        validate: bool = False,
    ) -> dict[str, Any]:
        """Run an evaluation-only pass without training."""
        if validate:
            report = self.validate()
            if not report.is_valid:
                raise ValidationFailedError(
                    f"Experiment '{self.name}' failed validation.",
                    report=report,
                )
        ctx = self._build_context(run=None)
        return self.backend.evaluate(ctx, split=split)

    def export(
        self,
        destination: str | Path,
        *,
        profile: Any = "standard",
        checkpoint_paths: Iterable[str | Path] = (),
        assets: Mapping[str, str | Path] | None = None,
        include_optimizer_state: bool = True,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Export this experiment snapshot to a portable ``.rfx`` ZIP archive."""
        from reframx.packaging.exporter import export_experiment

        return export_experiment(
            self,
            destination,
            profile=profile,
            checkpoint_paths=checkpoint_paths,
            assets=assets,
            include_optimizer_state=include_optimizer_state,
            extra_metadata=extra_metadata,
        )
