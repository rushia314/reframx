"""Run record capturing the outcome and history of one Experiment execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from reframx.core.validation.report import ValidationReport


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(slots=True)
class Run:
    """Represents one concrete execution attempt of an Experiment."""

    experiment_name: str
    run_id: str = field(default_factory=_new_run_id)
    status: RunStatus = RunStatus.CREATED
    validation_report: ValidationReport | None = None
    train_history: list[dict[str, Any]] = field(default_factory=list)
    eval_history: list[dict[str, Any]] = field(default_factory=list)
    epoch_history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    def mark_running(self) -> None:
        self.status = RunStatus.RUNNING
        self.started_at = _utc_now_iso()

    def mark_completed(self) -> None:
        self.status = RunStatus.COMPLETED
        self.finished_at = _utc_now_iso()

    def mark_interrupted(self, reason: str = "Interrupted by user") -> None:
        self.status = RunStatus.INTERRUPTED
        self.error = reason
        self.finished_at = _utc_now_iso()

    def mark_failed(self, exc: BaseException) -> None:
        self.status = RunStatus.FAILED
        self.error = f"{type(exc).__name__}: {exc}"
        self.finished_at = _utc_now_iso()
