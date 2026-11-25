"""Common validation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass(frozen = True, slots = True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    component: str | None = None
    location: str | None = None
    details: Mapping[str, Any] = field(default_factory = dict)

    def __post_init__(self) -> None:
        if not self.code or not self.code.strip():
            raise ValueError("ValidationIssue.code must be non-empty.")
        if not self.message or not self.message.strip():
            raise ValueError("ValidationIssue.message must be non-empty.")
        if isinstance(self.severity, str):
            object.__setattr__(self, "severity", ValidationSeverity(self.severity.lower()))
        object.__setattr__(self, "details", dict(self.details))

@dataclass(slots = True)
class ValidationReport:
    """Accumulated validation findings from framework and user validators."""
    issues: list[ValidationIssue] = field(default_factory = list)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.ERROR)
    
    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.WARNING)
    
    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity is ValidationSeverity.INFO)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add(self, issue: ValidationIssue) -> None:
        if not isinstance(issue,ValidationIssue):
            raise TypeError("ValidationReport.add() expects a ValidationIssue.")
        self.issues.append(issue)
    
    def extend(self, other: "ValidationReport") -> None:
        if not isinstance(other, ValidationReport):
            raise TypeError("ValidationReport.extend() expects a ValidationReport.")
        self.issues.extend(other.issues)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the report."""

        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "component": issue.component,
                    "location": issue.location,
                    "details": dict(issue.details),
                }
                for issue in self.issues
            ],
        }