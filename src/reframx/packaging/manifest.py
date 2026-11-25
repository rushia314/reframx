"""Manifest schemas and environment snapshot for portable ``.rfx`` archives."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ExportProfile = Literal["minimal", "standard", "portable", "full"]
VALID_EXPORT_PROFILES: tuple[ExportProfile, ...] = (
    "minimal",
    "standard",
    "portable",
    "full",
)
RFX_FORMAT_VERSION = "1.0"
RFX_EXTENSION = ".rfx"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Snapshot of the runtime environment used when exporting an experiment."""

    python_version: str
    platform: str
    packages: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": dict(self.packages),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvironmentInfo:
        return cls(
            python_version=str(data.get("python_version", "")),
            platform=str(data.get("platform", "")),
            packages=dict(data.get("packages", {})),
        )


def collect_environment_info(
    extra_packages: tuple[str, ...] = (),
) -> EnvironmentInfo:
    """Capture Python version, OS platform, and versions of key packages."""
    candidates = (
        "reframx",
        "torch",
        "mmengine",
        "numpy",
        "safetensors",
        "transformers",
        *extra_packages,
    )
    found: dict[str, str] = {}
    for pkg in dict.fromkeys(candidates):
        ver = _safe_package_version(pkg)
        if ver is not None:
            found[pkg] = ver
    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=found,
    )


@dataclass(frozen=True, slots=True)
class ComponentRecord:
    """Serializable identity of one component included in an experiment."""

    role: str
    name: str
    component_type: str
    version: str | None = None
    api_version: str | None = None
    source: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "component_type": self.component_type,
            "version": self.version,
            "api_version": self.api_version,
            "source": self.source,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentRecord:
        return cls(
            role=str(data["role"]),
            name=str(data["name"]),
            component_type=str(data.get("component_type", data["role"])),
            version=data.get("version"),
            api_version=data.get("api_version"),
            source=data.get("source"),
            description=data.get("description"),
        )


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """Top-level manifest stored at ``manifest.json`` inside a ``.rfx`` archive."""

    experiment_name: str
    profile: ExportProfile
    framework_version: str
    format_version: str = RFX_FORMAT_VERSION
    created_at: str = field(default_factory=_utc_now_iso)
    components: tuple[ComponentRecord, ...] = ()
    environment: EnvironmentInfo | None = None
    files: Mapping[str, str] = field(default_factory=dict)  # member_path -> sha256
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "framework_version": self.framework_version,
            "experiment_name": self.experiment_name,
            "profile": self.profile,
            "created_at": self.created_at,
            "components": [c.to_dict() for c in self.components],
            "environment": self.environment.to_dict() if self.environment else None,
            "files": dict(self.files),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArchiveManifest:
        profile = data.get("profile", "standard")
        if profile not in VALID_EXPORT_PROFILES:
            raise ValueError(f"Unsupported export profile in manifest: {profile!r}")
        env_raw = data.get("environment")
        return cls(
            format_version=str(data.get("format_version", RFX_FORMAT_VERSION)),
            framework_version=str(data.get("framework_version", "unknown")),
            experiment_name=str(data.get("experiment_name", "Experiment")),
            profile=profile,
            created_at=str(data.get("created_at", "")),
            components=tuple(
                ComponentRecord.from_dict(item)
                for item in data.get("components", ())
            ),
            environment=EnvironmentInfo.from_dict(env_raw) if env_raw else None,
            files=dict(data.get("files", {})),
            metadata=dict(data.get("metadata", {})),
        )
