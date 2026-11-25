"""Safe importer for portable ``.rfx`` experiment archives."""

from __future__ import annotations

import importlib.metadata
import io
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reframx.packaging.archive import ArchiveError, RFXArchiveReader
from reframx.packaging.manifest import ArchiveManifest


def _check_environment_compatibility(manifest: ArchiveManifest) -> list[str]:
    """Compare archive environment packages against active env without installing."""
    warnings: list[str] = []
    if manifest.environment is None:
        return warnings

    for pkg, expected_ver in manifest.environment.packages.items():
        try:
            installed_ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            warnings.append(
                f"Package '{pkg}=={expected_ver}' recorded in archive is not "
                "installed in the current environment."
            )
            continue
        if installed_ver != expected_ver:
            warnings.append(
                f"Package '{pkg}' version differs: archive has {expected_ver}, "
                f"current environment has {installed_ver}."
            )
    return warnings


def _decode_state_payload(
    member_name: str,
    raw: bytes,
    *,
    allow_pickle: bool,
) -> Any:
    if member_name.endswith(".json"):
        return json.loads(raw.decode("utf-8"))

    if member_name.endswith(".pt"):
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            try:
                import importlib

                torch_mod = importlib.import_module("torch")
            except ImportError as exc:
                raise ArchiveError(
                    f"State '{member_name}' requires PyTorch to load."
                ) from exc
        buf = io.BytesIO(raw)
        try:
            return torch_mod.load(buf, weights_only=not allow_pickle)
        except TypeError:
            if not allow_pickle:
                raise ArchiveError(
                    f"Refusing to load '{member_name}' without weights_only support; "
                    "pass allow_pickle=True if you trust this archive."
                ) from None
            return torch_mod.load(buf)

    if member_name.endswith(".pkl"):
        if not allow_pickle:
            raise ArchiveError(
                f"State '{member_name}' uses pickle serialization. Pass "
                "allow_pickle=True only if you trust the source of this .rfx archive."
            )
        return pickle.loads(raw)

    return raw


@dataclass(slots=True)
class ImportedExperiment:
    """Loaded contents of an ``.rfx`` archive (never executes untrusted code)."""

    path: Path
    manifest: ArchiveManifest
    config: dict[str, Any]
    runs: list[dict[str, Any]] = field(default_factory=list)
    states: dict[str, Any] = field(default_factory=dict)
    checkpoint_files: tuple[str, ...] = ()
    asset_members: tuple[str, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def extract_assets(self, destination_dir: str | Path) -> list[Path]:
        """Extract all ``assets/`` entries into ``destination_dir``."""
        return RFXArchiveReader(self.path).extract_prefix("assets", destination_dir)

    def extract_checkpoints(self, destination_dir: str | Path) -> list[Path]:
        """Extract all ``checkpoints/files/`` entries into ``destination_dir``."""
        return RFXArchiveReader(self.path).extract_prefix(
            "checkpoints/files", destination_dir
        )

    def restore_component_state(self, role: str, component: Any) -> bool:
        """Restore saved state for ``role`` into ``component`` if present."""
        if role not in self.states:
            return False
        state = self.states[role]
        load_state = getattr(component, "load_component_state", None)
        if callable(load_state):
            load_state(state)
            return True
        load_sd = getattr(component, "load_state_dict", None)
        if callable(load_sd):
            load_sd(state)
            return True
        raise TypeError(
            f"Component '{role}' ({type(component).__name__}) does not define "
            "load_component_state() or load_state_dict()."
        )


def load_experiment(
    path: str | Path,
    *,
    verify_checksums: bool = True,
    load_states: bool = True,
    allow_pickle: bool = False,
) -> ImportedExperiment:
    """Safely load an ``.rfx`` archive without installing deps or executing plugins."""
    reader = RFXArchiveReader(path)
    manifest_raw = reader.read_json("manifest.json")
    manifest = ArchiveManifest.from_dict(manifest_raw)

    if verify_checksums:
        mismatches = reader.verify_checksums(manifest.files)
        if mismatches:
            raise ArchiveError(
                "Archive integrity verification failed: " + "; ".join(mismatches)
            )

    config = reader.read_json("config.json") if reader.has_member("config.json") else {}
    runs = reader.read_json("runs.json") if reader.has_member("runs.json") else []
    artifacts = (
        reader.read_json("artifacts/run_artifacts.json")
        if reader.has_member("artifacts/run_artifacts.json")
        else {}
    )

    members = reader.list_members()
    checkpoint_files = tuple(
        m for m in members if m.startswith("checkpoints/files/")
    )
    asset_members = tuple(m for m in members if m.startswith("assets/"))

    states: dict[str, Any] = {}
    warnings = _check_environment_compatibility(manifest)

    if load_states:
        for member in members:
            if not member.startswith("checkpoints/") or member.startswith(
                "checkpoints/files/"
            ):
                continue
            stem = Path(member).stem
            raw = reader.read_bytes(member)
            states[stem] = _decode_state_payload(
                member,
                raw,
                allow_pickle=allow_pickle,
            )

    return ImportedExperiment(
        path=Path(path),
        manifest=manifest,
        config=config,
        runs=runs,
        states=states,
        checkpoint_files=checkpoint_files,
        asset_members=asset_members,
        artifacts=artifacts,
        warnings=warnings,
    )
