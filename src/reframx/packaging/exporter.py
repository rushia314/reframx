"""Export an ``Experiment`` snapshot into a portable ``.rfx`` ZIP archive."""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from reframx.core.components.specification import ComponentSpec
from reframx.packaging.archive import RFXArchiveWriter
from reframx.packaging.manifest import (
    VALID_EXPORT_PROFILES,
    ArchiveManifest,
    ComponentRecord,
    ExportProfile,
    collect_environment_info,
)


def _serialize_json_safe(value: Any) -> Any:
    """Convert nested specs/dataclasses/objects into JSON-serializable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _serialize_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize_json_safe(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_json_safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _serialize_json_safe(to_dict())
    return repr(value)


def _serialize_spec(spec: ComponentSpec) -> dict[str, Any]:
    return {
        "metadata": _serialize_json_safe(spec.metadata),
        "requirements": _serialize_json_safe(spec.requirements),
        "capabilities": _serialize_json_safe(spec.capabilities),
        "attributes": _serialize_json_safe(spec.attributes),
    }


def _extract_component_state(component: Any) -> Any | None:
    get_state = getattr(component, "get_component_state", None)
    if callable(get_state):
        return get_state()
    state_dict = getattr(component, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return None


def _stage_state_entry(
    writer: RFXArchiveWriter,
    member_prefix: str,
    state: Any,
    *,
    scratch_dir: Path,
) -> None:
    """Serialize component/optimizer state without buffering multi-GB weights in RAM.

    Prefers JSON for small pure-JSON dicts, streams ``torch.save`` via a temp
    file on disk when PyTorch is active, and falls back to streamed pickle.
    """
    try:
        raw_json = json.dumps(state, sort_keys=True, ensure_ascii=False).encode("utf-8")
        writer.add_bytes(f"{member_prefix}.json", raw_json)
        return
    except TypeError:
        pass

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and callable(getattr(torch_mod, "save", None)):
        tmp_file = scratch_dir / f"{Path(member_prefix).name}.pt"
        torch_mod.save(state, tmp_file)
        writer.add_file(f"{member_prefix}.pt", tmp_file, delete_after_write=True)
        return

    tmp_file = scratch_dir / f"{Path(member_prefix).name}.pkl"
    with tmp_file.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    writer.add_file(f"{member_prefix}.pkl", tmp_file, delete_after_write=True)


def _serialize_run(run: Any) -> dict[str, Any]:
    report = getattr(run, "validation_report", None)
    return {
        "run_id": run.run_id,
        "experiment_name": run.experiment_name,
        "status": run.status.value if hasattr(run.status, "value") else str(run.status),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
        "metadata": _serialize_json_safe(run.metadata),
        "validation_report": report.to_dict() if report is not None else None,
        "train_history": _serialize_json_safe(run.train_history),
        "eval_history": _serialize_json_safe(run.eval_history),
        "epoch_history": _serialize_json_safe(run.epoch_history),
    }


def export_experiment(
    experiment: Any,
    destination: str | Path,
    *,
    profile: ExportProfile = "standard",
    checkpoint_paths: Iterable[str | Path] = (),
    assets: Mapping[str, str | Path] | None = None,
    include_optimizer_state: bool = True,
    base_model_ref: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export an ``Experiment`` snapshot into a single portable ``.rfx`` ZIP archive.

    Export profiles
    ---------------
    - ``"minimal"``: manifest + resolved component specs & config.
    - ``"standard"``: minimal + run history + component/optimizer states & checkpoints.
    - ``"portable"``: standard + user-provided ``assets`` (tokenizers, configs, ...).
    - ``"full"``: portable + run artifacts (instrumentation records, logs, ...).
    """
    if profile not in VALID_EXPORT_PROFILES:
        raise ValueError(
            f"Unsupported export profile {profile!r}; "
            f"expected one of {VALID_EXPORT_PROFILES}."
        )

    from reframx import __version__ as framework_version

    writer = RFXArchiveWriter()
    handles = experiment.get_component_handles()

    component_records: list[ComponentRecord] = []
    component_specs: dict[str, Any] = {}
    for idx, handle in enumerate(handles):
        meta = handle.spec.metadata
        component_records.append(
            ComponentRecord(
                role=handle.role,
                name=meta.name,
                component_type=meta.component_type,
                version=meta.version,
                api_version=meta.api_version,
                source=meta.source,
                description=meta.description,
            )
        )
        key = handle.role if handle.role not in component_specs else f"{handle.role}_{idx}"
        component_specs[key] = _serialize_spec(handle.spec)

    # 1. Config & resolved component specs (included in ALL profiles).
    config_payload = {
        "experiment_name": experiment.name,
        "strict_contract": experiment.strict_contract,
        "metadata": _serialize_json_safe(experiment.metadata),
        "components": component_specs,
    }
    writer.add_json("config.json", config_payload)

    # 2. Standard+: runs + streamed component/optimizer states + external checkpoints.
    with tempfile.TemporaryDirectory(prefix="reframx_stage_") as scratch:
        scratch_dir = Path(scratch)
        if profile in ("standard", "portable", "full"):
            runs_payload = [_serialize_run(r) for r in getattr(experiment, "runs", ())]
            writer.add_json("runs.json", runs_payload)

            for handle in handles:
                state = _extract_component_state(handle.instance)
                if state is not None:
                    _stage_state_entry(
                        writer,
                        f"checkpoints/{handle.role}",
                        state,
                        scratch_dir=scratch_dir,
                    )

            if include_optimizer_state:
                for opt_name, opt in getattr(experiment, "optimizers", {}).items():
                    state = _extract_component_state(opt)
                    if state is not None:
                        _stage_state_entry(
                            writer,
                            f"checkpoints/optimizer_{opt_name}",
                            state,
                            scratch_dir=scratch_dir,
                        )

            for ckpt_path in checkpoint_paths:
                src = Path(ckpt_path)
                writer.add_file(f"checkpoints/files/{src.name}", src)

        # 3. Portable+: user assets (files or directories).
        if profile in ("portable", "full") and assets:
            for asset_name, asset_path in assets.items():
                src = Path(asset_path)
                if src.is_dir():
                    writer.add_directory(f"assets/{asset_name}", src)
                else:
                    writer.add_file(f"assets/{asset_name}", src)

        # 4. Full: run artifacts (instrumentation records, custom artifacts).
        if profile == "full":
            all_artifacts: dict[str, Any] = {}
            for run in getattr(experiment, "runs", ()):
                if run.artifacts:
                    all_artifacts[run.run_id] = _serialize_json_safe(run.artifacts)
            if all_artifacts:
                writer.add_json("artifacts/run_artifacts.json", all_artifacts)

        manifest_meta = _serialize_json_safe(extra_metadata or {})
        if base_model_ref is not None:
            manifest_meta["base_model_ref"] = base_model_ref

        # Finalize manifest with SHA-256 inventory of every written member.
        manifest = ArchiveManifest(
            experiment_name=experiment.name,
            profile=profile,
            framework_version=framework_version,
            components=tuple(component_records),
            environment=collect_environment_info(),
            files=writer.checksums,
            metadata=manifest_meta,
        )
        writer.add_json("manifest.json", manifest.to_dict())
        return writer.write(destination)
