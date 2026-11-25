"""Tests for portable ``.rfx`` ZIP experiment packaging (export & import)."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

import reframx as rf


class StatefulToyModel(rf.BaseModel):
    def __init__(self, weight: float = 0.0) -> None:
        super().__init__(capabilities=(rf.Capability("forward"),))
        self.weight = weight

    def __call__(self, x: float) -> float:
        return self.weight * x

    def get_component_state(self) -> dict[str, float]:
        return {"weight": self.weight}

    def load_component_state(self, state: dict[str, float]) -> None:
        self.weight = float(state["weight"])


class ToyMethod(rf.BaseMethod):
    def __init__(self) -> None:
        super().__init__(capabilities=(rf.Capability("predictions"),))

    def train_step(
        self,
        batch: dict[str, float],
        ctx: rf.ExecutionContext,
    ) -> rf.MethodResult:
        pred = ctx.model(batch["x"])
        ctx.model.weight += 1.0
        return rf.MethodResult(
            task_outputs={"predictions": pred},
            metrics={"loss": abs(pred - batch["y"])},
        )


def _make_run_experiment() -> tuple[rf.Experiment, StatefulToyModel]:
    model = StatefulToyModel(weight=2.5)
    inst = rf.InstrumentationManager()
    inst.watch_probe("dbg/*", rf.tensor_stats)

    exp = rf.Experiment(
        name="RFX_Demo",
        task=rf.RegressionTask(),
        method=ToyMethod(),
        model=model,
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0, "y": 5.0}]}),
        hooks=[rf.InstrumentationHook(inst)],
        metadata={"author": "researcher"},
    )
    exp.run(max_epochs=2, eval_split=None)
    return exp, model


def test_export_profiles_and_round_trip_state_restoration(tmp_path: Path):
    exp, model = _make_run_experiment()
    assert model.weight == 4.5

    asset_file = tmp_path / "tokenizer.txt"
    asset_file.write_text("vocab_content", encoding="utf-8")
    ckpt_file = tmp_path / "best.ckpt"
    ckpt_file.write_bytes(b"FAKE_CKPT_BYTES")

    # 1. Minimal profile: only manifest + config
    min_path = exp.export(tmp_path / "exp_min", profile="minimal")
    assert min_path.suffix == ".rfx"
    assert zipfile.is_zipfile(min_path)
    min_loaded = rf.load_experiment(min_path)
    assert min_loaded.manifest.profile == "minimal"
    assert min_loaded.config["experiment_name"] == "RFX_Demo"
    assert min_loaded.runs == []
    assert min_loaded.states == {}

    # 2. Standard profile: includes runs + component state + external checkpoints
    std_path = exp.export(
        tmp_path / "exp_std.rfx",
        profile="standard",
        checkpoint_paths=[ckpt_file],
    )
    std_loaded = rf.load_experiment(std_path)
    assert len(std_loaded.runs) == 1
    assert std_loaded.runs[0]["status"] == "completed"
    assert std_loaded.states["model"] == {"weight": 4.5}
    assert "checkpoints/files/best.ckpt" in std_loaded.checkpoint_files

    fresh_model = StatefulToyModel(weight=0.0)
    assert std_loaded.restore_component_state("model", fresh_model) is True
    assert fresh_model.weight == 4.5

    # 3. Full profile: includes assets + run artifacts
    full_path = exp.export(
        tmp_path / "exp_full.rfx",
        profile="full",
        checkpoint_paths=[ckpt_file],
        assets={"tokenizer.txt": asset_file},
    )
    full_loaded = rf.load_experiment(full_path)
    assert "assets/tokenizer.txt" in full_loaded.asset_members
    assert len(full_loaded.artifacts) == 1

    extracted_assets = full_loaded.extract_assets(tmp_path / "out_assets")
    assert len(extracted_assets) == 1
    assert extracted_assets[0].read_text(encoding="utf-8") == "vocab_content"

    extracted_ckpts = full_loaded.extract_checkpoints(tmp_path / "out_ckpts")
    assert len(extracted_ckpts) == 1
    assert extracted_ckpts[0].read_bytes() == b"FAKE_CKPT_BYTES"


def test_rfx_checksum_detects_tampering(tmp_path: Path):
    exp, _ = _make_run_experiment()
    archive_path = exp.export(tmp_path / "tamper.rfx", profile="minimal")

    # Tamper with config.json inside the zip archive.
    tampered = tmp_path / "tampered.rfx"
    with zipfile.ZipFile(archive_path, "r") as zin, zipfile.ZipFile(tampered, "w") as zout:
        for name in zin.namelist():
            if name == "config.json":
                zout.writestr(name, b'{"experiment_name": "HACKED"}')
            else:
                zout.writestr(name, zin.read(name))

    with pytest.raises(rf.ArchiveError, match="Checksum mismatch"):
        rf.load_experiment(tampered)


def test_rfx_pickle_state_requires_explicit_opt_in(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    class BinaryStatefulModel(rf.BaseModel):
        def get_component_state(self) -> dict[str, Any]:
            # bytes is not JSON-serializable -> triggers .pkl fallback when torch is absent
            return {"blob": b"\x01\x02\x03"}

    exp = rf.Experiment(
        name="BinaryState",
        model=BinaryStatefulModel(),
    )
    path = exp.export(tmp_path / "binary.rfx", profile="standard")

    with pytest.raises(rf.ArchiveError, match="allow_pickle=True"):
        rf.load_experiment(path, allow_pickle=False)

    loaded = rf.load_experiment(path, allow_pickle=True)
    assert loaded.states["model"] == {"blob": b"\x01\x02\x03"}


def test_rfx_rejects_zip_slip_and_reports_missing_packages(tmp_path: Path):
    # 1. Zip Slip protection
    evil_zip = tmp_path / "evil.rfx"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../escape.txt", b"bad")
        zf.writestr("manifest.json", json.dumps({"experiment_name": "x", "profile": "minimal"}))

    with pytest.raises(rf.ArchiveError, match="Unsafe archive member path"):
        rf.load_experiment(evil_zip, verify_checksums=False)

    # 2. Missing package warning (never auto-installs)
    warn_zip = tmp_path / "missing_pkg.rfx"
    manifest = {
        "format_version": "1.0",
        "framework_version": "0.1.1",
        "experiment_name": "WarnExp",
        "profile": "minimal",
        "environment": {
            "python_version": "3.12.0",
            "platform": "Linux",
            "packages": {"nonexistent-research-pkg-xyz": "9.9.9"},
        },
        "files": {},
    }
    with zipfile.ZipFile(warn_zip, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))

    imported = rf.load_experiment(warn_zip)
    assert any("nonexistent-research-pkg-xyz" in w for w in imported.warnings)
