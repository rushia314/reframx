"""Unit and integration tests for ComponentRegistry and PluginManager."""

from __future__ import annotations

import importlib.metadata
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import reframx as rf


def test_registry_register_create_overwrite_and_unregister():
    reg = rf.ComponentRegistry()

    @reg.register("objective", "custom_l1", version="1.0.0", tags=("loss",))
    class CustomL1(rf.BaseObjective):
        def __init__(self, scale: float = 1.0) -> None:
            super().__init__(capabilities=(rf.Capability("loss"),))
            self.scale = scale

        def compute(self, pred: float, target: float) -> float:
            return self.scale * abs(pred - target)

    assert reg.has("objective", "custom_l1")
    assert reg.categories() == ("objective",)

    obj = reg.create("objective", "custom_l1", scale=2.0)
    assert obj(5.0, 2.0) == 6.0

    with pytest.raises(ValueError, match="already registered"):
        reg.register_factory("objective", "custom_l1", CustomL1)

    reg.register_factory("objective", "custom_l1", CustomL1, overwrite=True)
    reg.unregister("objective", "custom_l1")
    assert not reg.has("objective", "custom_l1")


def test_plugin_load_unload_and_reload_lifecycle():
    pm = rf.PluginManager()
    unregistered_called: list[str] = []

    class MyResearchPlugin:
        manifest = rf.PluginManifest(
            name="my_research_plugin",
            version="0.2.0",
            categories=("method", "objective"),
        )

        def register(self, registry: rf.ComponentRegistry) -> None:
            @registry.register("method", "toy_method")
            class ToyMethod(rf.BaseMethod):
                def __init__(self) -> None:
                    super().__init__(capabilities=(rf.Capability("predictions"),))

                def train_step(self, batch: Any, ctx: Any) -> rf.MethodResult:
                    return rf.MethodResult(
                        task_outputs={"predictions": batch["x"]},
                        metrics={"loss": 0.0},
                    )

            registry.register_factory("objective", "zero_loss", lambda p, y: 0.0)

        def unregister(self, registry: rf.ComponentRegistry) -> None:
            unregistered_called.append("called")

    loaded = pm.load_plugin(MyResearchPlugin)
    assert pm.is_loaded("my_research_plugin")
    assert set(loaded.registered_entries) == {
        ("method", "toy_method"),
        ("objective", "zero_loss"),
    }
    assert pm.registry.has("method", "toy_method")
    assert pm.registry.has("objective", "zero_loss")

    # Run a real Experiment using components created from the plugin.
    exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=pm.registry.create("method", "toy_method"),
        dataset=rf.BaseDataset(splits_data={"train": [{"x": 1.0}]}),
    )
    run = exp.run(max_epochs=1, eval_split=None)
    assert run.status is rf.RunStatus.COMPLETED

    # Reload unloads previous entries cleanly before re-registering.
    pm.load_plugin(MyResearchPlugin, reload=True)
    assert len(unregistered_called) == 1
    assert pm.registry.has("method", "toy_method")

    # Unload calls unregister() and removes all registered components.
    assert pm.unload_plugin("my_research_plugin") is True
    assert not pm.is_loaded("my_research_plugin")
    assert not pm.registry.has("method", "toy_method")
    assert not pm.registry.has("objective", "zero_loss")
    assert len(unregistered_called) == 2

    assert pm.unload_plugin("my_research_plugin", missing_ok=True) is False
    with pytest.raises(KeyError):
        pm.unload_plugin("my_research_plugin")


def test_plugin_rollback_on_registration_failure():
    pm = rf.PluginManager()

    def broken_plugin(registry: rf.ComponentRegistry) -> None:
        registry.register_factory("model", "partial_model", object)
        raise RuntimeError("boom during register")

    with pytest.raises(rf.PluginLoadError, match="boom during register"):
        pm.load_plugin(broken_plugin, name="broken")

    assert not pm.is_loaded("broken")
    assert not pm.registry.has("model", "partial_model")


def test_load_and_unload_standalone_file_plugin(tmp_path: Path):
    plugin_file = tmp_path / "custom_ext.py"
    plugin_file.write_text(
        "import reframx as rf\n"
        "PLUGIN_MANIFEST = rf.PluginManifest(name='file_ext', version='1.0')\n"
        "def register(registry):\n"
        "    registry.register_factory('hook', 'noop_hook', rf.BaseHook)\n",
        encoding="utf-8",
    )

    pm = rf.PluginManager()
    loaded = pm.load_file(plugin_file)
    assert loaded.manifest.name == "file_ext"
    assert pm.registry.has("hook", "noop_hook")

    unloaded = pm.unload_all()
    assert unloaded == ("file_ext",)
    assert not pm.registry.has("hook", "noop_hook")


def test_discover_entry_points_does_not_execute_until_explicitly_loaded(monkeypatch):
    executed = False

    def fake_plugin_register(registry: rf.ComponentRegistry) -> None:
        nonlocal executed
        executed = True
        registry.register_factory("strategy", "ep_strategy", rf.StandardBackwardStrategy)

    class FakeEntryPoint:
        name = "demo_ep"
        value = "demo_pkg:register"
        dist = types.SimpleNamespace(name="demo-pkg", version="0.1.0")

        def load(self) -> Any:
            return fake_plugin_register

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda *, group: [FakeEntryPoint()] if group == rf.DEFAULT_ENTRY_POINT_GROUP else [],
    )

    pm = rf.PluginManager()
    discovered = pm.discover_entry_points()
    assert len(discovered) == 1
    assert discovered[0].name == "demo_ep"
    assert discovered[0].dist_name == "demo-pkg"
    assert executed is False  # Discovery must not execute plugin code!

    loaded = pm.load_entry_points(allowlist={"demo_ep"})
    assert len(loaded) == 1
    assert executed is True
    assert pm.registry.has("strategy", "ep_strategy")

    pm.unload_plugin("demo_ep")
    assert not pm.registry.has("strategy", "ep_strategy")


def test_load_path_multi_file_package_with_relative_imports_and_unload(tmp_path: Path):
    pkg_dir = tmp_path / "my_multi_plugin"
    pkg_dir.mkdir()
    (pkg_dir / "helpers.py").write_text(
        "def make_bias(x: float) -> float:\n"
        "    return x + 10.0\n",
        encoding="utf-8",
    )
    (pkg_dir / "__init__.py").write_text(
        "import reframx as rf\n"
        "from .helpers import make_bias\n"
        "PLUGIN_MANIFEST = rf.PluginManifest(name='multi_pkg', version='1.0.0')\n"
        "def register(registry):\n"
        "    registry.register_factory('objective', 'biased', make_bias)\n",
        encoding="utf-8",
    )

    pm = rf.PluginManager()
    loaded = pm.load_path(pkg_dir)
    assert loaded.manifest.name == "multi_pkg"
    assert pm.registry.create("objective", "biased", 5.0) == 15.0
    assert any(k.startswith("rf_plugin_pkg_my_multi_plugin") for k in sys.modules)

    pm.unload_plugin("multi_pkg")
    assert not pm.registry.has("objective", "biased")
    assert not any(k.startswith("rf_plugin_pkg_my_multi_plugin") for k in sys.modules)


def test_load_directory_loads_both_files_and_package_plugins(tmp_path: Path):
    plugins_root = tmp_path / "plugins_folder"
    plugins_root.mkdir()

    # 1. Standalone .py plugin file
    (plugins_root / "alpha.py").write_text(
        "import reframx as rf\n"
        "PLUGIN_MANIFEST = rf.PluginManifest(name='alpha')\n"
        "def register(registry):\n"
        "    registry.register_factory('hook', 'alpha_hook', rf.BaseHook)\n",
        encoding="utf-8",
    )

    # 2. Multi-file package plugin
    beta_dir = plugins_root / "beta_pkg"
    beta_dir.mkdir()
    (beta_dir / "core.py").write_text("VALUE = 42\n", encoding="utf-8")
    (beta_dir / "__init__.py").write_text(
        "import reframx as rf\n"
        "from .core import VALUE\n"
        "PLUGIN_MANIFEST = rf.PluginManifest(name='beta')\n"
        "def register(registry):\n"
        "    registry.register_factory('model', 'beta_val', lambda: VALUE)\n",
        encoding="utf-8",
    )

    # Ignored private file
    (plugins_root / "_ignored.py").write_text("raise RuntimeError('should not load')\n")

    pm = rf.PluginManager()
    loaded = pm.load_directory(plugins_root)
    assert {p.manifest.name for p in loaded} == {"alpha", "beta"}
    assert pm.registry.has("hook", "alpha_hook")
    assert pm.registry.create("model", "beta_val") == 42

    pm.unload_all()
    assert not pm.registry.has("hook", "alpha_hook")
    assert not pm.registry.has("model", "beta_val")
