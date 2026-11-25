# Developer Guide: Writing Plugins

This guide explains how to author, structure, distribute, load, reload, and unload plugins for **Research Framework**.

---

## 1. How the Plugin System Works

The plugin system has two core pieces in `reframx.core.registry`:

1. **`ComponentRegistry`**: Stores component classes and factory functions indexed by `(category, name)`.
   - Categories are **open strings** (`"task"`, `"method"`, `"model"`, `"dataset"`, `"objective"`, `"evaluator"`, `"strategy"`, `"hook"`, `"backend"`, or any custom category name).
2. **`PluginManager`**: Discovers and explicitly loads plugins into a `ComponentRegistry`.
   - Every plugin receives a scoped registry proxy during `register(registry)`.
   - `PluginManager` automatically tracks every `(category, name)` registered by that plugin.
   - If `register(registry)` raises an exception halfway through, all partially registered components are **rolled back automatically**.
   - When `unload_plugin(name)` is called, `PluginManager` invokes the plugin's optional `unregister(registry)` hook, removes all tracked entries from the registry, and cleans up dynamically loaded `sys.modules`.

---

## 2. Plugin Formats

### Format A: Single `.py` File Plugin (Fast Prototyping)

Create `my_plugins/focal_loss.py`:

```python
import torch
import torch.nn.functional as F
import reframx as rf

PLUGIN_MANIFEST = rf.PluginManifest(
    name="focal_loss_plugin",
    version="0.1.0",
    api_version="0.1",
    author="Research Team",
    description="Binary focal loss for imbalanced classification.",
    categories=("objective",),
    tags=("loss", "classification"),
)


def register(registry: rf.ComponentRegistry) -> None:
    @registry.register("objective", "focal_loss", version="0.1.0", tags=("loss",))
    class BinaryFocalLoss(rf.BaseObjective):
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
            super().__init__(capabilities=["loss", "focal_loss"])
            self.alpha = alpha
            self.gamma = gamma

        def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            pt = torch.exp(-bce)
            return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()
```

Load and use it:

```python
pm = rf.PluginManager()
pm.load_path("my_plugins/focal_loss.py")

loss_fn = pm.registry.create("objective", "focal_loss", gamma=1.5)
```

---

### Format B: Multi-File Package Directory (No `pip install` Required)

Organize larger plugins into a directory containing `__init__.py`:

```text
my_plugins/dpo_plugin/
├── __init__.py
├── loss.py
└── method.py
```

`my_plugins/dpo_plugin/loss.py`:
```python
import reframx as rf

class DPOLoss(rf.BaseObjective):
    def __init__(self, beta: float = 0.1) -> None:
        super().__init__(capabilities=["dpo_loss", "loss"])
        self.beta = beta

    def compute(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps):
        ...
```

`my_plugins/dpo_plugin/__init__.py`:
```python
import reframx as rf
from .loss import DPOLoss
from .method import DPOMethod

PLUGIN_MANIFEST = rf.PluginManifest(
    name="dpo_plugin",
    version="1.0.0",
    api_version="0.1",
    categories=("method", "objective"),
)


def register(registry: rf.ComponentRegistry) -> None:
    registry.register_factory("method", "dpo", DPOMethod, version="1.0.0")
    registry.register_factory("objective", "dpo_loss", DPOLoss, version="1.0.0")


def unregister(registry: rf.ComponentRegistry) -> None:
    """Optional teardown hook for releasing plugin-owned resources."""
```

Load directly from the local path (relative imports work automatically):

```python
pm = rf.PluginManager()
pm.load_path("my_plugins/dpo_plugin")       # loads the package directory
# Or load every file and package inside a folder:
pm.load_directory("my_plugins/")
```

---

### Format C: Distributable Python Package (Entry Points)

To share a plugin as an installable package via `pip`, declare an entry point under `reframx.plugins` in your plugin's `pyproject.toml`:

```toml
[project]
name = "rf-dpo-plugin"
version = "1.0.0"

[project.entry-points."reframx.plugins"]
dpo_plugin = "dpo_plugin:register"
```

Users can inspect installed plugins **without executing any plugin code**, then opt in explicitly:

```python
pm = rf.PluginManager()

# Safe metadata inspection only (does NOT import or execute plugins)
discovered = pm.discover_entry_points()
for item in discovered:
    print(item.name, item.dist_name, item.dist_version)

# Explicit opt-in loading
pm.load_entry_points(allowlist={"dpo_plugin"})
```

---

## 3. `@registry.register(...)` vs `registry.register_factory(...)`

Both accept the same metadata arguments:

| Argument | Default | Purpose |
|---|---|---|
| `category` | *(required)* | Open string category (`"task"`, `"method"`, `"model"`, `"objective"`, ...). |
| `name` | `obj.__name__` | Lookup key for `registry.get(category, name)` / `registry.create(...)`. |
| `version` | `None` | Optional component version string. |
| `description` | First line of `__doc__` | Human-readable summary. |
| `source` | `obj.__module__` | Origin module path. |
| `tags` | `()` | Searchable classification tags. |
| `metadata` | `{}` | Arbitrary descriptive metadata dictionary. |
| `overwrite` | `False` | Prevents accidental duplicate registration; set `True` to replace. |

- Use **`@registry.register(...)`** when defining classes/factories inline inside `register(registry)`.
- Use **`registry.register_factory(...)`** when registering classes imported from other modules or registering lambdas/callables.

---

## 4. Reloading & Unloading Plugins

```python
# Reload an updated plugin cleanly (automatically unloads previous entries first)
pm.load_path("my_plugins/dpo_plugin", reload=True)

# Unload a single plugin (calls optional unregister(), removes all registry entries & sys.modules)
pm.unload_plugin("dpo_plugin")

# Unload everything in reverse load order
pm.unload_all()
```

---

## 5. Testing Your Plugin

Always test plugins against an isolated `rf.PluginManager()` instance so tests never pollute `rf.default_registry`:

```python
def test_my_plugin():
    pm = rf.PluginManager()
    loaded = pm.load_path("my_plugins/focal_loss.py")
    assert loaded.manifest.name == "focal_loss_plugin"
    assert pm.registry.has("objective", "focal_loss")

    loss_fn = pm.registry.create("objective", "focal_loss", gamma=2.0)
    assert loss_fn is not None

    pm.unload_plugin("focal_loss_plugin")
    assert not pm.registry.has("objective", "focal_loss")
```
