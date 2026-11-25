# User Guide

This guide covers every major subsystem of **Research Framework**, from standard built-in workflows to fully custom research algorithms, plugins, instrumentation, and portable `.rfx` packaging.

---

## 1. Contract System & Validation

Every component exposes a `ComponentSpec` declaring:
- `requirements`: a tuple of `Requirement` or `RequirementGroup(mode="all_of" | "any_of")`.
- `capabilities`: a tuple of `Capability`, each optionally carrying an `IOContract(kind, value_type, semantics, dtype, shape, attributes)`.

```python
import reframx as rf

req = rf.Requirement(
    "logits",
    provider_type="method",
    contract=rf.IOContract(kind="data", value_type="tensor", dtype="float32", shape=("B", "C")),
)
```

- Calling `experiment.validate()` (or `experiment.run()`, which validates by default) matches every **Consumer <-> Provider** contract and runs each component's optional `validate_component(context)` method.
- If any `ERROR` issue is found, `experiment.run()` fails fast with `ValidationFailedError` carrying `.report`.
- **Shorthand syntax:** `requirements` and `capabilities` accept plain strings (`"logits"`), dicts (`{"name": "logits", "contract": {"shape": ("B", "C")}}`), or group dicts (`{"name": "prediction", "mode": "any_of", "requirements": [...]}`) — both as lists/tuples or directly as a single item — and automatically coerce them into `Requirement`, `RequirementGroup`, `Capability`, and `IOContract` objects. `problem`, `evaluation`, and `objective_goals` on `BaseTask`/`TaskSpec` also accept dicts.

---

## 2. Building & Composing Components

### 2.1 Task
- Built-in tasks: `rf.ClassificationTask()`, `rf.RegressionTask()`, `rf.CausalLMTask()`.
- Custom task:
  ```python
  task = rf.BaseTask(
      requirements=(rf.Requirement("my_output"),),
      problem=rf.ProblemDefinition(name="Custom Problem"),
      evaluation=rf.EvaluationDefinition(primary_metrics=("my_metric",)),
  )
  ```

### 2.2 Model
No mandatory inheritance is required — wrap any `torch.nn.Module` (or plain Python object) non-invasively:
```python
from reframx.integrations.pytorch import TorchModelAdapter

model = TorchModelAdapter(raw_nn_module, capabilities=(rf.Capability("forward"),))
```
`TorchModelAdapter` preserves `raw_nn_module` without mutating it, records `total_parameters` / `trainable_parameters` in `ModelSpec`, and implements `StateProvider` (`state_dict` / `load_state_dict`).

### 2.3 Dataset & `DatasetFieldAdapter`
```python
from reframx.integrations.pytorch import TorchDatasetAdapter

dataset = TorchDatasetAdapter(
    datasets={"train": train_ds, "val": val_ds},
    dataloader_kwargs={"batch_size": 64, "collate_fn": my_collate},
    capabilities=(rf.Capability("image"), rf.Capability("label")),
)

# If your Method expects different field names ("inputs", "targets"), wrap with DatasetFieldAdapter:
adapted = rf.DatasetFieldAdapter(
    dataset,
    field_map={"image": "inputs", "label": "targets"},
)
```

### 2.4 Objective & Evaluator
- **Built-in PyTorch Objectives:** `CrossEntropyObjective`, `BCEWithLogitsObjective`, `MSEObjective`, or wrap any PyTorch loss module/callable with `TorchLossAdapter(nn.HuberLoss())`.
- **Built-in PyTorch Evaluators:** `TorchAccuracyEvaluator` (handles both binary and multiclass logits) and `TorchRegressionEvaluator` (`mse`, `rmse`, `mae`).
- **Custom:** subclass `rf.BaseObjective` (implement `compute(...)`) and `rf.BaseEvaluator` (implement `update(result, batch)` and `compute()`).

### 2.5 Method — Default vs Full Research Freedom

**Mode A — Standard Supervised Path (`SupervisedMethod` or `TorchClassificationMethod` / `TorchRegressionMethod`):**
```python
class MySupervisedMethod(rf.SupervisedMethod):
    def __init__(self):
        super().__init__(
            requirements=(
                rf.Requirement("forward", provider_type="model"),
                rf.Requirement("loss", provider_type="objective"),
            ),
            capabilities=(rf.Capability("logits"),),
        )

    def compute_step(self, batch, ctx):
        logits = ctx.model(batch["inputs"])
        loss = ctx.objective(logits, batch["targets"])
        return rf.MethodResult(task_outputs={"logits": logits}, loss=loss)
```

**Mode B — Full Manual Control (`BaseMethod` — multiple forwards, backwards, and optimizers):**
```python
class DualOptimMethod(rf.BaseMethod):
    def __init__(self):
        super().__init__(capabilities=(rf.Capability("generated"),))

    def train_step(self, batch, ctx):
        loss_d = ...
        ctx.backward(loss_d)
        ctx.step("disc")

        loss_g = ...
        ctx.backward(loss_g)
        ctx.step("gen")
        return rf.MethodResult(
            task_outputs={"generated": ...},
            metrics={"loss_d": loss_d, "loss_g": loss_g},
        )
```
*(If a `BaseMethod` subclass does not override `eval_step`, evaluation is skipped automatically even when the dataset provides a `"val"` split.)*

---

## 3. Strategy & Pluggable Execution Backends

### Strategy
- `rf.StandardBackwardStrategy(grad_accumulation_steps=1)`: backend-agnostic duck-typed `.backward()` / `.step()` / `.zero_grad()`.
- `TorchStrategy(device="auto", precision="fp16", max_grad_norm=1.0, grad_accumulation_steps=1)`:
  - Automatically moves model and batches to `"cuda"` / `"mps"` / `"cpu"`.
  - Automatically enables `GradScaler` for `fp16` on CUDA and unscales gradients before clipping.

### Execution Backends (Swappable Without Changing User Code)
- **`rf.NativeBackend()` (default):** lightweight, zero-dependency engine supporting `ctx.request_stop()` from hooks.
- **`MMEngineBackend(...)`:** powered by `mmengine.runner.Runner` (GPU/DDP/FSDP, `OptimWrapper`, `ParamScheduler`, `CheckpointHook` with `save_best`/`save_last`, `Visualizer`):
  ```python
  from reframx.integrations.mmengine import MMEngineBackend

  backend = MMEngineBackend(
      work_dir="E:/experiments/exp_01",
      default_hooks={
          "checkpoint": {
              "type": "CheckpointHook",
              "interval": 1,
              "save_best": "accuracy",
              "rule": "greater",
              "save_last": True,
              "max_keep_ckpts": 2,
          }
      },
  )
  ```

### Stopping Training Early & Safe Interrupt Export
- **Ctrl+C (`KeyboardInterrupt`):** pass `catch_interrupt=True` to `experiment.run(...)` so interrupting returns `Run(status=RunStatus.INTERRUPTED)` cleanly and lets subsequent `experiment.export(...)` run.
- **Programmatic stop (from a Hook, timer, UI, or another thread):** call `experiment.stop(reason="...")` — stops gracefully after the current step on both `NativeBackend` and `MMEngineBackend`.

```python
import threading

threading.Timer(60.0, experiment.stop, args=("time_budget_reached",)).start()
run = experiment.run(max_epochs=100, catch_interrupt=True)
experiment.export("outputs/snapshot.rfx", profile="standard")
```

---

## 4. Model Instrumentation

Two complementary, autograd-safe mechanisms (tensors passed to callbacks are automatically detached while the original tensor continues through the computation graph):

```python
# 1. Explicit Probe inside forward() — no need to pass `ctx` into forward()
hidden = rf.probe("encoder/hidden", hidden)

# 2. Non-invasive module hooks by object reference or glob pattern
inst = rf.InstrumentationManager(max_records=500)
inst.watch_probe("encoder/*", rf.tensor_stats, every_n_steps=10)
inst.watch_probe("encoder/*", lambda ev: rf.check_nan_inf(ev, raise_on_error=True))
inst.watch_modules(raw_model, pattern="*.attn", stages=("post_forward", "backward"), callback=rf.tensor_stats)

exp = rf.Experiment(..., hooks=[rf.InstrumentationHook(inst)])
run = exp.run(max_epochs=5)
# Recorded metrics are stored in run.train_history and run.artifacts["instrumentation"]
```

---

## 5. Writing, Loading & Unloading Plugins

A plugin only requires a `register(registry)` function (with optional `PLUGIN_MANIFEST` and `unregister(registry)`):

```python
# my_plugins/custom_loss.py
import reframx as rf

PLUGIN_MANIFEST = rf.PluginManifest(name="custom_loss", version="0.1.0")

def register(registry: rf.ComponentRegistry) -> None:
    @registry.register("objective", "my_loss")
    class MyLoss(rf.BaseObjective):
        def __init__(self):
            super().__init__(capabilities=(rf.Capability("loss"),))
        def compute(self, pred, target):
            ...
```

Loading, instantiating, and unloading plugins:
```python
pm = rf.PluginManager()
pm.load_path("my_plugins/custom_loss.py")   # single .py file OR multi-file package dir with __init__.py
pm.load_directory("my_plugins/")            # load all plugin files and packages in a directory
pm.discover_entry_points()                  # inspect entry points WITHOUT executing code
pm.load_entry_points(allowlist={"my_pkg"})  # explicitly load trusted entry points

loss = pm.registry.create("objective", "my_loss")
pm.unload_plugin("custom_loss")             # calls unregister() and cleans up registry & sys.modules
```

---

## 6. Portable `.rfx` Experiment Packaging

An `.rfx` file is an **unencrypted standard ZIP archive** written atomically with SHA-256 checksums for every member.

| Profile | Archive Contents |
|---|---|
| `minimal` | `manifest.json` + `config.json` (metadata, specs, environment snapshot) |
| `standard` *(default)* | `minimal` + `runs.json` + `checkpoints/` (component/optimizer states + external checkpoints) |
| `portable` | `standard` + `assets/` (tokenizers, configs, dataset assets) |
| `full` | `portable` + `artifacts/` (instrumentation records, logs, run artifacts) |

```python
# Export an experiment snapshot
path = experiment.export(
    "outputs/exp_01.rfx",
    profile="full",
    checkpoint_paths=["work_dirs/best_accuracy_epoch_2.pth"],
    assets={"vocab": "path/to/vocab.txt"},
)

# Load safely (never auto-installs packages or executes plugins)
imported = rf.load_experiment("outputs/exp_01.rfx")
imported.restore_component_state("model", fresh_model)

# If you manually edited files inside the .rfx ZIP archive:
imported = rf.load_experiment("outputs/exp_01.rfx", verify_checksums=False)
```
