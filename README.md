# Research Framework

A modular, Python-first, code-first framework for neural network research.

**Core philosophy:** Researchers own their architectures, losses, data pipelines, and training algorithms. The framework provides contract-driven composition, validation, pluggable execution backends (lightweight native & MMEngine), model instrumentation, plugins, and portable `.rfx` experiment snapshots — without forcing a fixed model hierarchy or a single `forward -> loss -> backward -> step` loop.

## Key Features

- **Zero-dependency core:** Core contracts, validation, execution engine, instrumentation, plugins, and `.rfx` packaging use only the Python standard library.
- **Contract-driven composition:** `ComponentSpec`, `Requirement`, `RequirementGroup` (`all_of` / `any_of`), `Capability`, `IOContract`, and `validate_requirements()`.
- **Full research freedom + sensible defaults:**
  - `BaseMethod` for arbitrary multi-forward / multi-backward / multi-optimizer algorithms.
  - `SupervisedMethod` and PyTorch built-ins (`TorchClassificationMethod`, `TorchRegressionMethod`, `CrossEntropyObjective`, `BCEWithLogitsObjective`, `MSEObjective`, `TorchAccuracyEvaluator`, `TorchRegressionEvaluator`) for fast standard training.
- **Pluggable Execution Backends:**
  - `NativeBackend` (default, 0 dependencies, easy debugging).
  - `MMEngineBackend` (optional, lazy-imported: GPU/DDP, `OptimWrapper`, `ParamScheduler`, `CheckpointHook` with `save_best`/`save_last`, `Visualizer`).
- **PyTorch GPU integration (non-invasive):** `TorchModelAdapter`, `TorchDatasetAdapter`, `TorchStrategy` (auto device, AMP `fp16`/`bf16`, `GradScaler`, `max_grad_norm`, gradient accumulation) — never overwrites your existing PyTorch/CUDA build.
- **Model Instrumentation:** `rf.probe()` / `rf.Probe` (zero-overhead pass-through via `contextvars`), `watch_module` / `watch_modules`, auto-detach, `tensor_stats`, and `check_nan_inf`.
- **Plugin System:** `ComponentRegistry` and `PluginManager` supporting `load_plugin`, `load_file`, `load_path` (multi-file packages), `load_directory`, `discover_entry_points` / `load_entry_points`, and clean `unload_plugin`.
- **Portable `.rfx` Archives:** Atomic ZIP-based experiment snapshots (`minimal`, `standard`, `portable`, `full` profiles) with SHA-256 verification and safe state restoration.

## Installation

```bash
pip install -e ".[dev]"
```

> **Note:** `torch` and `mmengine` are intentionally **not** declared as mandatory dependencies so installing `reframx` never overwrites your platform-specific CUDA/PyTorch build. Install them separately as needed for your hardware.

## Quick Example (PyTorch GPU + MMEngine)

```python
import torch
import reframx as rf
from reframx.integrations.mmengine import MMEngineBackend
from reframx.integrations.pytorch import (
    BCEWithLogitsObjective,
    TorchAccuracyEvaluator,
    TorchClassificationMethod,
    TorchDatasetAdapter,
    TorchModelAdapter,
    TorchStrategy,
)

experiment = rf.Experiment(
    name="QuickDemo",
    task=rf.ClassificationTask(num_classes=2),
    method=TorchClassificationMethod(),
    model=TorchModelAdapter(raw_model),
    dataset=TorchDatasetAdapter(datasets={"train": train_ds, "val": val_ds}),
    objectives=BCEWithLogitsObjective(),
    optimizers=torch.optim.AdamW(raw_model.parameters(), lr=1e-3),
    evaluator=TorchAccuracyEvaluator(),
    strategy=TorchStrategy(device="auto", precision="fp16", max_grad_norm=1.0),
    backend=MMEngineBackend(work_dir="./work_dirs/quick_demo"),
)

run = experiment.run(max_epochs=5)
experiment.export("./work_dirs/quick_demo.rfx", profile="standard")
```

## Documentation

- [Getting Started & Quickstart](docs/getting_started/quickstart.md)
- [User Guide](docs/user_guide/usage_guide.md)
- [Developer Guide: Writing Custom Components](docs/developer_guide/writing_custom_components.md)
- [Developer Guide: Writing Plugins](docs/developer_guide/writing_plugins.md)
- [Architecture & Class/Module Diagrams](docs/developer_guide/architecture_diagrams.md)
- [API Reference](docs/api_reference/index.md)

## Examples

- [examples/common_component.py](examples/common_component.py) — Contract & compatibility validation.
- [examples/basic/pytorch_experiment.py](examples/basic/pytorch_experiment.py) — Concise PyTorch experiment using built-ins.
- [examples/basic/text_attention_binary_classification.py](examples/basic/text_attention_binary_classification.py) — Multi-Head Attention text classification on GPU with `tiktoken`, HuggingFace `datasets`, and `MMEngineBackend`.
- [examples/custom_task/multilabel_task.py](examples/custom_task/multilabel_task.py) — Custom `BaseTask` with shorthand `RequirementGroup` & `IOContract`.
- [examples/custom_model/residual_model.py](examples/custom_model/residual_model.py) — Plain PyTorch `nn.Module` + `TorchModelAdapter` + custom validator + `.rfx` restore.
- [examples/custom_method/dual_optimizer_method.py](examples/custom_method/dual_optimizer_method.py) — Manual `BaseMethod` with 2 models, 2 losses, and 2 optimizers.
- [examples/custom_execution/custom_strategy_and_hooks.py](examples/custom_execution/custom_strategy_and_hooks.py) — Gradient accumulation, `EarlyStopOnLossHook`, `experiment.stop()`, and `catch_interrupt=True`.
- [examples/custom_instrumentation/probes_and_module_watch.py](examples/custom_instrumentation/probes_and_module_watch.py) — `rf.probe()` + non-invasive `watch_modules()` (`post_forward` & `backward`) + `check_nan_inf`.
- [examples/basic/full_experiment.py](examples/basic/full_experiment.py) & [examples/basic/real_binary_classification.py](examples/basic/real_binary_classification.py) — Zero-dependency pure-Python examples.

## Running Tests

```bash
python -m pytest
python -m ruff check src tests examples
python -m mypy src
```
