# reframx

[![PyPI version](https://img.shields.io/pypi/v/reframx.svg)](https://pypi.org/project/reframx/)
[![Python versions](https://img.shields.io/pypi/pyversions/reframx.svg)](https://pypi.org/project/reframx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/rushia314/reframx/blob/main/LICENSE)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-brightgreen.svg)](https://peps.python.org/pep-0561/)

A modular, Python-first, code-first framework for neural network research.

> **Core philosophy:** Researchers own their architectures, losses, data pipelines, and training algorithms. `reframx` provides contract-driven composition, validation, pluggable execution backends, model instrumentation, plugins, and portable `.rfx` experiment snapshots — without forcing a fixed model hierarchy or a single `forward -> loss -> backward -> step` loop.

## Key Features

- **Zero-dependency core:** Core contracts, validation, execution engine, instrumentation, plugins, and `.rfx` packaging rely only on the Python standard library.
- **Contract-driven composition:** `ComponentSpec`, `Requirement`, `RequirementGroup` (`all_of` / `any_of`), `Capability`, `IOContract`, and `validate_requirements()` with concise `str`/`dict` shorthand.
- **Full research freedom + sensible defaults:**
  - `BaseMethod` for arbitrary multi-forward / multi-backward / multi-optimizer algorithms.
  - `SupervisedMethod` and PyTorch built-ins (`TorchClassificationMethod`, `TorchRegressionMethod`, `CrossEntropyObjective`, `BCEWithLogitsObjective`, `MSEObjective`, `TorchAccuracyEvaluator`, `TorchRegressionEvaluator`) for rapid standard workflows.
- **Pluggable execution backends:**
  - `NativeBackend` (default, zero dependencies, easy debugging, `experiment.stop()`).
  - `MMEngineBackend` (optional, lazy-imported: GPU/DDP/FSDP, `OptimWrapper`, `ParamScheduler`, `CheckpointHook` with `save_best`/`save_last`, `Visualizer`).
- **Non-invasive PyTorch GPU integration:** `TorchModelAdapter`, `TorchDatasetAdapter`, and `TorchStrategy` (auto device placement, AMP `fp16`/`bf16`, `GradScaler`, `max_grad_norm`, gradient accumulation) — never overwrites your existing PyTorch/CUDA installation.
- **Model instrumentation:** `rf.probe()` / `rf.Probe` (zero-overhead pass-through via `contextvars`), `watch_module` / `watch_modules`, automatic tensor detach, `tensor_stats`, and `check_nan_inf`.
- **Plugin system:** `ComponentRegistry` and `PluginManager` supporting `load_plugin`, `load_file`, `load_path` (multi-file packages), `load_directory`, safe `discover_entry_points` / `load_entry_points`, and clean `unload_plugin`.
- **Portable `.rfx` archives:** Atomic, unencrypted ZIP experiment snapshots (`minimal`, `standard`, `portable`, `full` profiles) with SHA-256 verification and safe state restoration.

## Installation

```bash
pip install reframx
```

> **Optional integrations:** `torch` and `mmengine` are intentionally **not** installed automatically so `pip install reframx` never overwrites your platform-specific CUDA/PyTorch build. Install PyTorch for your hardware from [pytorch.org](https://pytorch.org/get-started/locally/), and optionally `pip install mmengine` for `MMEngineBackend`.

## Quick Start

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset
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

# 1. Standard PyTorch model and dataset
raw_model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
x_train, y_train = torch.randn(256, 16), torch.randint(0, 2, (256, 1)).float()
x_val, y_val = torch.randn(64, 16), torch.randint(0, 2, (64, 1)).float()
train_ds, val_ds = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)

# 2. Compose & run on GPU with MMEngine
experiment = rf.Experiment(
    name="QuickDemo",
    task=rf.ClassificationTask(num_classes=2),
    method=TorchClassificationMethod(),
    model=TorchModelAdapter(raw_model),
    dataset=TorchDatasetAdapter(
        datasets={"train": train_ds, "val": val_ds},
        dataloader_kwargs={"batch_size": 32},
    ),
    objectives=BCEWithLogitsObjective(),
    optimizers=torch.optim.AdamW(raw_model.parameters(), lr=1e-3),
    evaluator=TorchAccuracyEvaluator(),
    strategy=TorchStrategy(device="auto", precision="fp16", max_grad_norm=1.0),
    backend=MMEngineBackend(work_dir="./work_dirs/quick_demo"),
)

run = experiment.run(max_epochs=5, catch_interrupt=True)
experiment.export("./work_dirs/quick_demo.rfx", profile="standard")
```

## Documentation

- [Getting Started & Quickstart](https://github.com/rushia314/reframx/blob/main/docs/getting_started/quickstart.md)
- [User Guide](https://github.com/rushia314/reframx/blob/main/docs/user_guide/usage_guide.md)
- [Developer Guide: Writing Custom Components](https://github.com/rushia314/reframx/blob/main/docs/developer_guide/writing_custom_components.md)
- [Developer Guide: Writing Plugins](https://github.com/rushia314/reframx/blob/main/docs/developer_guide/writing_plugins.md)
- [Architecture & Class/Module Diagrams](https://github.com/rushia314/reframx/blob/main/docs/developer_guide/architecture_diagrams.md)
- [API Reference](https://github.com/rushia314/reframx/blob/main/docs/api_reference/index.md)

## Examples

- [examples/common_component.py](https://github.com/rushia314/reframx/blob/main/examples/common_component.py) — Contract & compatibility validation.
- [examples/basic/pytorch_experiment.py](https://github.com/rushia314/reframx/blob/main/examples/basic/pytorch_experiment.py) — Concise PyTorch experiment using built-ins.
- [examples/basic/text_attention_binary_classification.py](https://github.com/rushia314/reframx/blob/main/examples/basic/text_attention_binary_classification.py) — Multi-Head Attention text classification on GPU with `tiktoken`, HuggingFace `datasets`, and `MMEngineBackend`.
- [examples/custom_task/multilabel_task.py](https://github.com/rushia314/reframx/blob/main/examples/custom_task/multilabel_task.py) — Custom `BaseTask` with shorthand `RequirementGroup` & `IOContract`.
- [examples/custom_model/residual_model.py](https://github.com/rushia314/reframx/blob/main/examples/custom_model/residual_model.py) — Plain PyTorch `nn.Module` + `TorchModelAdapter` + custom validator + `.rfx` restore.
- [examples/custom_method/dual_optimizer_method.py](https://github.com/rushia314/reframx/blob/main/examples/custom_method/dual_optimizer_method.py) — Manual `BaseMethod` with 2 models, 2 losses, and 2 optimizers.
- [examples/custom_execution/custom_strategy_and_hooks.py](https://github.com/rushia314/reframx/blob/main/examples/custom_execution/custom_strategy_and_hooks.py) — Gradient accumulation, `EarlyStopOnLossHook`, `experiment.stop()`, and `catch_interrupt=True`.
- [examples/custom_instrumentation/probes_and_module_watch.py](https://github.com/rushia314/reframx/blob/main/examples/custom_instrumentation/probes_and_module_watch.py) — `rf.probe()` + non-invasive `watch_modules()` (`post_forward` & `backward`) + `check_nan_inf`.
- [examples/basic/full_experiment.py](https://github.com/rushia314/reframx/blob/main/examples/basic/full_experiment.py) & [examples/basic/real_binary_classification.py](https://github.com/rushia314/reframx/blob/main/examples/basic/real_binary_classification.py) — Zero-dependency pure-Python examples.

## License

Released under the [MIT License](https://github.com/rushia314/reframx/blob/main/LICENSE).
