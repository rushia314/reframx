# Getting Started — Quickstart Guide

## 1. Installation

### Development Environment (Zero-dependency Core)
```bash
pip install reframx
```

### Optional Integrations (PyTorch GPU & MMEngine)
The package intentionally does **not** declare `torch` or `mmengine` as mandatory dependencies in `pyproject.toml`, ensuring that installing `reframx` never overwrites your existing platform-specific PyTorch/CUDA build.

- Install the PyTorch build matching your hardware from https://pytorch.org/get-started/locally/
- Optionally install MMEngine for `MMEngineBackend`:
  ```bash
  pip install mmengine
  ```

---

## 2. Run Your First Experiment (Zero External Dependencies)

```bash
python examples/basic/real_binary_classification.py
```

This example runs a complete binary classification workflow on **Fisher's Iris** (100 real samples, stratified 80/20 split), demonstrating:
- `ClassificationTask`, `BaseDataset`, and `DatasetFieldAdapter`
- `ModelAdapter`, `SupervisedMethod`, `BaseObjective`, and `BaseEvaluator`
- `InstrumentationManager` + `Probe` (`tensor_stats`, `check_nan_inf`)
- `Experiment.validate()` -> `Experiment.run()` -> `Experiment.export(".rfx")` -> `load_experiment()`

---

## 3. Run a PyTorch GPU + Attention + MMEngine Experiment

```bash
python examples/basic/text_attention_binary_classification.py
```

This example trains a compact `TinyAttentionClassifier` (`nn.TransformerEncoder`) on GPU for binary text classification (`sg247/binary-classification`, tokenized with `tiktoken`), using `TorchStrategy` + `MMEngineBackend` (with automatic `save_best` / `save_last` checkpointing) and exporting a portable `.rfx` archive.

---

## 4. Standard 5-Step Pipeline

```python
import reframx as rf

# 1. Reuse built-in components or write your own Task, Model, Dataset, Objective, Evaluator, Method
# 2. Compose an Experiment
experiment = rf.Experiment(
    name="MyExperiment",
    task=task,
    method=method,
    model=model,
    dataset=dataset,
    objectives=objective,
    optimizers=optimizer,
    evaluator=evaluator,
    strategy=strategy,    # default: StandardBackwardStrategy()
    backend=backend,      # default: NativeBackend() | optional: MMEngineBackend(...)
    hooks=[...],
)

# 3. Validate contracts and custom component validators
report = experiment.validate()
assert report.is_valid

# 4. Run training and evaluation
run = experiment.run(max_epochs=10)

# 5. Export the experiment snapshot as a portable .rfx archive
experiment.export("outputs/my_experiment.rfx", profile="standard")
```
