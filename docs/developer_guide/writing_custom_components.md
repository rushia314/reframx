# Developer Guide: Writing Custom Components

Research Framework is designed around **loose coupling and optional inheritance**. You can subclass the `Base*` convenience classes, wrap existing PyTorch/Python objects with adapters, or use plain Python objects that implement `get_spec()`.

---

## 1. Core Design Rules

1. **Declare contracts, not concrete classes:**
   - Components declare what they need (`Requirement` / `RequirementGroup`) and what they provide (`Capability`), optionally with an `IOContract`.
   - Never hard-code a list of "supported tasks" inside a `Method`, or "supported methods" inside a `Task`.
2. **Keep `ComponentSpec` descriptive:**
   - `ComponentSpec` holds metadata, requirements, capabilities, and static attributes — never runtime tensors or mutable training state.
3. **Add custom domain validation via `validate_component(context)`:**
   - Contract checking verifies declared metadata; implement `validate_component(self, context: ValidationContext) -> ValidationReport` for architecture-specific or runtime checks.

---

## 2. Custom Task

A `Task` defines the problem and accepted outputs — no forward pass, loss, or training loop. You can pass explicit `Requirement`/`RequirementGroup`/`IOContract` objects or use the concise `str`/`dict` shorthand (automatically normalized by `ComponentSpec` and `TaskSpec`):

```python
import reframx as rf


class MultiLabelTask(rf.BaseTask):
    def __init__(self, num_labels: int) -> None:
        super().__init__(
            requirements={
                "name": "prediction",
                "mode": "any_of",
                "requirements": [
                    {
                        "name": "logits",
                        "contract": {
                            "kind": "data",
                            "value_type": "tensor",
                            "semantics": "multilabel_logits",
                            "shape": ("B", num_labels),
                        },
                    },
                    "probabilities",
                ],
            },
            problem={
                "name": "Multi-Label Classification",
                "output_semantics": "independent_binary_scores",
            },
            evaluation={"primary_metrics": ("macro_f1",)},
            attributes={"num_labels": num_labels},
        )
```

---

## 3. Custom Model (With Probes, StateProvider & Custom Validator)

You do **not** need to inherit from `rf.BaseModel` — keep your `torch.nn.Module` completely standard and wrap it with `TorchModelAdapter`:

```python
import torch
from torch import nn
import reframx as rf
from reframx.integrations.pytorch import TorchModelAdapter


class CustomResidualNet(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, num_classes)
        self.delta_probe = rf.Probe("res/delta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.relu(self.fc1(x))
        delta = self.delta_probe(delta)   # zero-overhead unless watched
        return self.fc2(x + delta)

    def validate_component(self, ctx: rf.ValidationContext) -> rf.ValidationReport:
        report = rf.ValidationReport()
        if self.dim < 8:
            report.add(
                rf.ValidationIssue(
                    severity=rf.ValidationSeverity.WARNING,
                    code="SMALL_HIDDEN_DIM",
                    message=f"Hidden dim {self.dim} may underfit.",
                )
            )
        return report


# Automatically exposes Capability("forward", contract=IOContract(kind="api", value_type="callable"))
model = TorchModelAdapter(CustomResidualNet(dim=64, num_classes=4))

# Or pass custom capabilities using str/dict shorthand:
# model = TorchModelAdapter(raw_model, capabilities=["forward", {"name": "logits", "contract": {"shape": ("B", 4)}}])
```

`TorchModelAdapter` automatically exposes `StateProvider` (`get_component_state` / `load_component_state`) so `.rfx` export and state restoration work out of the box.

---

## 4. Custom Dataset & Schema Adaptation

Implement `BaseDataset` (or any object providing `get_split(split)`):

```python
import reframx as rf


class CustomStreamingDataset(rf.BaseDataset):
    def __init__(self, train_iter_fn, val_iter_fn) -> None:
        super().__init__(
            capabilities=["raw_signal", "target_y"],
            splits=("train", "val"),
        )
        self._train_fn = train_iter_fn
        self._val_fn = val_iter_fn

    def get_split(self, split: str):
        if split == "train":
            return self._train_fn()
        if split == "val":
            return self._val_fn()
        raise KeyError(split)
```

To reuse this dataset with a `Method` expecting `inputs` and `targets` without modifying either class, wrap it with `DatasetFieldAdapter`:

```python
adapted_dataset = rf.DatasetFieldAdapter(
    dataset,
    field_map={"raw_signal": "inputs", "target_y": "targets"},
    transform=lambda batch: {**batch, "inputs": batch["inputs"] * 0.5},
)
```

---

## 5. Custom Method (Full Manual Freedom)

Subclass `rf.BaseMethod` whenever your algorithm does not fit a single `forward -> loss -> backward -> step` (e.g., GANs, actor-critic RL, distillation, bilevel optimization, custom gradient accumulation):

```python
import reframx as rf


class GANMethod(rf.BaseMethod):
    def __init__(self) -> None:
        super().__init__(
            requirements=[
                {"name": "real_images", "provider_type": "dataset"},
                {"name": "gan_loss", "provider_type": "objective"},
            ],
            capabilities="generated_images",
        )

    def train_step(self, batch, ctx: rf.ExecutionContext) -> rf.MethodResult:
        gen, disc = ctx.model["generator"], ctx.model["discriminator"]
        loss_fn = ctx.objectives["gan"]

        # 1. Update discriminator
        fake = gen(batch["noise"]).detach()
        loss_d = loss_fn.disc_loss(disc(batch["real_images"]), disc(fake))
        ctx.backward(loss_d)
        ctx.step("disc")

        # 2. Update generator
        fake_for_g = gen(batch["noise"])
        loss_g = loss_fn.gen_loss(disc(fake_for_g))
        ctx.backward(loss_g)
        ctx.step("gen")

        return rf.MethodResult(
            task_outputs={"generated_images": fake_for_g.detach()},
            objectives={"loss_d": loss_d, "loss_g": loss_g},
            metrics={"loss_d": loss_d, "loss_g": loss_g},
        )
```

- `train_step` is an `@abstractmethod` (enforced at instantiation to prevent silent bugs).
- `eval_step` is optional: if you do not override it, evaluation is automatically skipped even if the dataset has a `"val"` split.

---

## 6. Custom Objective & Evaluator

```python
import reframx as rf


class HuberObjective(rf.BaseObjective):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__(capabilities=["loss", "huber_loss"])  # defaults to ("loss",) if omitted
        self.delta = delta

    def compute(self, pred, target):
        return torch.nn.functional.huber_loss(pred, target, delta=self.delta)


class TopKAccuracyEvaluator(rf.BaseEvaluator):
    def __init__(self, k: int = 5) -> None:
        super().__init__(requirements="logits")
        self.k = k
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, result: rf.MethodResult, batch) -> None:
        logits = result.task_outputs["logits"]
        topk = logits.topk(self.k, dim=-1).indices
        targets = batch["targets"].unsqueeze(-1)
        self.correct += int((topk == targets).any(dim=-1).sum().item())
        self.total += int(targets.numel())

    def compute(self) -> dict[str, float]:
        return {f"top{self.k}_acc": self.correct / max(self.total, 1)}
```

---

## 7. Custom Hook (Lifecycle Plugin)

Hooks can inherit from `rf.BaseHook` or be plain Python objects defining any subset of lifecycle methods (`on_run_start`, `on_epoch_start`, `on_step_start`, `on_step_end`, `on_eval_start`, `on_eval_end`, `on_epoch_end`, `on_run_end`):

```python
import reframx as rf


class EarlyStoppingHook(rf.BaseHook):
    def __init__(self, metric: str = "eval/loss", patience: int = 3) -> None:
        self.metric = metric
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def on_epoch_end(self, ctx: rf.ExecutionContext, epoch_metrics) -> None:
        value = epoch_metrics.get(self.metric)
        if value is None:
            return
        if value < self.best:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                ctx.request_stop()
```

---

## 8. Custom Strategy & Custom Execution Backend

- **Custom Strategy:** subclass `rf.BaseStrategy` and implement `backward(loss, ctx)`, `optimizer_step(optimizer, ctx, *, zero_grad=True)`, and `zero_grad(optimizer, ctx)` (plus optional `prepare_model`, `prepare_batch`, `eval_context`).
- **Custom Execution Backend:** implement the two-method `rf.ExecutionBackend` protocol (`run(ctx, ...)` and `evaluate(ctx, split=...)`) and pass it via `rf.Experiment(..., backend=MyBackend())`.

---

## 9. Component Testing Checklist

1. **Contract test:** verify `rf.validate_requirements(consumer.spec, [provider.spec]).is_valid`.
2. **Isolation test:** run `Experiment.run(max_epochs=1)` with a 1-batch in-memory `BaseDataset`.
3. **State & `.rfx` test:** if your component holds state, verify `experiment.export(...)` + `load_experiment(...)` + `restore_component_state(...)`.
4. **Stop/interrupt test:** verify `experiment.stop()` or `catch_interrupt=True` leaves clean state.
