"""End-to-end binary classification on real data (Fisher's Iris: versicolor vs virginica).

Exercises the full Research Framework stack with zero external dependencies:
- Built-in ClassificationTask
- BaseDataset + DatasetFieldAdapter (schema mapping)
- Plain 2-layer MLP wrapped via ModelAdapter (with StateProvider + Probe)
- Custom Binary Cross-Entropy Objective with L2 weight decay
- Custom Binary Metrics Evaluator (accuracy, precision, recall, F1)
- SupervisedMethod + StandardBackwardStrategy
- InstrumentationManager + InstrumentationHook (hidden stats + NaN check)
- Full Experiment.validate() -> Experiment.run() -> Experiment.export(.rfx) -> load_experiment()
"""

from __future__ import annotations

import math
import random
import tempfile
from pathlib import Path
from typing import Any

import reframx as rf

# ---------------------------------------------------------------------------
# Real dataset: Fisher's Iris (1936) — Iris versicolor (0) vs Iris virginica (1)
# 100 real samples, 4 features: (sepal_len, sepal_wid, petal_len, petal_wid)
# ---------------------------------------------------------------------------
IRIS_VERSICOLOR_VS_VIRGINICA: list[tuple[list[float], float]] = [
    # Iris versicolor (50 real samples, label = 0.0)
    ([7.0, 3.2, 4.7, 1.4], 0.0), ([6.4, 3.2, 4.5, 1.5], 0.0), ([6.9, 3.1, 4.9, 1.5], 0.0),
    ([5.5, 2.3, 4.0, 1.3], 0.0), ([6.5, 2.8, 4.6, 1.5], 0.0), ([5.7, 2.8, 4.5, 1.3], 0.0),
    ([6.3, 3.3, 4.7, 1.6], 0.0), ([4.9, 2.4, 3.3, 1.0], 0.0), ([6.6, 2.9, 4.6, 1.3], 0.0),
    ([5.2, 2.7, 3.9, 1.4], 0.0), ([5.0, 2.0, 3.5, 1.0], 0.0), ([5.9, 3.0, 4.2, 1.5], 0.0),
    ([6.0, 2.2, 4.0, 1.0], 0.0), ([6.1, 2.9, 4.7, 1.4], 0.0), ([5.6, 2.9, 3.6, 1.3], 0.0),
    ([6.7, 3.1, 4.4, 1.4], 0.0), ([5.6, 3.0, 4.5, 1.5], 0.0), ([5.8, 2.7, 4.1, 1.0], 0.0),
    ([6.2, 2.2, 4.5, 1.5], 0.0), ([5.6, 2.5, 3.9, 1.1], 0.0), ([5.9, 3.2, 4.8, 1.8], 0.0),
    ([6.1, 2.8, 4.0, 1.3], 0.0), ([6.3, 2.5, 4.9, 1.5], 0.0), ([6.1, 2.8, 4.7, 1.2], 0.0),
    ([6.4, 2.9, 4.3, 1.3], 0.0), ([6.6, 3.0, 4.4, 1.4], 0.0), ([6.8, 2.8, 4.8, 1.4], 0.0),
    ([6.7, 3.0, 5.0, 1.7], 0.0), ([6.0, 2.9, 4.5, 1.5], 0.0), ([5.7, 2.6, 3.5, 1.0], 0.0),
    ([5.5, 2.4, 3.8, 1.1], 0.0), ([5.5, 2.4, 3.7, 1.0], 0.0), ([5.8, 2.7, 3.9, 1.2], 0.0),
    ([6.0, 2.7, 5.1, 1.6], 0.0), ([5.4, 3.0, 4.5, 1.5], 0.0), ([6.0, 3.4, 4.5, 1.6], 0.0),
    ([6.7, 3.1, 4.7, 1.5], 0.0), ([6.3, 2.3, 4.4, 1.3], 0.0), ([5.6, 3.0, 4.1, 1.3], 0.0),
    ([5.5, 2.5, 4.0, 1.3], 0.0), ([5.5, 2.6, 4.4, 1.2], 0.0), ([6.1, 3.0, 4.6, 1.4], 0.0),
    ([5.8, 2.6, 4.0, 1.2], 0.0), ([5.0, 2.3, 3.3, 1.0], 0.0), ([5.6, 2.7, 4.2, 1.3], 0.0),
    ([5.7, 3.0, 4.2, 1.2], 0.0), ([5.7, 2.9, 4.2, 1.3], 0.0), ([6.2, 2.9, 4.3, 1.3], 0.0),
    ([5.1, 2.5, 3.0, 1.1], 0.0), ([5.7, 2.8, 4.1, 1.3], 0.0),
    # Iris virginica (50 real samples, label = 1.0)
    ([6.3, 3.3, 6.0, 2.5], 1.0), ([5.8, 2.7, 5.1, 1.9], 1.0), ([7.1, 3.0, 5.9, 2.1], 1.0),
    ([6.3, 2.9, 5.6, 1.8], 1.0), ([6.5, 3.0, 5.8, 2.2], 1.0), ([7.6, 3.0, 6.6, 2.1], 1.0),
    ([4.9, 2.5, 4.5, 1.7], 1.0), ([7.3, 2.9, 6.3, 1.8], 1.0), ([6.7, 2.5, 5.8, 1.8], 1.0),
    ([7.2, 3.6, 6.1, 2.5], 1.0), ([6.5, 3.2, 5.1, 2.0], 1.0), ([6.4, 2.7, 5.3, 1.9], 1.0),
    ([6.8, 3.0, 5.5, 2.1], 1.0), ([5.7, 2.5, 5.0, 2.0], 1.0), ([5.8, 2.8, 5.1, 2.4], 1.0),
    ([6.4, 3.2, 5.3, 2.3], 1.0), ([6.5, 3.0, 5.5, 1.8], 1.0), ([7.7, 3.8, 6.7, 2.2], 1.0),
    ([7.7, 2.6, 6.9, 2.3], 1.0), ([6.0, 2.2, 5.0, 1.5], 1.0), ([6.9, 3.2, 5.7, 2.3], 1.0),
    ([5.6, 2.8, 4.9, 2.0], 1.0), ([7.7, 2.8, 6.7, 2.0], 1.0), ([6.3, 2.7, 4.9, 1.8], 1.0),
    ([6.7, 3.3, 5.7, 2.1], 1.0), ([7.2, 3.2, 6.0, 1.8], 1.0), ([6.2, 2.8, 4.8, 1.8], 1.0),
    ([6.1, 3.0, 4.9, 1.8], 1.0), ([6.4, 2.8, 5.6, 2.1], 1.0), ([7.2, 3.0, 5.8, 1.6], 1.0),
    ([7.4, 2.8, 6.1, 1.9], 1.0), ([7.9, 3.8, 6.4, 2.0], 1.0), ([6.4, 2.8, 5.6, 2.2], 1.0),
    ([6.3, 2.8, 5.1, 1.5], 1.0), ([6.1, 2.6, 5.6, 1.4], 1.0), ([7.7, 3.0, 6.1, 2.3], 1.0),
    ([6.3, 3.4, 5.6, 2.4], 1.0), ([6.4, 3.1, 5.5, 1.8], 1.0), ([6.0, 3.0, 4.8, 1.8], 1.0),
    ([6.9, 3.1, 5.4, 2.1], 1.0), ([6.7, 3.1, 5.6, 2.4], 1.0), ([6.9, 3.1, 5.1, 2.3], 1.0),
    ([5.8, 2.7, 5.1, 1.9], 1.0), ([6.8, 3.2, 5.9, 2.3], 1.0), ([6.7, 3.3, 5.7, 2.5], 1.0),
    ([6.7, 3.0, 5.2, 2.3], 1.0), ([6.3, 2.5, 5.0, 1.9], 1.0), ([6.5, 3.0, 5.2, 2.0], 1.0),
    ([6.2, 3.4, 5.4, 2.3], 1.0), ([5.9, 3.0, 5.1, 1.8], 1.0),
]


# ---------------------------------------------------------------------------
# Lightweight pure-Python autograd (Value) + Momentum SGD
# ---------------------------------------------------------------------------


class Value:
    __slots__ = ("data", "grad", "_prev", "_backward")

    def __init__(self, data: float, _children: tuple[Value, ...] = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = _children
        self._backward = lambda: None

    def item(self) -> float:
        return self.data

    def detach(self) -> float:
        return self.data

    def __add__(self, other: Value | float) -> Value:
        ov = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + ov.data, (self, ov))

        def _bwd() -> None:
            self.grad += out.grad
            ov.grad += out.grad

        out._backward = _bwd
        return out

    def __mul__(self, other: Value | float) -> Value:
        ov = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * ov.data, (self, ov))

        def _bwd() -> None:
            self.grad += ov.data * out.grad
            ov.grad += self.data * out.grad

        out._backward = _bwd
        return out

    def __truediv__(self, denom: float) -> Value:
        return self * (1.0 / denom)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _bwd() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _bwd
        return out

    def bce_with_logit(self, target: float) -> Value:
        """Numerically stable binary cross-entropy from raw logit."""
        z = self.data
        loss_val = max(z, 0.0) - z * target + math.log1p(math.exp(-abs(z)))
        out = Value(loss_val, (self,))

        def _bwd() -> None:
            sig = 1.0 / (1.0 + math.exp(-z))
            self.grad += (sig - target) * out.grad

        out._backward = _bwd
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class MomentumSGD:
    def __init__(
        self,
        params: list[Value],
        lr: float = 0.08,
        momentum: float = 0.8,
        weight_decay: float = 1e-3,
    ) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [0.0] * len(params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            g = p.grad + self.weight_decay * p.data
            self.velocities[i] = self.momentum * self.velocities[i] + g
            p.data -= self.lr * self.velocities[i]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


# ---------------------------------------------------------------------------
# User research components: MLP, BCE Objective, Binary Metrics Evaluator, Method
# ---------------------------------------------------------------------------


class BinaryMLP:
    """Plain 2-layer MLP (4 -> 8 -> 1) with Probe and StateProvider."""

    def __init__(self, in_dim: int = 4, hidden_dim: int = 8, seed: int = 42) -> None:
        rng = random.Random(seed)
        scale1 = math.sqrt(2.0 / in_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)
        self.w1 = [
            [Value(rng.gauss(0.0, scale1)) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [Value(0.0) for _ in range(hidden_dim)]
        self.w2 = [Value(rng.gauss(0.0, scale2)) for _ in range(hidden_dim)]
        self.b2 = Value(0.0)
        self.hidden_probe = rf.Probe("mlp/hidden")

    def parameters(self) -> list[Value]:
        return [w for row in self.w1 for w in row] + self.b1 + self.w2 + [self.b2]

    def get_component_state(self) -> dict[str, Any]:
        return {
            "w1": [[w.data for w in row] for row in self.w1],
            "b1": [b.data for b in self.b1],
            "w2": [w.data for w in self.w2],
            "b2": self.b2.data,
        }

    def load_component_state(self, state: dict[str, Any]) -> None:
        for r, row in enumerate(self.w1):
            for c, w in enumerate(row):
                w.data = float(state["w1"][r][c])
        for i, b in enumerate(self.b1):
            b.data = float(state["b1"][i])
        for i, w in enumerate(self.w2):
            w.data = float(state["w2"][i])
        self.b2.data = float(state["b2"])

    def forward(self, batch_x: list[list[float]]) -> list[Value]:
        logits: list[Value] = []
        hidden_acts: list[Value] = []
        for x in batch_x:
            h: list[Value] = []
            for weights, bias in zip(self.w1, self.b1, strict=True):
                act = bias
                for wi, xi in zip(weights, x, strict=True):
                    act = act + wi * xi
                h_val = act.tanh()
                h.append(h_val)
                hidden_acts.append(h_val)
            logit = self.b2
            for wi, hi in zip(self.w2, h, strict=True):
                logit = logit + wi * hi
            logits.append(logit)
        self.hidden_probe(hidden_acts)
        return logits


class BinaryCrossEntropyWithLogits(rf.BaseObjective):
    def __init__(self) -> None:
        super().__init__(capabilities=(rf.Capability("bce_loss"), rf.Capability("loss")))

    def compute(self, logits: list[Value], targets: list[float]) -> Value:
        total = Value(0.0)
        for z, y in zip(logits, targets, strict=True):
            total = total + z.bce_with_logit(y)
        return total / float(len(logits))


class BinaryMetricsEvaluator(rf.BaseEvaluator):
    def __init__(self) -> None:
        super().__init__(requirements=(rf.Requirement("logits"),))
        self.tp = self.tn = self.fp = self.fn = 0

    def reset(self) -> None:
        self.tp = self.tn = self.fp = self.fn = 0

    def update(self, result: rf.MethodResult, batch: dict[str, list]) -> None:
        logits: list[Value] = result.task_outputs["logits"]
        for z, y in zip(logits, batch["targets"], strict=True):
            pred = 1 if z.item() >= 0.0 else 0
            label = int(y)
            if pred == 1 and label == 1:
                self.tp += 1
            elif pred == 0 and label == 0:
                self.tn += 1
            elif pred == 1 and label == 0:
                self.fp += 1
            else:
                self.fn += 1

    def compute(self) -> dict[str, float]:
        total = max(self.tp + self.tn + self.fp + self.fn, 1)
        acc = (self.tp + self.tn) / total
        prec = self.tp / max(self.tp + self.fp, 1)
        rec = self.tp / max(self.tp + self.fn, 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


class IrisBinaryMethod(rf.SupervisedMethod):
    def __init__(self) -> None:
        super().__init__(
            requirements=(
                rf.Requirement("inputs", provider_type="dataset"),
                rf.Requirement("targets", provider_type="dataset"),
                rf.Requirement("forward", provider_type="model"),
                rf.Requirement("bce_loss", provider_type="objective"),
            ),
            capabilities=(rf.Capability("logits"),),
        )

    def compute_step(
        self,
        batch: dict[str, list],
        ctx: rf.ExecutionContext,
    ) -> rf.MethodResult:
        logits = ctx.model.forward(batch["inputs"])
        loss = ctx.objective(logits, batch["targets"])
        return rf.MethodResult(task_outputs={"logits": logits}, loss=loss)


class ProgressLogger(rf.BaseHook):
    def __init__(self, every: int = 10) -> None:
        self.every = every

    def on_epoch_end(self, ctx: rf.ExecutionContext, epoch_metrics: dict[str, Any]) -> None:
        ep = ctx.state.epoch
        if ep == 1 or ep % self.every == 0:
            loss = epoch_metrics.get("train/loss", float("nan"))
            val_loss = epoch_metrics.get("eval/loss", float("nan"))
            acc = epoch_metrics.get("eval/accuracy", float("nan"))
            f1 = epoch_metrics.get("eval/f1", float("nan"))
            print(
                f"Epoch {ep:2d} | train/loss={loss:.4f} | "
                f"val/loss={val_loss:.4f} | val/acc={acc:.2%} | val/f1={f1:.4f}"
            )


def prepare_iris_splits(
    *,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[list[dict[str, list]], list[dict[str, list]]]:
    """Stratified 80/20 split + z-score standardization fitted on train only."""
    rng = random.Random(seed)
    class0 = [row for row in IRIS_VERSICOLOR_VS_VIRGINICA if row[1] == 0.0]
    class1 = [row for row in IRIS_VERSICOLOR_VS_VIRGINICA if row[1] == 1.0]
    rng.shuffle(class0)
    rng.shuffle(class1)

    train_rows = class0[:40] + class1[:40]
    val_rows = class0[40:] + class1[40:]
    rng.shuffle(train_rows)

    # Fit feature mean/std strictly on train split (no leakage).
    num_feats = 4
    means = [
        sum(x[j] for x, _ in train_rows) / len(train_rows) for j in range(num_feats)
    ]
    stds = [
        math.sqrt(
            sum((x[j] - means[j]) ** 2 for x, _ in train_rows) / len(train_rows)
        )
        or 1.0
        for j in range(num_feats)
    ]

    def normalize(rows: list[tuple[list[float], float]]) -> list[tuple[list[float], float]]:
        return [
            ([(x[j] - means[j]) / stds[j] for j in range(num_feats)], y)
            for x, y in rows
        ]

    def make_batches(
        rows: list[tuple[list[float], float]],
    ) -> list[dict[str, list]]:
        batches: list[dict[str, list]] = []
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            batches.append(
                {
                    "features": [x for x, _ in chunk],
                    "species": [y for _, y in chunk],
                }
            )
        return batches

    return make_batches(normalize(train_rows)), make_batches(normalize(val_rows))


def main() -> None:
    train_batches, val_batches = prepare_iris_splits(batch_size=16, seed=42)

    # Dataset uses its native field names ("features", "species")...
    raw_dataset = rf.BaseDataset(
        capabilities=(rf.Capability("features"), rf.Capability("species")),
        splits_data={"train": train_batches, "val": val_batches},
    )
    # ...and DatasetFieldAdapter maps them cleanly to ("inputs", "targets")!
    dataset = rf.DatasetFieldAdapter(
        raw_dataset,
        field_map={"features": "inputs", "species": "targets"},
    )

    raw_model = BinaryMLP(in_dim=4, hidden_dim=8, seed=42)
    model = rf.ModelAdapter(raw_model, capabilities=(rf.Capability("forward"),))

    inst = rf.InstrumentationManager()
    inst.watch_probe("mlp/hidden", rf.tensor_stats, every_n_steps=5)
    inst.watch_probe(
        "mlp/hidden",
        lambda ev: rf.check_nan_inf(ev, raise_on_error=True),
    )

    experiment = rf.Experiment(
        name="Iris_Binary_Classification",
        task=rf.ClassificationTask(num_classes=2),
        method=IrisBinaryMethod(),
        model=model,
        dataset=dataset,
        objectives=BinaryCrossEntropyWithLogits(),
        optimizers=MomentumSGD(raw_model.parameters(), lr=0.05, momentum=0.8),
        evaluator=BinaryMetricsEvaluator(),
        hooks=[ProgressLogger(every=10), rf.InstrumentationHook(inst)],
        metadata={"dataset": "Fisher Iris (versicolor vs virginica)", "split": "80/20"},
    )

    report = experiment.validate()
    print(f"Contract validation passed: {report.is_valid}")

    run = experiment.run(max_epochs=50)
    print(f"\nRun {run.run_id} finished: status={run.status.value}")
    print(f"Best/Final val metrics: {run.eval_history[-1]}")
    print(f"Recorded instrumentation events: {len(run.artifacts['instrumentation'])}")

    # Export full .rfx archive and verify restoring weights into a fresh model.
    with tempfile.TemporaryDirectory() as tmpdir:
        rfx_path = experiment.export(Path(tmpdir) / "iris_binary.rfx", profile="full")
        imported = rf.load_experiment(rfx_path)
        fresh_model = BinaryMLP(in_dim=4, hidden_dim=8, seed=999)
        assert imported.restore_component_state("model", fresh_model)
        # Verify restored model reproduces exact same validation accuracy.
        eval_exp = rf.Experiment(
            task=rf.ClassificationTask(num_classes=2),
            method=IrisBinaryMethod(),
            model=rf.ModelAdapter(fresh_model, capabilities=(rf.Capability("forward"),)),
            dataset=dataset,
            objectives=BinaryCrossEntropyWithLogits(),
            evaluator=BinaryMetricsEvaluator(),
        )
        restored_metrics = eval_exp.evaluate(split="val")
        print(
            f"Exported .rfx ({rfx_path.stat().st_size} bytes) & restored model "
            f"val accuracy={restored_metrics['accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
