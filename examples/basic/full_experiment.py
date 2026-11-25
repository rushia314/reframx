"""Complete runnable end-to-end experiment using Research Framework.

Demonstrates:
- Custom Task, Dataset, Model (wrapped via ModelAdapter), Objective, Evaluator, Method
- Real loss.backward() training of a 2-layer neural network on XOR (pure Python)
- Custom plugin Hook (progress logger)
- Full Experiment.validate() -> Experiment.run() lifecycle
"""

from __future__ import annotations

import math
import random

import reframx as rf

# ---------------------------------------------------------------------------
# Minimal pure-Python scalar autograd engine (so this example runs out of the
# box with zero external dependencies, using the same .backward() interface
# as PyTorch).
# ---------------------------------------------------------------------------


class Value:
    def __init__(self, data: float, _children: tuple[Value, ...] = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = _children
        self._backward = lambda: None

    def item(self) -> float:
        return self.data

    def __add__(self, other: Value | float) -> Value:
        other_v = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other_v.data, (self, other_v))

        def _backward() -> None:
            self.grad += out.grad
            other_v.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other_v = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other_v.data, (self, other_v))

        def _backward() -> None:
            self.grad += other_v.data * out.grad
            other_v.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __sub__(self, other: Value | float) -> Value:
        return self + (other * -1.0)

    def __truediv__(self, other: float) -> Value:
        return self * (1.0 / other)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,))

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
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


class SGDOptimizer:
    def __init__(self, params: list[Value], lr: float = 0.5) -> None:
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0.0


# ---------------------------------------------------------------------------
# User-defined research components (Model, Objective, Evaluator, Method, Hook)
# ---------------------------------------------------------------------------


class TinyMLP:
    """Plain user neural network; does not inherit from the framework."""

    def __init__(self, in_dim: int, hidden_dim: int, seed: int = 7) -> None:
        rng = random.Random(seed)
        self.w1 = [
            [Value(rng.uniform(-1.0, 1.0)) for _ in range(in_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [Value(0.0) for _ in range(hidden_dim)]
        self.w2 = [Value(rng.uniform(-1.0, 1.0)) for _ in range(hidden_dim)]
        self.b2 = Value(0.0)

    def parameters(self) -> list[Value]:
        return [w for row in self.w1 for w in row] + self.b1 + self.w2 + [self.b2]

    def forward(self, batch_x: list[list[float]]) -> list[Value]:
        outputs: list[Value] = []
        for x in batch_x:
            hidden: list[Value] = []
            for weights, bias in zip(self.w1, self.b1, strict=True):
                act = bias
                for wi, xi in zip(weights, x, strict=True):
                    act = act + wi * xi
                hidden.append(act.tanh())
            logit = self.b2
            for wi, hi in zip(self.w2, hidden, strict=True):
                logit = logit + wi * hi
            outputs.append(logit.sigmoid())
        return outputs


class BinaryMSELoss(rf.BaseObjective):
    def __init__(self) -> None:
        super().__init__(capabilities=(rf.Capability("binary_loss"),))

    def compute(self, probs: list[Value], targets: list[float]) -> Value:
        total: Value = Value(0.0)
        for p, y in zip(probs, targets, strict=True):
            diff = p - y
            total = total + diff * diff
        return total / float(len(probs))


class BinaryAccuracyEvaluator(rf.BaseEvaluator):
    def __init__(self) -> None:
        super().__init__(requirements=(rf.Requirement("class_probabilities"),))
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, result: rf.MethodResult, batch: dict[str, list]) -> None:
        probs: list[Value] = result.task_outputs["class_probabilities"]
        for p, y in zip(probs, batch["targets"], strict=True):
            pred = 1.0 if p.item() >= 0.5 else 0.0
            if pred == y:
                self.correct += 1
            self.total += 1

    def compute(self) -> dict[str, float]:
        return {"accuracy": self.correct / max(self.total, 1)}


class XORSupervisedMethod(rf.SupervisedMethod):
    def __init__(self) -> None:
        super().__init__(
            requirements=(
                rf.Requirement("inputs", provider_type="dataset"),
                rf.Requirement("targets", provider_type="dataset"),
                rf.Requirement("forward", provider_type="model"),
                rf.Requirement("binary_loss", provider_type="objective"),
            ),
            capabilities=(rf.Capability("class_probabilities"),),
        )

    def compute_step(
        self,
        batch: dict[str, list],
        ctx: rf.ExecutionContext,
    ) -> rf.MethodResult:
        probs = ctx.model.forward(batch["inputs"])
        loss = ctx.objective(probs, batch["targets"])
        return rf.MethodResult(
            task_outputs={"class_probabilities": probs},
            loss=loss,
        )


class ProgressPlugin(rf.BaseHook):
    def __init__(self, log_every: int = 50) -> None:
        self.log_every = log_every

    def on_epoch_end(self, ctx: rf.ExecutionContext, epoch_metrics) -> None:
        if ctx.state.epoch == 1 or ctx.state.epoch % self.log_every == 0:
            loss = epoch_metrics.get("train/loss", float("nan"))
            acc = epoch_metrics.get("eval/accuracy", float("nan"))
            print(
                f"Epoch {ctx.state.epoch:3d} | "
                f"train/loss={loss:.4f} | eval/accuracy={acc:.2%}"
            )


def main() -> None:
    x_xor = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    y_xor = [0.0, 1.0, 1.0, 0.0]

    dataset = rf.BaseDataset(
        capabilities=(rf.Capability("inputs"), rf.Capability("targets")),
        splits_data={
            "train": [{"inputs": x_xor, "targets": y_xor}],
            "val": [{"inputs": x_xor, "targets": y_xor}],
        },
    )

    raw_model = TinyMLP(in_dim=2, hidden_dim=6, seed=7)
    model = rf.ModelAdapter(
        raw_model,
        capabilities=(rf.Capability("forward"),),
        public_apis=("forward",),
    )

    task = rf.BaseTask(
        requirements=(rf.Requirement("class_probabilities"),),
        problem=rf.ProblemDefinition(name="XOR Classification"),
        evaluation=rf.EvaluationDefinition(primary_metrics=("accuracy",)),
    )

    experiment = rf.Experiment(
        name="XOR_Demo",
        task=task,
        method=XORSupervisedMethod(),
        model=model,
        dataset=dataset,
        objectives=BinaryMSELoss(),
        optimizers=SGDOptimizer(raw_model.parameters(), lr=1.5),
        evaluator=BinaryAccuracyEvaluator(),
        hooks=[ProgressPlugin(log_every=50)],
    )

    report = experiment.validate()
    print(f"Validation passed: {report.is_valid}")

    run = experiment.run(max_epochs=250)
    print(f"Run {run.run_id} finished with status={run.status.value}")
    print(f"Final eval metrics: {run.eval_history[-1]}")


if __name__ == "__main__":
    main()
