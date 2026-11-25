"""Unit and integration tests for the optional PyTorch integration.

Runs both:
- Deterministic tests with a lightweight fake ``torch`` module (always executed,
  even in environments where PyTorch is not installed).
- Real PyTorch end-to-end test (automatically executed when ``torch`` is
  present in the user's environment).
"""

from __future__ import annotations

import contextlib
import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest

import reframx as rf
from reframx.integrations.pytorch import (
    BCEWithLogitsObjective,
    CrossEntropyObjective,
    MSEObjective,
    TorchAccuracyEvaluator,
    TorchClassificationMethod,
    TorchDatasetAdapter,
    TorchLossAdapter,
    TorchModelAdapter,
    TorchRegressionEvaluator,
    TorchRegressionMethod,
    TorchStrategy,
    is_torch_available,
    move_to_device,
    require_torch,
    resolve_device,
)


def _make_fake_torch(*, cuda: bool = False, mps: bool = False) -> Any:
    """Construct a minimal fake ``torch`` module for dependency-free tests."""
    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, spec: str) -> None:
            self.type = spec.split(":")[0]
            self.spec = spec

        def __str__(self) -> str:
            return self.spec

        def __eq__(self, other: object) -> bool:
            return isinstance(other, FakeDevice) and self.spec == other.spec

    class FakeTensor:
        def __init__(
            self,
            value: float,
            *,
            requires_grad: bool = False,
            device: FakeDevice | None = None,
        ) -> None:
            self.value = float(value)
            self.requires_grad = requires_grad
            self.device = device or FakeDevice("cpu")
            self.grad: FakeTensor | None = None
            self.backward_calls = 0

        def numel(self) -> int:
            return 1

        def item(self) -> float:
            return self.value

        def to(self, device: Any, non_blocking: bool = False) -> FakeTensor:
            moved = FakeTensor(
                self.value,
                requires_grad=self.requires_grad,
                device=device if isinstance(device, FakeDevice) else FakeDevice(str(device)),
            )
            moved.grad = self.grad
            return moved

        def __truediv__(self, other: float) -> FakeTensor:
            return FakeTensor(
                self.value / other,
                requires_grad=self.requires_grad,
                device=self.device,
            )

        def backward(self) -> None:
            self.backward_calls += 1

    class FakeModule:
        def __init__(self) -> None:
            self._params: list[FakeTensor] = [
                FakeTensor(1.0, requires_grad=True),
                FakeTensor(2.0, requires_grad=False),
            ]
            self.device = FakeDevice("cpu")
            self.training = True
            self._state: dict[str, float] = {"w": 1.0}

        def parameters(self) -> list[FakeTensor]:
            return self._params

        def to(self, device: Any) -> FakeModule:
            self.device = device
            return self

        def train(self, mode: bool = True) -> FakeModule:
            self.training = mode
            return self

        def state_dict(self) -> dict[str, float]:
            return dict(self._state)

        def load_state_dict(self, state: dict[str, float]) -> None:
            self._state = dict(state)

        def forward(self, x: FakeTensor) -> FakeTensor:
            return FakeTensor(x.value * self._state["w"], requires_grad=True, device=x.device)

        def __call__(self, x: FakeTensor) -> FakeTensor:
            return self.forward(x)

    clipped_calls: list[tuple[int, float]] = []

    def clip_grad_norm_(params: list[Any], max_norm: float) -> float:
        clipped_calls.append((len(params), max_norm))
        return max_norm

    class FakeGradScaler:
        def __init__(self, device_type: str) -> None:
            self.device_type = device_type
            self.events: list[str] = []

        def scale(self, loss: Any) -> Any:
            self.events.append("scale")
            return loss

        def unscale_(self, optimizer: Any) -> None:
            self.events.append("unscale")

        def step(self, optimizer: Any) -> None:
            self.events.append("step")
            optimizer.step()

        def update(self) -> None:
            self.events.append("update")

    autocast_calls: list[tuple[str, Any]] = []
    inference_mode_calls = 0

    @contextlib.contextmanager
    def autocast(*, device_type: str, dtype: Any):
        autocast_calls.append((device_type, dtype))
        yield

    @contextlib.contextmanager
    def inference_mode():
        nonlocal inference_mode_calls
        inference_mode_calls += 1
        yield

    class FakeDataLoader:
        def __init__(self, dataset: Any, **kwargs: Any) -> None:
            self.dataset = dataset
            self.kwargs = kwargs

        def __iter__(self):
            return iter(self.dataset)

    fake_torch.device = FakeDevice  # type: ignore[attr-defined]
    fake_torch.Tensor = FakeTensor  # type: ignore[attr-defined]
    fake_torch.float16 = "float16"  # type: ignore[attr-defined]
    fake_torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    fake_torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    fake_torch.nn = types.SimpleNamespace(  # type: ignore[attr-defined]
        Module=FakeModule,
        utils=types.SimpleNamespace(clip_grad_norm_=clip_grad_norm_),
    )
    fake_torch.amp = types.SimpleNamespace(  # type: ignore[attr-defined]
        GradScaler=FakeGradScaler,
        autocast=autocast,
    )
    fake_torch.inference_mode = inference_mode  # type: ignore[attr-defined]
    fake_torch.utils = types.SimpleNamespace(  # type: ignore[attr-defined]
        data=types.SimpleNamespace(DataLoader=FakeDataLoader)
    )
    fake_torch._clipped_calls = clipped_calls  # type: ignore[attr-defined]
    fake_torch._autocast_calls = autocast_calls  # type: ignore[attr-defined]
    fake_torch._get_inference_mode_calls = lambda: inference_mode_calls  # type: ignore[attr-defined]
    return fake_torch


def test_require_torch_raises_helpful_error_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match="https://pytorch.org"):
        require_torch()


def test_resolve_device_and_move_to_device(monkeypatch):
    fake_torch = _make_fake_torch(cuda=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert str(resolve_device("auto")) == "cuda"
    assert str(resolve_device("cpu")) == "cpu"

    @dataclass
    class SampleBatch:
        x: Any
        meta: str

    batch = {
        "tensors": [fake_torch.Tensor(1.0), (fake_torch.Tensor(2.0), "keep")],
        "dc": SampleBatch(x=fake_torch.Tensor(3.0), meta="ok"),
    }
    moved = move_to_device(batch, fake_torch.device("cuda"))
    assert str(moved["tensors"][0].device) == "cuda"
    assert str(moved["tensors"][1][0].device) == "cuda"
    assert moved["tensors"][1][1] == "keep"
    assert str(moved["dc"].x.device) == "cuda"
    assert moved["dc"].meta == "ok"


def test_torch_model_adapter_and_state_provider(monkeypatch):
    fake_torch = _make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(TypeError, match="torch.nn.Module"):
        TorchModelAdapter(object())

    module = fake_torch.nn.Module()
    adapter = TorchModelAdapter(module)
    assert adapter.spec.attributes["framework"] == "pytorch"
    assert adapter.spec.attributes["total_parameters"] == 2
    assert adapter.spec.attributes["trainable_parameters"] == 1
    assert isinstance(adapter, rf.StateProvider)

    state = adapter.get_component_state()
    assert state == {"w": 1.0}
    adapter.load_component_state({"w": 5.0})
    assert adapter.get_component_state() == {"w": 5.0}


def test_torch_dataset_adapter_and_strategy_e2e(monkeypatch):
    fake_torch = _make_fake_torch(cuda=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    module = fake_torch.nn.Module()
    param = module.parameters()[0]
    param.grad = fake_torch.Tensor(0.5)

    class FakeOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"params": [param]}]
            self.steps = 0
            self.zero_grads = 0

        def step(self) -> None:
            self.steps += 1

        def zero_grad(self, *, set_to_none: bool = False) -> None:
            assert set_to_none is True
            self.zero_grads += 1

    optimizer = FakeOptimizer()
    strategy = TorchStrategy(
        device="cuda",
        precision="fp16",
        max_grad_norm=1.0,
        grad_accumulation_steps=1,
    )
    assert strategy.scaler is not None

    dataset = TorchDatasetAdapter(
        datasets={
            "train": [{"x": fake_torch.Tensor(2.0)}],
            "val": [{"x": fake_torch.Tensor(3.0)}],
        },
        capabilities=(rf.Capability("x"),),
    )

    class SimpleTorchMethod(rf.SupervisedMethod):
        def __init__(self) -> None:
            super().__init__(
                requirements=(
                    rf.Requirement("x", provider_type="dataset"),
                    rf.Requirement("forward", provider_type="model"),
                ),
                capabilities=(rf.Capability("out"),),
            )

        def compute_step(self, batch: dict[str, Any], ctx: rf.ExecutionContext) -> rf.MethodResult:
            assert str(batch["x"].device) == "cuda"
            with ctx.strategy.autocast_context():
                out = ctx.model(batch["x"])
            return rf.MethodResult(task_outputs={"out": out}, loss=out)

    experiment = rf.Experiment(
        task=rf.BaseTask(requirements=(rf.Requirement("out"),)),
        method=SimpleTorchMethod(),
        model=TorchModelAdapter(module),
        dataset=dataset,
        optimizers=optimizer,
        strategy=strategy,
    )

    run = experiment.run(max_epochs=2)
    assert run.status is rf.RunStatus.COMPLETED
    assert str(module.device) == "cuda"
    assert optimizer.steps == 2
    assert optimizer.zero_grads == 2
    assert strategy.scaler.events == [
        "scale",
        "unscale",
        "step",
        "update",
        "scale",
        "unscale",
        "step",
        "update",
    ]
    assert len(fake_torch._clipped_calls) == 2
    assert fake_torch._get_inference_mode_calls() == 2


@pytest.mark.skipif(not is_torch_available(), reason="torch is not installed")
def test_real_pytorch_experiment_end_to_end():
    import torch

    torch.manual_seed(0)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = 2.0 * x + 1.0

    raw_model = torch.nn.Linear(1, 1)
    model = TorchModelAdapter(raw_model)
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1)

    class MSE(rf.BaseObjective):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("mse"),))

        def compute(self, pred: Any, target: Any) -> Any:
            return torch.nn.functional.mse_loss(pred, target)

    class LinearMethod(rf.SupervisedMethod):
        def __init__(self) -> None:
            super().__init__(
                requirements=(
                    rf.Requirement("x", provider_type="dataset"),
                    rf.Requirement("y", provider_type="dataset"),
                    rf.Requirement("forward", provider_type="model"),
                    rf.Requirement("mse", provider_type="objective"),
                ),
                capabilities=(rf.Capability("prediction"),),
            )

        def compute_step(self, batch: dict[str, Any], ctx: rf.ExecutionContext) -> rf.MethodResult:
            pred = ctx.model(batch["x"])
            loss = ctx.objective(pred, batch["y"])
            return rf.MethodResult(task_outputs={"prediction": pred}, loss=loss)

    dataset = TorchDatasetAdapter(
        loaders={"train": [{"x": x, "y": y}], "val": [{"x": x, "y": y}]},
        capabilities=(rf.Capability("x"), rf.Capability("y")),
    )
    experiment = rf.Experiment(
        task=rf.BaseTask(requirements=(rf.Requirement("prediction"),)),
        method=LinearMethod(),
        model=model,
        dataset=dataset,
        objectives=MSE(),
        optimizers=optimizer,
        strategy=TorchStrategy(device="cpu", max_grad_norm=5.0),
    )

    run = experiment.run(max_epochs=20)
    assert run.status is rf.RunStatus.COMPLETED
    assert run.epoch_history[-1]["train/loss"] < run.epoch_history[0]["train/loss"]


def test_builtin_tasks_and_pytorch_builtins_with_tuple_and_dict_batches(monkeypatch):
    fake_torch = _make_fake_torch()

    def fake_ce(logits, targets, **kwargs):
        return fake_torch.Tensor(0.42, requires_grad=True)

    def fake_mse(preds, targets, **kwargs):
        return fake_torch.Tensor(0.25, requires_grad=True)

    def fake_bce(logits, targets, **kwargs):
        return fake_torch.Tensor(0.31, requires_grad=True)

    fake_torch.nn.functional = types.SimpleNamespace(
        cross_entropy=fake_ce,
        mse_loss=fake_mse,
        binary_cross_entropy_with_logits=fake_bce,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Verify built-in tasks exist and declare expected contracts.
    assert rf.CausalLMTask().spec.accepted_outputs[0].name == "token_logits"

    class DummyOpt:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

        def zero_grad(self, *, set_to_none: bool = False) -> None:
            pass

    # 1. Classification with (x, y) tuple batches + CrossEntropyObjective + TorchLossAdapter.
    cls_opt = DummyOpt()
    cls_exp = rf.Experiment(
        task=rf.ClassificationTask(num_classes=3),
        method=TorchClassificationMethod(),
        model=TorchModelAdapter(fake_torch.nn.Module()),
        dataset=TorchDatasetAdapter(
            loaders={"train": [(fake_torch.Tensor(1.0), fake_torch.Tensor(1.0))]},
        ),
        objectives=CrossEntropyObjective(label_smoothing=0.1),
        optimizers=cls_opt,
        strategy=TorchStrategy(device="cpu"),
    )
    cls_run = cls_exp.run(max_epochs=2, eval_split=None)
    assert cls_run.status is rf.RunStatus.COMPLETED
    assert cls_opt.steps == 2
    assert cls_run.train_history[-1]["loss"] == pytest.approx(0.42)

    # 2. Regression with dict batches + MSEObjective + BCEWithLogitsObjective + TorchLossAdapter.
    reg_opt = DummyOpt()
    reg_exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=TorchRegressionMethod(),
        model=TorchModelAdapter(fake_torch.nn.Module()),
        dataset=TorchDatasetAdapter(
            loaders={
                "train": [
                    {
                        "inputs": fake_torch.Tensor(2.0),
                        "targets": fake_torch.Tensor(4.0),
                    }
                ]
            },
        ),
        objectives=MSEObjective(),
        optimizers=reg_opt,
        strategy=TorchStrategy(device="cpu"),
    )
    reg_run = reg_exp.run(max_epochs=1, eval_split=None)
    assert reg_run.status is rf.RunStatus.COMPLETED
    assert reg_run.train_history[0]["loss"] == pytest.approx(0.25)

    bce = BCEWithLogitsObjective()
    assert bce(fake_torch.Tensor(0.0), fake_torch.Tensor(1.0)).item() == pytest.approx(0.31)

    wrapped_loss = TorchLossAdapter(lambda pred, target: fake_torch.Tensor(0.11))
    loss_val = wrapped_loss(fake_torch.Tensor(1.0), fake_torch.Tensor(1.0)).item()
    assert loss_val == pytest.approx(0.11)

    # 3. Verify built-in evaluators contract compatibility.
    acc_eval = TorchAccuracyEvaluator()
    reg_eval = TorchRegressionEvaluator()
    assert acc_eval.spec.requirements[0].name == "logits"
    assert reg_eval.spec.requirements[0].name == "predictions"
