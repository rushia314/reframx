"""End-to-end integration tests for runnable experiments."""

from __future__ import annotations

import pytest

import reframx as rf


class ScalarVar:
    """Tiny scalar autograd variable for dependency-free backward tests."""

    def __init__(self, value: float, *, requires_grad: bool = True) -> None:
        self.value = float(value)
        self.grad = 0.0
        self.requires_grad = requires_grad
        self._backward = lambda: None

    def item(self) -> float:
        return self.value

    def __truediv__(self, denom: float) -> ScalarVar:
        out = ScalarVar(self.value / denom, requires_grad=self.requires_grad)

        def _bwd() -> None:
            self.grad += out.grad / denom
            self.backward()

        out._backward = _bwd
        return out

    def backward(self) -> None:
        self._backward()


class SimpleOptimizer:
    def __init__(self, params: list[ScalarVar], lr: float = 0.1) -> None:
        self.params = params
        self.lr = lr
        self.step_count = 0
        self.zero_grad_count = 0

    def step(self) -> None:
        self.step_count += 1
        for param in self.params:
            param.value -= self.lr * param.grad

    def zero_grad(self) -> None:
        self.zero_grad_count += 1
        for param in self.params:
            param.grad = 0.0


class LinearScalarModel:
    """Plain Python model (no framework inheritance) wrapped via ModelAdapter."""

    def __init__(self, w: float = 0.0) -> None:
        self.w = ScalarVar(w)
        self.training = True

    def train(self, mode: bool = True) -> None:
        self.training = mode

    def predict(self, x: float) -> ScalarVar:
        out = ScalarVar(self.w.value * x)

        def _bwd() -> None:
            self.w.grad += out.grad * x

        out._backward = _bwd
        return out


class SquaredErrorObjective(rf.BaseObjective):
    def __init__(self) -> None:
        super().__init__(capabilities=(rf.Capability("mse_loss"),))

    def compute(self, pred: ScalarVar, target: float) -> ScalarVar:
        diff = pred.value - target
        loss = ScalarVar(diff * diff)

        def _bwd() -> None:
            pred.grad += 2.0 * diff
            pred.backward()

        loss._backward = _bwd
        return loss


class MeanAbsErrorEvaluator(rf.BaseEvaluator):
    def __init__(self) -> None:
        super().__init__(requirements=(rf.Capability("prediction").name,) and ())
        self._errors: list[float] = []

    def reset(self) -> None:
        self._errors.clear()

    def update(self, result: rf.MethodResult, batch: dict[str, float]) -> None:
        pred = result.task_outputs["prediction"].item()
        self._errors.append(abs(pred - batch["y"]))

    def compute(self) -> dict[str, float]:
        if not self._errors:
            return {"mae": 0.0}
        return {"mae": sum(self._errors) / len(self._errors)}


class RegressionSupervisedMethod(rf.SupervisedMethod):
    def __init__(self) -> None:
        super().__init__(
            requirements=(
                rf.Requirement("x", provider_type="dataset"),
                rf.Requirement("y", provider_type="dataset"),
                rf.Requirement("predict", provider_type="model"),
                rf.Requirement("mse_loss", provider_type="objective"),
            ),
            capabilities=(rf.Capability("prediction"),),
        )

    def compute_step(
        self,
        batch: dict[str, float],
        ctx: rf.ExecutionContext,
    ) -> rf.MethodResult:
        pred = ctx.model.predict(batch["x"])
        loss = ctx.objective(pred, batch["y"])
        return rf.MethodResult(
            task_outputs={"prediction": pred},
            loss=loss,
        )


def test_supervised_experiment_end_to_end_trains_and_evaluates():
    raw_model = LinearScalarModel(w=0.0)
    model = rf.ModelAdapter(raw_model, capabilities=(rf.Capability("predict"),))
    optimizer = SimpleOptimizer([raw_model.w], lr=0.1)

    dataset = rf.BaseDataset(
        capabilities=(rf.Capability("x"), rf.Capability("y")),
        splits_data={
            "train": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}],
            "val": [{"x": 3.0, "y": 6.0}],
        },
    )
    task = rf.BaseTask(requirements=(rf.Requirement("prediction"),))

    events: list[str] = []

    class TraceHook(rf.BaseHook):
        def on_run_start(self, ctx: rf.ExecutionContext) -> None:
            events.append("run_start")

        def on_epoch_end(self, ctx: rf.ExecutionContext, epoch_metrics) -> None:
            events.append(f"epoch_{ctx.state.epoch}")

        def on_run_end(self, ctx: rf.ExecutionContext, run: rf.Run) -> None:
            events.append("run_end")

    experiment = rf.Experiment(
        name="LinearRegression",
        task=task,
        method=RegressionSupervisedMethod(),
        model=model,
        dataset=dataset,
        objectives=SquaredErrorObjective(),
        optimizers=optimizer,
        evaluator=MeanAbsErrorEvaluator(),
        hooks=[TraceHook()],
    )

    report = experiment.validate()
    assert report.is_valid

    run = experiment.run(max_epochs=6)
    assert run.status is rf.RunStatus.COMPLETED
    assert len(run.epoch_history) == 6
    assert len(run.eval_history) == 6
    assert run.epoch_history[-1]["train/loss"] < run.epoch_history[0]["train/loss"]
    assert run.eval_history[-1]["mae"] < run.eval_history[0]["mae"]
    assert abs(raw_model.w.value - 2.0) < 0.05
    assert events[0] == "run_start"
    assert events[-1] == "run_end"

    eval_only = experiment.evaluate(split="val")
    assert eval_only["mae"] < 0.2


def test_custom_manual_method_with_multiple_optimizers_and_backwards():
    model_a = LinearScalarModel(w=0.0)
    model_b = LinearScalarModel(w=0.0)
    opt_a = SimpleOptimizer([model_a.w], lr=0.1)
    opt_b = SimpleOptimizer([model_b.w], lr=0.1)

    class DualOptimizerMethod(rf.BaseMethod):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("dual_output"),))

        def train_step(
            self,
            batch: dict[str, float],
            ctx: rf.ExecutionContext,
        ) -> rf.MethodResult:
            loss_fn = ctx.objective

            pred_a = ctx.model["a"].predict(batch["x"])
            loss_a = loss_fn(pred_a, batch["ya"])
            ctx.backward(loss_a)
            ctx.step("opt_a")

            pred_b = ctx.model["b"].predict(batch["x"])
            loss_b = loss_fn(pred_b, batch["yb"])
            ctx.backward(loss_b)
            ctx.step("opt_b")

            return rf.MethodResult(
                task_outputs={"dual_output": (pred_a, pred_b)},
                objectives={"loss_a": loss_a, "loss_b": loss_b},
                metrics={"loss_a": loss_a, "loss_b": loss_b},
            )

    experiment = rf.Experiment(
        name="DualOptim",
        task=rf.BaseTask(requirements=(rf.Requirement("dual_output"),)),
        method=DualOptimizerMethod(),
        model={"a": model_a, "b": model_b},
        dataset=rf.BaseDataset(
            splits_data={"train": [{"x": 1.0, "ya": 3.0, "yb": -2.0}]}
        ),
        objectives=SquaredErrorObjective(),
        optimizers={"opt_a": opt_a, "opt_b": opt_b},
    )

    run = experiment.run(max_epochs=15, eval_split=None)
    assert run.status is rf.RunStatus.COMPLETED
    assert opt_a.step_count == 15
    assert opt_b.step_count == 15
    assert abs(model_a.w.value - 3.0) < 0.2
    assert abs(model_b.w.value - (-2.0)) < 0.2


def test_grad_accumulation_and_early_stopping_hook():
    raw_model = LinearScalarModel(w=0.0)
    model = rf.ModelAdapter(raw_model, capabilities=(rf.Capability("predict"),))
    optimizer = SimpleOptimizer([raw_model.w], lr=0.1)

    class EarlyStopAfter3Steps:
        """Plain object hook (no BaseHook inheritance)."""

        def on_step_end(self, ctx: rf.ExecutionContext, batch, result) -> None:
            if ctx.state.global_step >= 3:
                ctx.request_stop()

    experiment = rf.Experiment(
        task=rf.BaseTask(requirements=(rf.Requirement("prediction"),)),
        method=RegressionSupervisedMethod(),
        model=model,
        dataset=rf.BaseDataset(
            capabilities=(rf.Capability("x"), rf.Capability("y")),
            splits_data={
                "train": [
                    {"x": 1.0, "y": 2.0},
                    {"x": 1.0, "y": 2.0},
                    {"x": 1.0, "y": 2.0},
                    {"x": 1.0, "y": 2.0},
                ]
            },
        ),
        objectives=SquaredErrorObjective(),
        optimizers=optimizer,
        strategy=rf.StandardBackwardStrategy(grad_accumulation_steps=2),
        hooks=[EarlyStopAfter3Steps()],
    )

    run = experiment.run(max_epochs=5, eval_split=None)
    assert len(run.train_history) == 3
    assert optimizer.step_count == 1


def test_experiment_fails_fast_on_contract_or_custom_validation_error():
    task = rf.BaseTask(requirements=(rf.Requirement("nonexistent_output"),))

    class FailingCustomModel(rf.BaseModel):
        def validate_component(self, context: rf.ValidationContext) -> rf.ValidationReport:
            return rf.ValidationReport(
                issues=[
                    rf.ValidationIssue(
                        severity=rf.ValidationSeverity.ERROR,
                        code="BAD_MODEL_CONFIG",
                        message="Custom model validation failed.",
                    )
                ]
            )

    experiment = rf.Experiment(
        task=task,
        method=RegressionSupervisedMethod(),
        model=FailingCustomModel(),
        dataset=rf.BaseDataset(splits_data={"train": []}),
    )

    with pytest.raises(rf.ValidationFailedError) as exc_info:
        experiment.run(max_epochs=1)

    assert exc_info.value.report is not None
    error_codes = {issue.code for issue in exc_info.value.report.errors}
    assert "REQUIRED_CAPABILITY_MISSING" in error_codes
    assert "BAD_MODEL_CONFIG" in error_codes


def test_base_method_without_eval_step_skips_eval_even_when_val_split_exists():
    raw_model = LinearScalarModel(w=0.0)
    opt = SimpleOptimizer([raw_model.w], lr=0.1)

    class TrainOnlyMethod(rf.BaseMethod):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("prediction"),))

        def train_step(
            self,
            batch: dict[str, float],
            ctx: rf.ExecutionContext,
        ) -> rf.MethodResult:
            pred = ctx.model.predict(batch["x"])
            loss = ctx.objective(pred, batch["y"])
            ctx.backward(loss)
            ctx.step()
            return rf.MethodResult(task_outputs={"prediction": pred}, loss=loss)

    experiment = rf.Experiment(
        task=rf.BaseTask(requirements=(rf.Requirement("prediction"),)),
        method=TrainOnlyMethod(),
        model=rf.ModelAdapter(raw_model),
        dataset=rf.BaseDataset(
            splits_data={
                "train": [{"x": 1.0, "y": 2.0}],
                "val": [{"x": 2.0, "y": 4.0}],
            }
        ),
        objectives=SquaredErrorObjective(),
        optimizers=opt,
    )

    run = experiment.run(max_epochs=2)
    assert run.status is rf.RunStatus.COMPLETED
    assert len(run.train_history) == 2
    assert run.eval_history == []
    assert experiment.evaluate(split="val") == {}


def test_catch_interrupt_marks_interrupted_and_allows_export(tmp_path):
    raw_model = LinearScalarModel(w=0.0)
    opt = SimpleOptimizer([raw_model.w], lr=0.1)

    class InterruptOnStep2Hook(rf.BaseHook):
        def on_step_end(self, ctx: rf.ExecutionContext, batch, result) -> None:
            if ctx.state.global_step == 2:
                raise KeyboardInterrupt

    exp = rf.Experiment(
        name="Interruptible",
        task=rf.BaseTask(requirements=(rf.Requirement("prediction"),)),
        method=RegressionSupervisedMethod(),
        model=rf.ModelAdapter(raw_model, capabilities=(rf.Capability("predict"),)),
        dataset=rf.BaseDataset(
            capabilities=(rf.Capability("x"), rf.Capability("y")),
            splits_data={"train": [{"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0}]},
        ),
        objectives=SquaredErrorObjective(),
        optimizers=opt,
        hooks=[InterruptOnStep2Hook()],
    )

    # With catch_interrupt=True, returns Run(status=INTERRUPTED) and export works cleanly.
    run = exp.run(max_epochs=5, eval_split=None, catch_interrupt=True)
    assert run.status is rf.RunStatus.INTERRUPTED
    assert len(run.train_history) == 2

    rfx_path = exp.export(tmp_path / "interrupted.rfx", profile="standard")
    imported = rf.load_experiment(rfx_path)
    assert imported.runs[-1]["status"] == "interrupted"

    # With catch_interrupt=False (default), KeyboardInterrupt still propagates.
    with pytest.raises(KeyboardInterrupt):
        exp.run(max_epochs=5, eval_split=None, catch_interrupt=False)
    assert exp.runs[-1].status is rf.RunStatus.INTERRUPTED


def test_experiment_stop_method_stops_training_gracefully():
    raw_model = LinearScalarModel(w=0.0)
    opt = SimpleOptimizer([raw_model.w], lr=0.1)
    exp_ref: list[rf.Experiment] = []

    class StopViaExperimentHook(rf.BaseHook):
        def on_step_end(self, ctx: rf.ExecutionContext, batch, result) -> None:
            if ctx.state.global_step == 2:
                assert exp_ref[0].stop("user_clicked_stop") is True

    exp = rf.Experiment(
        name="Stoppable",
        task=rf.BaseTask(requirements=(rf.Requirement("prediction"),)),
        method=RegressionSupervisedMethod(),
        model=rf.ModelAdapter(raw_model, capabilities=(rf.Capability("predict"),)),
        dataset=rf.BaseDataset(
            capabilities=(rf.Capability("x"), rf.Capability("y")),
            splits_data={"train": [{"x": 1.0, "y": 2.0}] * 5},
        ),
        objectives=SquaredErrorObjective(),
        optimizers=opt,
        hooks=[StopViaExperimentHook()],
    )
    exp_ref.append(exp)

    assert exp.stop() is False  # Not running yet
    run = exp.run(max_epochs=10, eval_split=None)
    assert run.status is rf.RunStatus.INTERRUPTED
    assert run.error == "user_clicked_stop"
    assert len(run.train_history) == 2
    assert exp.stop() is False  # Already finished


def test_step_based_training_cycles_epochs_and_runs_eval_every_steps():
    raw_model = LinearScalarModel(w=0.0)
    opt = SimpleOptimizer([raw_model.w], lr=0.1)

    train_modes_seen: list[bool] = []

    class ModeCheckHook(rf.BaseHook):
        def on_step_start(self, ctx: rf.ExecutionContext, batch) -> None:
            train_modes_seen.append(raw_model.training)

    exp = rf.Experiment(
        name="StepBased",
        task=rf.BaseTask(requirements="prediction"),
        method=RegressionSupervisedMethod(),
        model=rf.ModelAdapter(raw_model, capabilities="predict"),
        dataset=rf.BaseDataset(
            capabilities=["x", "y"],
            splits_data={
                # Only 2 batches per epoch -> 5 steps requires cycling 3 epochs!
                "train": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}],
                "val": [{"x": 3.0, "y": 6.0}],
            },
        ),
        objectives=SquaredErrorObjective(),
        optimizers=opt,
        evaluator=MeanAbsErrorEvaluator(),
        hooks=[ModeCheckHook()],
    )

    run = exp.run(max_steps=5, eval_every_steps=2)
    assert run.status is rf.RunStatus.COMPLETED
    assert len(run.train_history) == 5
    # Eval ran strictly at steps 2 and 4 (mid-epoch & end-of-epoch).
    assert [entry["step"] for entry in run.eval_history] == [2, 4]
    # Model was always restored back to train mode after mid-epoch eval.
    assert train_modes_seen == [True, True, True, True, True]
    assert run.eval_history[-1]["mae"] < run.eval_history[0]["mae"]

    # Empty train split in step-based mode raises instead of infinite-looping.
    empty_exp = rf.Experiment(
        task=rf.BaseTask(requirements="prediction"),
        method=RegressionSupervisedMethod(),
        model=rf.ModelAdapter(LinearScalarModel(), capabilities="predict"),
        dataset=rf.BaseDataset(capabilities=["x", "y"], splits_data={"train": []}),
        objectives=SquaredErrorObjective(),
        optimizers=SimpleOptimizer([]),
    )
    with pytest.raises(ValueError, match="yielded 0 batches"):
        empty_exp.run(max_steps=3, eval_split=None)
