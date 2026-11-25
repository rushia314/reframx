"""Unit and integration tests for model instrumentation (Probes & module hooks)."""

from __future__ import annotations

from typing import Any

import pytest

import reframx as rf
from reframx.instrumentation import get_active_manager


class FakeTrackedTensor:
    """Minimal tensor-like object tracking whether ``.detach()`` was called."""

    def __init__(self, values: list[float], *, detached: bool = False) -> None:
        self.values = list(values)
        self.detached = detached

    def detach(self) -> FakeTrackedTensor:
        return FakeTrackedTensor(self.values, detached=True)

    def __iter__(self):
        return iter(self.values)


def test_probe_is_zero_overhead_passthrough_when_inactive():
    x = FakeTrackedTensor([1.0, 2.0])
    assert rf.probe("attn/scores", x) is x
    p = rf.Probe("memory/delta")
    assert p(x) is x


def test_probe_detaches_for_callback_tracks_call_index_and_sampling():
    seen_events: list[rf.InstrumentationEvent] = []
    manager = rf.InstrumentationManager()
    manager.watch_probe(
        "attn/*",
        lambda ev: (seen_events.append(ev), rf.tensor_stats(ev))[1],
        every_n_steps=2,
        phases=("train",),
    )

    with manager.activate(epoch=1, step=1, phase="train"):
        raw = FakeTrackedTensor([3.0, 4.0])
        returned = rf.probe("attn/scores", raw)
        assert returned is raw
        assert returned.detached is False
        assert len(seen_events) == 0  # step 1 skipped by every_n_steps=2

    with manager.activate(epoch=1, step=2, phase="train"):
        raw_a = FakeTrackedTensor([3.0, 4.0])
        raw_b = FakeTrackedTensor([6.0, 8.0])
        assert rf.probe("attn/scores", raw_a) is raw_a
        assert rf.probe("attn/scores", raw_b) is raw_b
        assert rf.probe("other/ignored", raw_a) is raw_a

    assert len(seen_events) == 2
    assert seen_events[0].value.detached is True
    assert seen_events[0].call_index == 0
    assert seen_events[1].call_index == 1
    assert manager.records[0].metrics["l2_norm"] == pytest.approx(5.0)
    assert manager.records[1].metrics["l2_norm"] == pytest.approx(10.0)

    # Eval phase is filtered out.
    with manager.activate(epoch=1, step=2, phase="eval"):
        rf.probe("attn/scores", FakeTrackedTensor([1.0]))
    assert len(seen_events) == 2


def test_watch_module_and_watch_modules_with_cleanup():
    class FakeHookHandle:
        def __init__(self, registry: list[Any], fn: Any) -> None:
            self._registry = registry
            self._fn = fn

        def remove(self) -> None:
            if self._fn in self._registry:
                self._registry.remove(self._fn)

    class FakeModule:
        def __init__(self, name: str) -> None:
            self.name = name
            self.pre_hooks: list[Any] = []
            self.post_hooks: list[Any] = []
            self.bwd_hooks: list[Any] = []
            self.children: dict[str, FakeModule] = {}

        def register_forward_pre_hook(self, fn: Any, *, with_kwargs: bool = False):
            self.pre_hooks.append(fn)
            return FakeHookHandle(self.pre_hooks, fn)

        def register_forward_hook(self, fn: Any, *, with_kwargs: bool = False):
            self.post_hooks.append(fn)
            return FakeHookHandle(self.post_hooks, fn)

        def register_full_backward_hook(self, fn: Any):
            self.bwd_hooks.append(fn)
            return FakeHookHandle(self.bwd_hooks, fn)

        def named_modules(self):
            yield "", self
            for cname, child in self.children.items():
                yield cname, child

        def run(self, x: FakeTrackedTensor) -> dict[str, FakeTrackedTensor]:
            for fn in self.pre_hooks:
                fn(self, (x,), {})
            out = {"hidden": FakeTrackedTensor([v * 2.0 for v in x.values])}
            for fn in self.post_hooks:
                fn(self, (x,), {}, out)
            for fn in self.bwd_hooks:
                fn(self, (x,), (out["hidden"],))
            return out

    root = FakeModule("root")
    attn = FakeModule("attn")
    mlp = FakeModule("mlp")
    root.children = {"layers.0.self_attn": attn, "layers.0.mlp": mlp}

    stages_seen: list[tuple[str, str, bool]] = []
    manager = rf.InstrumentationManager()
    handles = manager.watch_modules(
        root,
        pattern="*.self_attn",
        stages=("pre_forward", "post_forward", "backward"),
        callback=lambda ev: (
            stages_seen.append(
                (
                    ev.name,
                    ev.stage,
                    (
                        ev.output["hidden"].detached
                        if isinstance(ev.output, dict)
                        else ev.value[0].detached
                    ),
                )
            ),
            {"seen": 1},
        )[1],
    )
    assert len(handles) == 1
    assert len(mlp.post_hooks) == 0

    attn.run(FakeTrackedTensor([1.0, 2.0]))
    assert stages_seen == [
        ("layers.0.self_attn", "pre_forward", True),
        ("layers.0.self_attn", "post_forward", True),
        ("layers.0.self_attn", "backward", True),
    ]

    manager.close()
    assert attn.pre_hooks == []
    assert attn.post_hooks == []
    assert attn.bwd_hooks == []


def test_check_nan_inf_detects_and_optionally_raises():
    ev_ok = rf.InstrumentationEvent(name="x", stage="probe", value=[1.0, 2.0])
    assert rf.check_nan_inf(ev_ok) == {"has_nan": False, "has_inf": False}

    ev_nan = rf.InstrumentationEvent(name="x", stage="probe", value=[1.0, float("nan")])
    assert rf.check_nan_inf(ev_nan) == {"has_nan": True, "has_inf": False}
    with pytest.raises(FloatingPointError, match="Non-finite"):
        rf.check_nan_inf(ev_nan, raise_on_error=True)


def test_instrumentation_hook_integrates_probes_into_experiment_run():
    class ProbedModel(rf.BaseModel):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("forward"),))
            self.hidden_probe = rf.Probe("mlp/hidden")

        def __call__(self, x: float) -> float:
            h = self.hidden_probe([x, x * 2.0])
            return sum(h)

    class SimpleMethod(rf.BaseMethod):
        def __init__(self) -> None:
            super().__init__(capabilities=(rf.Capability("predictions"),))

        def train_step(
            self,
            batch: dict[str, float],
            ctx: rf.ExecutionContext,
        ) -> rf.MethodResult:
            pred = ctx.model(batch["x"])
            return rf.MethodResult(
                task_outputs={"predictions": pred},
                metrics={"loss": abs(pred - batch["y"])},
            )

    manager = rf.InstrumentationManager()
    manager.watch_probe("mlp/*", rf.tensor_stats)

    exp = rf.Experiment(
        task=rf.RegressionTask(),
        method=SimpleMethod(),
        model=ProbedModel(),
        dataset=rf.BaseDataset(
            splits_data={"train": [{"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 6.0}]}
        ),
        hooks=[rf.InstrumentationHook(manager)],
    )

    run = exp.run(max_epochs=1, eval_split=None)
    assert run.status is rf.RunStatus.COMPLETED
    assert len(run.artifacts["instrumentation"]) == 2
    assert "mlp/hidden/mean" in run.train_history[0]
    assert run.train_history[0]["mlp/hidden/mean"] == pytest.approx(1.5)
    assert run.train_history[1]["mlp/hidden/mean"] == pytest.approx(3.0)
    # Manager auto-closed and ContextVar restored after run.
    assert get_active_manager() is None
