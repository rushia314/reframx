"""Model instrumentation: Probes, module watches, and diagnostics."""

from reframx.instrumentation.events import (
    InstrumentationEvent,
    InstrumentationRecord,
    Stage,
    detach_value,
)
from reframx.instrumentation.manager import (
    InstrumentationHook,
    InstrumentationManager,
    get_active_manager,
)
from reframx.instrumentation.module_hooks import WatchHandle
from reframx.instrumentation.probes import (
    Probe,
    check_nan_inf,
    probe,
    tensor_stats,
)

__all__ = [
    "InstrumentationEvent",
    "InstrumentationHook",
    "InstrumentationManager",
    "InstrumentationRecord",
    "Probe",
    "Stage",
    "WatchHandle",
    "check_nan_inf",
    "detach_value",
    "get_active_manager",
    "probe",
    "tensor_stats",
]
