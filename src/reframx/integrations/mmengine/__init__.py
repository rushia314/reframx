"""Optional MMEngine execution backend for Research Framework."""

from reframx.integrations.mmengine.backend import MMEngineBackend
from reframx.integrations.mmengine.strategy import MMEngineStrategy
from reframx.integrations.mmengine.utils import (
    is_mmengine_available,
    require_mmengine,
)

__all__ = [
    "MMEngineBackend",
    "MMEngineStrategy",
    "is_mmengine_available",
    "require_mmengine",
]
