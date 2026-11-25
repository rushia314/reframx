"""Optional PyTorch integration and built-in components for Research Framework."""

from reframx.integrations.pytorch.dataset import TorchDatasetAdapter
from reframx.integrations.pytorch.evaluators import (
    TorchAccuracyEvaluator,
    TorchRegressionEvaluator,
)
from reframx.integrations.pytorch.methods import (
    TorchClassificationMethod,
    TorchRegressionMethod,
    TorchSupervisedMethod,
)
from reframx.integrations.pytorch.model import TorchModelAdapter
from reframx.integrations.pytorch.objectives import (
    BCEWithLogitsObjective,
    CrossEntropyObjective,
    MSEObjective,
    TorchLossAdapter,
)
from reframx.integrations.pytorch.strategy import TorchStrategy
from reframx.integrations.pytorch.utils import (
    is_torch_available,
    move_to_device,
    require_torch,
    resolve_device,
)

__all__ = [
    "BCEWithLogitsObjective",
    "CrossEntropyObjective",
    "MSEObjective",
    "TorchAccuracyEvaluator",
    "TorchClassificationMethod",
    "TorchDatasetAdapter",
    "TorchLossAdapter",
    "TorchModelAdapter",
    "TorchRegressionEvaluator",
    "TorchRegressionMethod",
    "TorchStrategy",
    "TorchSupervisedMethod",
    "is_torch_available",
    "move_to_device",
    "require_torch",
    "resolve_device",
]
