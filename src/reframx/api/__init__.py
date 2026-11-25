"""Stable, user-facing symbols for the Research Framework public API."""

from reframx.core.components.handle import ComponentHandle
from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.protocols import (
    ComponentValidator,
    ConfigProvider,
    SpecProvider,
    StateProvider,
)
from reframx.core.components.specification import ComponentSpec
from reframx.core.contracts.capabilities import Capability
from reframx.core.contracts.compatibility import validate_requirements
from reframx.core.contracts.io import IOContract
from reframx.core.contracts.requirements import Requirement, RequirementGroup
from reframx.core.exceptions import (
    ComponentIntegrationError,
    InvalidSpecificationError,
    PluginLoadError,
    ResearchFrameworkError,
    ValidationFailedError,
    ValidatorContractError,
)
from reframx.core.registry import (
    DEFAULT_ENTRY_POINT_GROUP,
    ComponentRegistry,
    DiscoveredPlugin,
    LoadedPlugin,
    PluginManager,
    PluginManifest,
    PluginProtocol,
    RegistrationEntry,
    default_plugin_manager,
    default_registry,
)
from reframx.core.validation.context import ValidationContext
from reframx.core.validation.report import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
)
from reframx.core.validation.runner import run_component_validator
from reframx.datasets import (
    BaseDataset,
    DatasetFieldAdapter,
    DatasetProtocol,
    DatasetSpec,
)
from reframx.evaluators import (
    BaseEvaluator,
    EvaluatorProtocol,
    EvaluatorSpec,
)
from reframx.execution import (
    BaseHook,
    ExecutionBackend,
    ExecutionContext,
    ExecutionEngine,
    ExecutionState,
    NativeBackend,
)
from reframx.experiments import Experiment, Run, RunStatus
from reframx.instrumentation import (
    InstrumentationEvent,
    InstrumentationHook,
    InstrumentationManager,
    InstrumentationRecord,
    Probe,
    WatchHandle,
    check_nan_inf,
    probe,
    tensor_stats,
)
from reframx.methods import (
    BaseMethod,
    MethodResult,
    MethodSpec,
    SupervisedMethod,
)
from reframx.models import BaseModel, ModelAdapter, ModelSpec
from reframx.objectives import BaseObjective, ObjectiveSpec
from reframx.packaging import (
    RFX_EXTENSION,
    ArchiveError,
    ArchiveManifest,
    ExportProfile,
    ImportedExperiment,
    export_experiment,
    load_experiment,
)
from reframx.strategies import (
    BaseStrategy,
    StandardBackwardStrategy,
    StrategySpec,
)
from reframx.tasks import (
    BaseTask,
    CausalLMTask,
    ClassificationTask,
    EvaluationDefinition,
    ObjectiveGoal,
    ProblemDefinition,
    RegressionTask,
    TaskSpec,
)

__all__ = [
    "ArchiveError",
    "ArchiveManifest",
    "BaseDataset",
    "BaseEvaluator",
    "BaseHook",
    "BaseMethod",
    "BaseModel",
    "BaseObjective",
    "BaseStrategy",
    "BaseTask",
    "Capability",
    "CausalLMTask",
    "ClassificationTask",
    "ComponentHandle",
    "ComponentIntegrationError",
    "ComponentMetadata",
    "ComponentRegistry",
    "ComponentSpec",
    "ComponentValidator",
    "ConfigProvider",
    "DEFAULT_ENTRY_POINT_GROUP",
    "DatasetFieldAdapter",
    "DatasetProtocol",
    "DatasetSpec",
    "DiscoveredPlugin",
    "EvaluationDefinition",
    "EvaluatorProtocol",
    "EvaluatorSpec",
    "ExecutionBackend",
    "ExecutionContext",
    "ExecutionEngine",
    "ExecutionState",
    "Experiment",
    "ExportProfile",
    "IOContract",
    "ImportedExperiment",
    "InstrumentationEvent",
    "InstrumentationHook",
    "InstrumentationManager",
    "InstrumentationRecord",
    "InvalidSpecificationError",
    "LoadedPlugin",
    "MethodResult",
    "MethodSpec",
    "ModelAdapter",
    "ModelSpec",
    "NativeBackend",
    "ObjectiveGoal",
    "ObjectiveSpec",
    "PluginLoadError",
    "PluginManager",
    "PluginManifest",
    "PluginProtocol",
    "Probe",
    "ProblemDefinition",
    "RFX_EXTENSION",
    "RegistrationEntry",
    "RegressionTask",
    "Requirement",
    "RequirementGroup",
    "ResearchFrameworkError",
    "Run",
    "RunStatus",
    "SpecProvider",
    "StandardBackwardStrategy",
    "StateProvider",
    "StrategySpec",
    "SupervisedMethod",
    "TaskSpec",
    "ValidationContext",
    "ValidationFailedError",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "ValidatorContractError",
    "WatchHandle",
    "check_nan_inf",
    "default_plugin_manager",
    "default_registry",
    "export_experiment",
    "load_experiment",
    "probe",
    "run_component_validator",
    "tensor_stats",
    "validate_requirements",
]
