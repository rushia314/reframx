# API Reference

## 1. Public Namespace (`import reframx as rf`)

### Core Components & Contracts
| Symbol | Description |
|---|---|
| `ComponentMetadata` | Identity metadata (`name`, `component_type`, `version`, `api_version`, `tags`, `source`). |
| `ComponentSpec` | Base specification holding `metadata`, `requirements`, `capabilities`, `attributes`. |
| `ComponentHandle` | Non-invasive wrapper (`ComponentHandle.wrap(obj, role=..., spec=...)`). |
| `SpecProvider`, `ConfigProvider`, `ComponentValidator`, `StateProvider` | Optional runtime-checkable protocols. |
| `IOContract` | Input/output schema (`kind`, `value_type`, `semantics`, `dtype`, `shape`, `attributes`). |
| `Requirement`, `RequirementGroup` | Consumer needs (`all_of`, `any_of`, optional `provider_type` filter). |
| `Capability` | Provider offering with optional `IOContract`. |
| `validate_requirements(consumer, providers, *, strict_contract=False)` | Generic compatibility checker returning `ValidationReport`. |
| `ValidationContext`, `ValidationIssue`, `ValidationReport`, `ValidationSeverity`, `run_component_validator` | Unified validation reporting and custom validator runner. |

### Domain Components
| Group | Symbols |
|---|---|
| **Tasks** | `TaskSpec`, `BaseTask`, `ProblemDefinition`, `ObjectiveGoal`, `EvaluationDefinition`, `ClassificationTask`, `RegressionTask`, `CausalLMTask` |
| **Models** | `ModelSpec`, `BaseModel`, `ModelAdapter` |
| **Datasets** | `DatasetSpec`, `DatasetProtocol`, `BaseDataset`, `DatasetFieldAdapter` |
| **Objectives** | `ObjectiveSpec`, `BaseObjective` |
| **Evaluators** | `EvaluatorSpec`, `EvaluatorProtocol`, `BaseEvaluator` |
| **Methods** | `MethodSpec`, `MethodResult`, `BaseMethod`, `SupervisedMethod` |
| **Strategies** | `StrategySpec`, `BaseStrategy`, `StandardBackwardStrategy` |

### Execution, Experiments & Hooks
| Symbol | Description |
|---|---|
| `Experiment` | Composes components; provides `validate()`, `run(..., catch_interrupt=False)`, `stop(reason=...)`, `evaluate()`, `export()`. |
| `Run`, `RunStatus` | Execution record (`run_id`, `status`: `CREATED`/`RUNNING`/`COMPLETED`/`INTERRUPTED`/`FAILED`, `train_history`, `eval_history`, `epoch_history`, `artifacts`). |
| `ExecutionContext`, `ExecutionState` | Runtime context passed to Methods and Hooks (`backward`, `step`, `zero_grad`, `log_metric`, `request_stop`). |
| `ExecutionBackend`, `NativeBackend`, `ExecutionEngine` | Pluggable execution backend protocol and default zero-dependency engine. |
| `BaseHook` | Optional convenience base class for lifecycle plugins. |

### Instrumentation
| Symbol | Description |
|---|---|
| `Probe`, `probe(name, value)` | Zero-overhead pass-through tensor/value probes using `contextvars`. |
| `InstrumentationManager`, `InstrumentationHook` | Manages probe subscriptions, `watch_module`, `watch_modules`, sampling, and automatic cleanup. |
| `InstrumentationEvent`, `InstrumentationRecord`, `WatchHandle` | Event payloads, recorded measurements, and removable handles. |
| `tensor_stats`, `check_nan_inf` | Built-in summary statistics and NaN/Inf diagnostics. |

### Registry & Plugins
| Symbol | Description |
|---|---|
| `ComponentRegistry`, `RegistrationEntry`, `default_registry` | Open-category registry (`@register`, `register_factory`, `get`, `create`, `unregister`). |
| `PluginManager`, `PluginManifest`, `PluginProtocol`, `DiscoveredPlugin`, `LoadedPlugin`, `default_plugin_manager` | Safe plugin discovery and explicit `load_plugin`, `load_file`, `load_path`, `load_directory`, `load_module`, `load_entry_points`, `unload_plugin`, `unload_all`. |

### Portable `.rfx` Packaging
| Symbol | Description |
|---|---|
| `export_experiment(experiment, destination, *, profile="standard", ...)` | Exports atomic `.rfx` ZIP archive (`minimal`, `standard`, `portable`, `full`). |
| `load_experiment(path, *, verify_checksums=True, load_states=True, allow_pickle=False)` | Safely loads `.rfx` archive into `ImportedExperiment` without executing plugins or installing packages. |
| `ArchiveManifest`, `ExportProfile`, `ImportedExperiment`, `RFX_EXTENSION` | Packaging types and constants. |

### Exceptions
`ResearchFrameworkError`, `ComponentIntegrationError`, `InvalidSpecificationError`, `ValidatorContractError`, `ValidationFailedError`, `PluginLoadError`, `ArchiveError`.

---

## 2. PyTorch Integration (`reframx.integrations.pytorch`)

- **Adapters:** `TorchModelAdapter`, `TorchDatasetAdapter`, `TorchLossAdapter`
- **Strategy:** `TorchStrategy` (`device`, `precision`, `grad_accumulation_steps`, `max_grad_norm`, `auto_move_batch`, `autocast_context`)
- **Built-in Objectives:** `CrossEntropyObjective`, `BCEWithLogitsObjective`, `MSEObjective`
- **Built-in Methods:** `TorchSupervisedMethod`, `TorchClassificationMethod`, `TorchRegressionMethod`
- **Built-in Evaluators:** `TorchAccuracyEvaluator`, `TorchRegressionEvaluator`
- **Utilities:** `is_torch_available`, `require_torch`, `resolve_device`, `move_to_device`

---

## 3. MMEngine Integration (`reframx.integrations.mmengine`)

- `MMEngineBackend` — `ExecutionBackend` powered by `mmengine.runner.Runner` (`work_dir`, `launcher`, `by_epoch`, `optim_wrapper`, `param_scheduler`, `default_hooks`, `custom_hooks`, `visualizer`, `resume`, `load_from`).
- `MMEngineStrategy` — bridges `ctx.backward()` / `ctx.step()` to `OptimWrapper` / `OptimWrapperDict`.
- `is_mmengine_available`, `require_mmengine`.
