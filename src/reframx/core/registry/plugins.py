"""Plugin manifest, discovery, and explicit loading for external extensions."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reframx.core.exceptions import PluginLoadError
from reframx.core.registry.registry import ComponentRegistry, default_registry

DEFAULT_ENTRY_POINT_GROUP = "reframx.plugins"


@dataclass(frozen=True, slots=True)
class PluginManifest:
    """Descriptive identity for a Research Framework plugin."""

    name: str
    version: str | None = None
    api_version: str | None = None
    description: str | None = None
    author: str | None = None
    categories: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("PluginManifest.name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "categories", tuple(self.categories))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class PluginProtocol(Protocol):
    """Protocol for object/class plugins exposing ``register(registry)``."""

    def register(self, registry: ComponentRegistry) -> None:
        """Register plugin components into ``registry``."""
        ...


@dataclass(frozen=True, slots=True)
class DiscoveredPlugin:
    """An entry-point plugin discovered in the environment (not yet executed)."""

    name: str
    value: str
    group: str
    dist_name: str | None = None
    dist_version: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    """Record of a plugin that has been loaded into a ``PluginManager``."""

    manifest: PluginManifest
    source: str
    plugin: Any
    registered_entries: tuple[tuple[str, str], ...] = ()
    loaded_modules: tuple[str, ...] = ()


class _ScopedRegistry:
    """Proxy tracking every ``(category, name)`` registered by one plugin."""

    def __init__(self, target: ComponentRegistry) -> None:
        self._target = target
        self.registered: list[tuple[str, str]] = []

    def register_factory(
        self,
        category: str,
        name: str,
        factory: Any,
        **kwargs: Any,
    ) -> Any:
        entry = self._target.register_factory(category, name, factory, **kwargs)
        key = (entry.category, entry.name)
        if key not in self.registered:
            self.registered.append(key)
        return entry

    def register(
        self,
        category: str,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        def _decorator(obj: Any) -> Any:
            resolved_name = name or getattr(obj, "__name__", None)
            if not resolved_name:
                raise ValueError(
                    "Cannot infer component name; pass `name=...` explicitly."
                )
            self.register_factory(category, resolved_name, obj, **kwargs)
            return obj

        return _decorator

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target, attr)


def _resolve_manifest(plugin: Any, *, fallback_name: str) -> PluginManifest:
    get_manifest = getattr(plugin, "get_manifest", None)
    if callable(get_manifest):
        manifest = get_manifest()
        if not isinstance(manifest, PluginManifest):
            raise PluginLoadError(
                f"Plugin '{fallback_name}' get_manifest() must return PluginManifest; "
                f"got {type(manifest).__name__}."
            )
        return manifest

    attr_manifest = getattr(plugin, "PLUGIN_MANIFEST", None) or getattr(
        plugin, "manifest", None
    )
    if isinstance(attr_manifest, PluginManifest):
        return attr_manifest

    inferred_name = (
        getattr(plugin, "__name__", None)
        or type(plugin).__name__
        or fallback_name
    )
    doc = getattr(plugin, "__doc__", None)
    description = doc.strip().splitlines()[0] if isinstance(doc, str) and doc.strip() else None
    return PluginManifest(
        name=str(inferred_name),
        version=getattr(plugin, "__version__", None),
        description=description,
    )


def _invoke_register(plugin: Any, registry: Any, *, label: str) -> Any:
    """Invoke plugin registration and return the concrete plugin instance/object."""
    register_fn = getattr(plugin, "register", None)
    if callable(register_fn) and not isinstance(plugin, type):
        register_fn(registry)
        return plugin

    if isinstance(plugin, type):
        try:
            instance = plugin()
        except TypeError as exc:
            raise PluginLoadError(
                f"Plugin class '{label}' must be instantiable with no arguments "
                "or define a callable register(registry)."
            ) from exc
        inst_register = getattr(instance, "register", None)
        if callable(inst_register):
            inst_register(registry)
            return instance

    if callable(plugin):
        plugin(registry)
        return plugin

    raise PluginLoadError(
        f"Plugin '{label}' must be a callable `register(registry)` or an "
        "object/module defining `register(registry)`."
    )


class PluginManager:
    """Manages plugin discovery and explicit loading into a ``ComponentRegistry``.

    Discovery via ``discover_entry_points()`` inspects package metadata only
    and never executes untrusted plugin code unless ``load=True`` or
    ``load_entry_points()`` is explicitly called.
    """

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self.registry: ComponentRegistry = (
            registry if registry is not None else ComponentRegistry()
        )
        self._loaded: dict[str, LoadedPlugin] = {}

    @property
    def loaded_plugins(self) -> dict[str, LoadedPlugin]:
        return dict(self._loaded)

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def load_plugin(
        self,
        plugin: Any,
        *,
        name: str | None = None,
        source: str = "object",
        reload: bool = False,
        loaded_modules: tuple[str, ...] = (),
    ) -> LoadedPlugin:
        """Load an in-memory plugin object, class, module, or callable."""
        fallback = name or str(getattr(plugin, "__name__", None) or type(plugin).__name__)
        manifest = _resolve_manifest(plugin, fallback_name=fallback)
        if name is not None and manifest.name != name:
            manifest = PluginManifest(
                name=name,
                version=manifest.version,
                api_version=manifest.api_version,
                description=manifest.description,
                author=manifest.author,
                categories=manifest.categories,
                tags=manifest.tags,
                metadata=manifest.metadata,
            )

        if manifest.name in self._loaded:
            if not reload:
                return self._loaded[manifest.name]
            self.unload_plugin(manifest.name)

        scoped = _ScopedRegistry(self.registry)
        try:
            active_plugin = _invoke_register(plugin, scoped, label=manifest.name)
        except PluginLoadError:
            # Roll back any partially registered components.
            for cat, comp_name in reversed(scoped.registered):
                if self.registry.has(cat, comp_name):
                    self.registry.unregister(cat, comp_name)
            raise
        except Exception as exc:
            for cat, comp_name in reversed(scoped.registered):
                if self.registry.has(cat, comp_name):
                    self.registry.unregister(cat, comp_name)
            raise PluginLoadError(
                f"Failed to register plugin '{manifest.name}' from {source}: {exc}"
            ) from exc

        record = LoadedPlugin(
            manifest=manifest,
            source=source,
            plugin=active_plugin,
            registered_entries=tuple(scoped.registered),
            loaded_modules=loaded_modules,
        )
        self._loaded[manifest.name] = record
        return record

    def unload_plugin(self, name: str, *, missing_ok: bool = False) -> bool:
        """Unload a loaded plugin, calling optional ``unregister(registry)``
        and removing all components it registered.
        """
        record = self._loaded.get(name)
        if record is None:
            if missing_ok:
                return False
            raise KeyError(f"Plugin '{name}' is not currently loaded.")

        unregister_fn = getattr(record.plugin, "unregister", None)
        if callable(unregister_fn):
            unregister_fn(self.registry)

        for cat, comp_name in reversed(record.registered_entries):
            if self.registry.has(cat, comp_name):
                self.registry.unregister(cat, comp_name)

        for mod_prefix in record.loaded_modules:
            for mod_name in list(sys.modules.keys()):
                if mod_name == mod_prefix or mod_name.startswith(f"{mod_prefix}."):
                    del sys.modules[mod_name]

        del self._loaded[name]
        return True

    def unload_all(self) -> tuple[str, ...]:
        """Unload all currently loaded plugins in reverse load order."""
        names = tuple(reversed(list(self._loaded.keys())))
        for name in names:
            self.unload_plugin(name)
        return names

    def load_module(
        self,
        module_path: str,
        *,
        reload: bool = False,
    ) -> LoadedPlugin:
        """Import a Python module by dotted path and register it as a plugin."""
        try:
            module = importlib.import_module(module_path)
            if reload:
                module = importlib.reload(module)
        except Exception as exc:
            raise PluginLoadError(
                f"Could not import plugin module '{module_path}': {exc}"
            ) from exc
        return self.load_plugin(module, source=f"module:{module_path}", reload=reload)

    def load_file(
        self,
        file_path: str | Path,
        *,
        module_name: str | None = None,
        reload: bool = False,
    ) -> LoadedPlugin:
        """Load a standalone ``.py`` plugin file from disk."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise PluginLoadError(f"Plugin file does not exist: {path}")

        mod_name = module_name or f"rf_plugin_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Could not create module spec for plugin file: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(mod_name, None)
            raise PluginLoadError(
                f"Error executing plugin file '{path}': {exc}"
            ) from exc

        return self.load_plugin(
            module,
            source=f"file:{path}",
            reload=reload,
            loaded_modules=(mod_name,),
        )

    def load_path(
        self,
        path: str | Path,
        *,
        module_name: str | None = None,
        reload: bool = False,
    ) -> LoadedPlugin:
        """Load a plugin from a local ``.py`` file or a multi-file package directory.

        A package directory must contain ``__init__.py`` and may freely use
        relative imports (e.g. ``from .models import MyModel``) without being
        installed into ``sys.path``.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise PluginLoadError(f"Plugin path does not exist: {resolved}")

        if resolved.is_file():
            return self.load_file(resolved, module_name=module_name, reload=reload)

        if resolved.is_dir():
            init_file = resolved / "__init__.py"
            if not init_file.is_file():
                raise PluginLoadError(
                    f"Plugin package directory '{resolved}' must contain '__init__.py'. "
                    "To load multiple plugins inside a folder, use load_directory()."
                )

            pkg_name = module_name or f"rf_plugin_pkg_{resolved.name}"
            if reload:
                for mod_key in list(sys.modules.keys()):
                    if mod_key == pkg_name or mod_key.startswith(f"{pkg_name}."):
                        del sys.modules[mod_key]

            spec = importlib.util.spec_from_file_location(
                pkg_name,
                init_file,
                submodule_search_locations=[str(resolved)],
            )
            if spec is None or spec.loader is None:
                raise PluginLoadError(
                    f"Could not create package spec for plugin directory: {resolved}"
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[pkg_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                for mod_key in list(sys.modules.keys()):
                    if mod_key == pkg_name or mod_key.startswith(f"{pkg_name}."):
                        del sys.modules[mod_key]
                raise PluginLoadError(
                    f"Error executing plugin package '{resolved}': {exc}"
                ) from exc

            return self.load_plugin(
                module,
                source=f"path:{resolved}",
                reload=reload,
                loaded_modules=(pkg_name,),
            )

        raise PluginLoadError(f"Unsupported plugin path: {resolved}")

    def load_directory(
        self,
        directory: str | Path,
        *,
        reload: bool = False,
    ) -> tuple[LoadedPlugin, ...]:
        """Load all ``.py`` plugin files and package directories inside ``directory``."""
        root = Path(directory).resolve()
        if not root.is_dir():
            raise PluginLoadError(f"Plugin directory does not exist: {root}")

        if (root / "__init__.py").is_file():
            return (self.load_path(root, reload=reload),)

        loaded: list[LoadedPlugin] = []
        for child in sorted(root.iterdir(), key=lambda p: p.name):
            if child.name.startswith("_") or child.name.startswith("."):
                continue
            if child.is_file() and child.suffix == ".py":
                loaded.append(self.load_file(child, reload=reload))
            elif child.is_dir() and (child / "__init__.py").is_file():
                loaded.append(self.load_path(child, reload=reload))
        return tuple(loaded)

    def discover_entry_points(
        self,
        *,
        group: str = DEFAULT_ENTRY_POINT_GROUP,
    ) -> tuple[DiscoveredPlugin, ...]:
        """Inspect installed entry points without executing plugin code."""
        eps = importlib.metadata.entry_points(group=group)
        discovered: list[DiscoveredPlugin] = []
        for ep in eps:
            dist = getattr(ep, "dist", None)
            discovered.append(
                DiscoveredPlugin(
                    name=ep.name,
                    value=ep.value,
                    group=group,
                    dist_name=getattr(dist, "name", None) if dist is not None else None,
                    dist_version=(
                        getattr(dist, "version", None) if dist is not None else None
                    ),
                )
            )
        return tuple(discovered)

    def load_entry_points(
        self,
        *,
        group: str = DEFAULT_ENTRY_POINT_GROUP,
        allowlist: set[str] | None = None,
        reload: bool = False,
    ) -> tuple[LoadedPlugin, ...]:
        """Explicitly load discovered entry-point plugins (optionally filtered)."""
        eps = importlib.metadata.entry_points(group=group)
        loaded: list[LoadedPlugin] = []
        for ep in eps:
            if allowlist is not None and ep.name not in allowlist:
                continue
            try:
                plugin_obj = ep.load()
            except Exception as exc:
                raise PluginLoadError(
                    f"Failed to load entry-point plugin '{ep.name}' ({ep.value}): {exc}"
                ) from exc
            loaded.append(
                self.load_plugin(
                    plugin_obj,
                    name=ep.name,
                    source=f"entry_point:{group}:{ep.name}",
                    reload=reload,
                )
            )
        return tuple(loaded)


default_plugin_manager = PluginManager(registry=default_registry)
