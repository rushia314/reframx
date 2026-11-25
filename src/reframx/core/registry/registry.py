"""Extensible component registry supporting built-in and plugin components."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RegistrationEntry:
    """Metadata record for one registered component factory or class."""

    category: str
    name: str
    factory: Any
    version: str | None = None
    description: str | None = None
    source: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.category or not self.category.strip():
            raise ValueError("RegistrationEntry.category must be non-empty.")
        if not self.name or not self.name.strip():
            raise ValueError("RegistrationEntry.name must be non-empty.")
        object.__setattr__(self, "category", self.category.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "metadata", dict(self.metadata))


class ComponentRegistry:
    """Stores component classes/factories grouped by open string category.

    Categories are open strings (e.g. ``"task"``, ``"method"``, ``"model"``,
    ``"dataset"``, ``"objective"``, ``"evaluator"``, ``"strategy"``, ``"hook"``,
    ``"backend"``) so plugins can introduce new categories without modifying core.
    """

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, RegistrationEntry]] = {}

    def register_factory(
        self,
        category: str,
        name: str,
        factory: Any,
        *,
        version: str | None = None,
        description: str | None = None,
        source: str | None = None,
        tags: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> RegistrationEntry:
        """Register a component class, callable factory, or object."""
        cat = category.strip()
        key = name.strip()
        if not cat:
            raise ValueError("category must be a non-empty string.")
        if not key:
            raise ValueError("name must be a non-empty string.")

        bucket = self._entries.setdefault(cat, {})
        if key in bucket and not overwrite:
            raise ValueError(
                f"Component '{key}' is already registered in category '{cat}'. "
                "Pass overwrite=True to replace it."
            )

        resolved_source = source or getattr(factory, "__module__", None)
        resolved_desc = description
        if resolved_desc is None and getattr(factory, "__doc__", None):
            resolved_desc = factory.__doc__.strip().splitlines()[0]

        entry = RegistrationEntry(
            category=cat,
            name=key,
            factory=factory,
            version=version,
            description=resolved_desc,
            source=resolved_source,
            tags=tuple(tags),
            metadata=metadata or {},
        )
        bucket[key] = entry
        return entry

    def register(
        self,
        category: str,
        name: str | None = None,
        *,
        version: str | None = None,
        description: str | None = None,
        source: str | None = None,
        tags: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Callable[[T], T]:
        """Decorator for registering a component class or factory function."""

        def _decorator(obj: T) -> T:
            resolved_name = name or getattr(obj, "__name__", None)
            if not resolved_name:
                raise ValueError(
                    "Cannot infer component name; pass `name=...` explicitly."
                )
            self.register_factory(
                category,
                resolved_name,
                obj,
                version=version,
                description=description,
                source=source,
                tags=tags,
                metadata=metadata,
                overwrite=overwrite,
            )
            return obj

        return _decorator

    def unregister(self, category: str, name: str) -> None:
        cat = category.strip()
        key = name.strip()
        bucket = self._entries.get(cat)
        if bucket is None or key not in bucket:
            raise KeyError(f"Component '{key}' is not registered in category '{cat}'.")
        del bucket[key]
        if not bucket:
            del self._entries[cat]

    def has(self, category: str, name: str) -> bool:
        return name.strip() in self._entries.get(category.strip(), {})

    def get_entry(self, category: str, name: str) -> RegistrationEntry:
        cat = category.strip()
        key = name.strip()
        bucket = self._entries.get(cat)
        if bucket is None or key not in bucket:
            available = sorted(bucket.keys()) if bucket else []
            raise KeyError(
                f"Component '{key}' not found in category '{cat}'. "
                f"Available: {available}"
            )
        return bucket[key]

    def get(self, category: str, name: str) -> Any:
        """Return the registered class/factory for ``(category, name)``."""
        return self.get_entry(category, name).factory

    def create(self, category: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered component by calling its factory."""
        factory = self.get(category, name)
        if not callable(factory):
            if args or kwargs:
                raise TypeError(
                    f"Registered object '{name}' in '{category}' is not callable."
                )
            return factory
        return factory(*args, **kwargs)

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries.keys()))

    def list(self, category: str | None = None) -> tuple[RegistrationEntry, ...]:
        """List all registered entries, optionally filtered by ``category``."""
        if category is not None:
            bucket = self._entries.get(category.strip(), {})
            return tuple(bucket[k] for k in sorted(bucket.keys()))
        entries: list[RegistrationEntry] = []
        for cat in sorted(self._entries.keys()):
            bucket = self._entries[cat]
            entries.extend(bucket[k] for k in sorted(bucket.keys()))
        return tuple(entries)

    def clear(self) -> None:
        self._entries.clear()


default_registry = ComponentRegistry()
