"""Non-invasive wrapper used by the framework to manage component objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reframx.core.components.metadata import ComponentMetadata
from reframx.core.components.specification import ComponentSpec
from reframx.core.exceptions import InvalidSpecificationError


@dataclass(slots = True)
class ComponentHandle:
    """Framework-side reference to a user-owned component instance.

    The original object is kept intact. The handle stores integration metadata
    alongside it instead of injecting framework attributes into user objects.
    """

    instance: Any
    role: str
    spec: ComponentSpec
    origin: str = "user"

    @classmethod
    def wrap(
        cls,
        instance: Any,
        *,
        role: str | None = None,
        spec: ComponentSpec | None = None,
        origin: str = "user",
    ) -> "ComponentHandle":
        """Wrap an object, using `get_spec()` when available.

        Plain Python objects are accepted. If they do not implement `get_spec`,
        the framework creates a minimal specification. Callers can also supply
        an explicit specification without modifying the original object.
        """
        if instance is None:
            raise ValueError("Cannot wrap None as a component.")

        resolved_role = role or getattr(instance, "component_type", None) or "component"

        if spec is None:
            get_spec = getattr(instance, "get_spec", None)
            if callable(get_spec):
                spec = get_spec()
                if not isinstance(spec, ComponentSpec):
                    raise InvalidSpecificationError(
                        f"{type(instance).__name__}.get_spec() must return ComponentSpec; "
                        f"got {type(spec).__name__}."
                    )
            else:
                cls_obj = type(instance)
                metadata = ComponentMetadata(
                    name=getattr(instance, "name", cls_obj.__name__),
                    component_type=resolved_role,
                    description=(cls_obj.__doc__.strip() if cls_obj.__doc__ else None),
                    source=getattr(cls_obj, "__module__", None),
                )
                spec = ComponentSpec(metadata=metadata)

        if not isinstance(spec,ComponentSpec):
            raise InvalidSpecificationError(
                f"spec must be ComponentSpec; got {type(spec).__name__}."
            )
        # If the caller did not choose a role, prefer the role declared by the spec.
        if role is None:
            resolved_role = spec.metadata.component_type

        return cls(instance=instance, role=resolved_role, spec=spec, origin=origin)
    @property
    def metadata(self) -> ComponentMetadata:
        """Convenient access to the component metadata."""
        return self.spec.metadata
    
    @property
    def component_type(self) -> str:
        """Return the declared type from the specification."""
        return self.spec.metadata.component_type

    def get_custom_validator(self):
        """Return the optional user-defined validator, if one is present."""
        validator = getattr(self.instance, "validate_component", None)
        return validator if callable(validator) else None
