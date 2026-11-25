import pytest

from reframx import (
    Capability,
    ComponentHandle,
    ComponentMetadata,
    ComponentSpec,
)
from reframx.core.exceptions import InvalidSpecificationError


class PlainPythonObject:
    """A plain object with no framework-specific methods."""


def test_wrap_plain_python_object_without_mutating_it():
    obj = PlainPythonObject()
    handle = ComponentHandle.wrap(obj, role="model")

    assert handle.instance is obj
    assert handle.role == "model"
    assert handle.component_type == "model"
    assert not hasattr(obj, "component_handle")
    assert handle.metadata.name == "PlainPythonObject"


def test_wrap_uses_component_get_spec():
    expected = ComponentSpec(
        metadata=ComponentMetadata(name="M", component_type="model"),
        capabilities=(Capability("forward"),),
    )

    class Model:
        def get_spec(self):
            return expected

    handle = ComponentHandle.wrap(Model())
    assert handle.spec is expected
    assert handle.role == "model"


def test_wrap_rejects_invalid_get_spec_result():
    class BrokenComponent:
        def get_spec(self):
            return {"not": "a spec"}

    with pytest.raises(InvalidSpecificationError):
        ComponentHandle.wrap(BrokenComponent())


def test_explicit_spec_can_be_used_for_plain_object():
    obj = object()
    spec = ComponentSpec(
        metadata=ComponentMetadata(name="External", component_type="dataset")
    )
    handle = ComponentHandle.wrap(obj, spec=spec)
    assert handle.spec is spec
    assert handle.role == "dataset"
