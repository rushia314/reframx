import pytest

from reframx import ComponentMetadata


def test_metadata_normalizes_name_type_and_tags():
    metadata = ComponentMetadata(
        name="  MyModel  ",
        component_type=" model ",
        tags=["nlp", "research"],
    )

    assert metadata.name == "MyModel"
    assert metadata.component_type == "model"
    assert metadata.tags == ("nlp", "research")


def test_metadata_rejects_empty_name():
    with pytest.raises(ValueError):
        ComponentMetadata(name="  ")
