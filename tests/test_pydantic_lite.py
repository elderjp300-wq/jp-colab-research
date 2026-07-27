from datetime import datetime
from typing import List, Optional, Tuple

import pytest

from colab_cli.pydantic_lite import BaseModel, Field, TypeAdapter, ValidationError


class Child(BaseModel):
    value: int


class Example(BaseModel):
    name: str
    enabled: bool = True
    values: List[int] = []
    child: Child = Field(..., alias="childValue")
    pair: Optional[Tuple[str, Optional[str]]] = None


def test_alias_nested_model_and_round_trip():
    model = Example(
        name="demo",
        enabled="false",
        values=["1", 2],
        childValue={"value": "3"},
        pair=["left", None],
    )

    assert model.enabled is False
    assert model.child.value == 3
    assert model.pair == ("left", None)
    assert model.model_dump() == {
        "name": "demo",
        "enabled": False,
        "values": [1, 2],
        "child": {"value": 3},
        "pair": ["left", None],
    }


def test_required_non_optional_field_rejects_none():
    with pytest.raises(ValidationError):
        Example(name=None, childValue={"value": 1})


def test_invalid_container_and_boolean_rejected():
    with pytest.raises(ValidationError):
        Example(name="demo", values="12", childValue={"value": 1})

    with pytest.raises(ValidationError):
        Example(name="demo", enabled="not-a-bool", childValue={"value": 1})


def test_mutable_defaults_are_not_shared():
    first = Example(name="first", childValue={"value": 1})
    second = Example(name="second", childValue={"value": 2})
    first.values.append(9)
    assert second.values == []


def test_model_copy_and_model_validate_instance():
    original = Example(name="before", childValue={"value": 1})
    copied = original.model_copy(update={"name": "after"}, deep=True)

    assert copied is not original
    assert copied.name == "after"
    assert copied.child is not original.child
    assert Example.model_validate(original) is original


def test_datetime_serialization():
    class WithDate(BaseModel):
        timestamp: datetime

    model = TypeAdapter(WithDate).validate_python(
        {"timestamp": "2026-01-02T03:04:05"}
    )
    assert model.model_dump_json() == '{"timestamp": "2026-01-02T03:04:05"}'
