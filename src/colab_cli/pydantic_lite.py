# pydantic_lite.py — thay thế pydantic bằng dataclasses + validate tay.
#
# pydantic (v2) delegates the actual parsing/validation work to
# pydantic-core, a Rust extension (pyo3/maturin) with no prebuilt wheel for
# Android/Termux arm64 -- same class of problem as rpds-py. colab-cli only
# uses a small slice of pydantic's features:
#   - BaseModel with typed fields, some required, some with defaults
#   - Field(..., alias="jsonName") to map camelCase JSON keys to snake_case
#     Python attributes
#   - TypeAdapter(SomeModel).validate_python(dict) to parse+validate a dict
#   - TypeAdapter(Union[A, B]).validate_python(dict) to try A, fall back to B
#   - .model_dump() / .model_dump_json() / .model_validate() (state.py)
#
# None of this needs a compiled extension -- it's straightforward attribute
# mapping and type coercion, which we do by hand below.

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields as dc_fields, MISSING
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, get_args, get_origin


class ValidationError(Exception):
    def __init__(self, message: str, model: str = ""):
        super().__init__(message)
        self.model = model


_UNSET = object()


def Field(default: Any = ..., *, alias: Optional[str] = None, **_kwargs):
    """Mimics pydantic.Field(...) closely enough for this codebase: only
    `default` (or `...` for required) and `alias` are actually used anywhere
    in colab-cli.
    """
    return _FieldInfo(default=default, alias=alias)


@dataclass
class _FieldInfo:
    default: Any
    alias: Optional[str] = None


class BaseModel:
    """Minimal stand-in for pydantic.BaseModel.

    Subclasses declare fields as plain class-level type annotations, with
    either a bare value default (`x: int = 3`) or `Field(..., alias=...)`
    for required-with-alias / defaulted-with-alias fields -- exactly the
    two patterns colab-cli's client.py and state.py use.
    """

    def __init__(self, **data: Any):
        annotations = self._collect_annotations()
        for py_name, py_type in annotations.items():
            field_info = self._field_info(py_name)
            alias = field_info.alias if field_info else None
            lookup_key = alias if alias and alias in data else py_name

            if lookup_key in data:
                raw_value = data[lookup_key]
            elif field_info is not None and field_info.default is not ...:
                raw_value = field_info.default
            elif hasattr(type(self), py_name) and not isinstance(
                getattr(type(self), py_name), _FieldInfo
            ):
                raw_value = getattr(type(self), py_name)
            else:
                raise ValidationError(
                    f"{type(self).__name__}: missing required field "
                    f"'{alias or py_name}'",
                    model=type(self).__name__,
                )

            setattr(self, py_name, _coerce(raw_value, py_type))

        # Allow extra kwargs silently (pydantic default is to forbid, but
        # colab-cli never relies on that behavior; being lenient here is
        # safer for forward-compat with API responses gaining new fields).

    @classmethod
    def _collect_annotations(cls) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            out.update(getattr(klass, "__annotations__", {}))
        return out

    @classmethod
    def _field_info(cls, py_name: str) -> Optional[_FieldInfo]:
        val = getattr(cls, py_name, None)
        return val if isinstance(val, _FieldInfo) else None

    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> "BaseModel":
        return cls(**data)

    def model_dump(self) -> Dict[str, Any]:
        out = {}
        for py_name in self._collect_annotations():
            val = getattr(self, py_name)
            out[py_name] = _dump_value(val)
        return out

    def model_dump_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.model_dump(), indent=indent, default=str)

    def __repr__(self):
        fields_str = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self._collect_annotations()
        )
        return f"{type(self).__name__}({fields_str})"


def _dump_value(val: Any) -> Any:
    if isinstance(val, BaseModel):
        return val.model_dump()
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, list):
        return [_dump_value(v) for v in val]
    if isinstance(val, tuple):
        return [_dump_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _dump_value(v) for k, v in val.items()}
    return val


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort conversion of a raw (JSON-decoded) value to target_type.
    Only handles the shapes colab-cli's models actually declare: str, int,
    bool, Enum subclasses, BaseModel subclasses, Optional[...], List[...],
    Tuple[...], and plain passthroughs.
    """
    if value is None:
        return None

    origin = get_origin(target_type)

    # Optional[X] / Union[X, None] / Union[A, B]
    if origin is Union:
        args = [a for a in get_args(target_type) if a is not type(None)]
        if len(args) == 1:
            return _coerce(value, args[0])
        # Multi-type union (e.g. Union[GetAssignmentResponse, Assignment]):
        # try each in order, first success wins. Mirrors TypeAdapter(Union)
        # behavior used in client.py's _get_assignment.
        last_err = None
        for candidate in args:
            try:
                return _coerce(value, candidate)
            except (ValidationError, TypeError, ValueError) as e:
                last_err = e
                continue
        raise ValidationError(
            f"Value did not match any type in {target_type}: {last_err}"
        )

    if origin in (list, List):
        (item_type,) = get_args(target_type) or (Any,)
        return [_coerce(v, item_type) for v in value]

    if origin in (tuple, Tuple):
        item_types = get_args(target_type)
        if not item_types:
            return tuple(value)
        return tuple(
            _coerce(v, t) for v, t in zip(value, item_types)
        )

    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        if isinstance(value, target_type):
            return value
        if isinstance(value, dict):
            return target_type(**value)
        raise ValidationError(f"Cannot coerce {value!r} into {target_type}")

    if isinstance(target_type, type) and issubclass(target_type, Enum):
        if isinstance(value, target_type):
            return value
        try:
            return target_type(value)
        except ValueError:
            raise ValidationError(f"{value!r} is not a valid {target_type.__name__}")

    if target_type is datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise ValidationError(f"Cannot coerce {value!r} into datetime")

    if target_type in (str, int, float, bool):
        if isinstance(value, target_type):
            return value
        try:
            return target_type(value)
        except (TypeError, ValueError):
            raise ValidationError(f"Cannot coerce {value!r} into {target_type}")

    # Any / unrecognized typing construct / plain object: passthrough
    return value


class TypeAdapter:
    """Mimics pydantic.TypeAdapter(SomeType).validate_python(data), covering
    the two call shapes client.py uses: a single BaseModel subclass, and
    Union[ModelA, ModelB].
    """

    def __init__(self, schema: Any):
        self.schema = schema

    def validate_python(self, data: Any) -> Any:
        if self.schema is BaseModel:
            # client.py's unassign() passes schema=BaseModel for a response
            # whose content it never reads -- just needs *something* parsed
            # without error. Return the raw dict/value as-is.
            return data
        return _coerce(data, self.schema)
