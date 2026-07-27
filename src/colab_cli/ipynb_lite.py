"""Small, pure-Python subset of :mod:`nbformat` used by colab-cli.

Notebook files are JSON.  The full nbformat dependency is useful when schema
validation and old-version conversion are required, but colab-cli only reads
and writes nbformat v4 notebooks.  This module preserves unknown fields and
provides NotebookNode-compatible attribute access without jsonschema/rpds-py.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any, Dict


def _cell_id() -> str:
    # nbformat uses compact, unique IDs matching the v4.5 cell-id grammar.
    return uuid.uuid4().hex[:8]


def _convert(value: Any) -> Any:
    if isinstance(value, NotebookNode):
        return value
    if isinstance(value, dict):
        return NotebookNode(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if isinstance(value, tuple):
        return [_convert(item) for item in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


class NotebookNode(dict):
    """Dictionary with recursive attribute access like nbformat.NotebookNode."""

    def __init__(self, value: Dict[str, Any] | None = None, **kwargs: Any):
        super().__init__()
        source = dict(value or {})
        source.update(kwargs)
        for key, item in source.items():
            super().__setitem__(key, _convert(item))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any):
        self[name] = value

    def __setitem__(self, key: str, value: Any):
        super().__setitem__(key, _convert(value))

    def update(self, *args: Any, **kwargs: Any):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return _to_plain(self)


# Compatibility aliases used by annotations and callers.
Cell = NotebookNode
Notebook = NotebookNode


def read(fileobj_or_path, as_version: int = 4) -> NotebookNode:
    """Read an nbformat v4 notebook without schema validation or migration."""
    if as_version != 4:
        raise ValueError("ipynb_lite only supports as_version=4")

    if hasattr(fileobj_or_path, "read"):
        raw = json.load(fileobj_or_path)
    else:
        with open(fileobj_or_path, "r", encoding="utf-8") as file_obj:
            raw = json.load(file_obj)

    version = raw.get("nbformat", 4)
    if version != 4:
        raise ValueError(
            f"ipynb_lite only supports nbformat v4, got {version}. "
            "Install the real `nbformat` package if you need older notebooks."
        )
    return NotebookNode(raw)


def write(nb: NotebookNode | Dict[str, Any], fileobj_or_path):
    data = _to_plain(nb)
    if hasattr(fileobj_or_path, "write"):
        json.dump(data, fileobj_or_path, indent=1, ensure_ascii=False)
        fileobj_or_path.write("\n")
    else:
        with open(fileobj_or_path, "w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, indent=1, ensure_ascii=False)
            file_obj.write("\n")


def new_output(output_type: str, **kwargs: Any) -> NotebookNode:
    defaults: Dict[str, Any] = {"output_type": output_type}
    if output_type == "stream":
        defaults.update(name="stdout", text="")
    elif output_type == "display_data":
        defaults.update(data={}, metadata={})
    elif output_type == "execute_result":
        defaults.update(data={}, metadata={}, execution_count=None)
    elif output_type == "error":
        defaults.update(ename="", evalue="", traceback=[])
    defaults.update(kwargs)
    return NotebookNode(defaults)


def new_notebook(**kwargs: Any) -> NotebookNode:
    raw: Dict[str, Any] = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    raw.update(kwargs)
    return NotebookNode(raw)


def new_markdown_cell(source: str | list[str] = "", **kwargs: Any) -> NotebookNode:
    raw: Dict[str, Any] = {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": source,
    }
    raw.update(kwargs)
    return NotebookNode(raw)


def new_code_cell(
    source: str | list[str] = "", outputs=None, **kwargs: Any
) -> NotebookNode:
    raw: Dict[str, Any] = {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {},
        "outputs": [] if outputs is None else outputs,
        "source": source,
    }
    raw.update(kwargs)
    return NotebookNode(raw)


def ensure_cell_ids(nb: NotebookNode):
    for cell in nb.cells:
        if not getattr(cell, "id", None):
            cell.id = _cell_id()


class _V4Shim(SimpleNamespace):
    pass


v4 = _V4Shim(
    new_notebook=new_notebook,
    new_output=new_output,
    new_markdown_cell=new_markdown_cell,
    new_code_cell=new_code_cell,
)
