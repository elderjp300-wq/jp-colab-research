# ipynb_lite.py — thay thế nbformat bằng json thuần.
#
# nbformat kéo jsonschema -> referencing -> rpds-py (Rust). Ta không cần
# validate schema .ipynb đầy đủ — chỉ cần đọc cells, gán id nếu thiếu, và
# ghi output ngược lại. .ipynb chỉ là JSON có cấu trúc cố định (nbformat 4),
# nên parse/serialize bằng `json` chuẩn là đủ.

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class Cell(SimpleNamespace):
    """Mimics the attribute-style access nbformat's NotebookNode gives
    (cell.source, cell.cell_type, cell.id, cell.outputs), backed by a plain
    dict so we can serialize it back out unchanged.
    """

    def __init__(self, raw: Dict[str, Any]):
        super().__init__(**raw)
        self._raw = raw

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "_raw" and hasattr(self, "_raw"):
            self._raw[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


class Notebook(SimpleNamespace):
    def __init__(self, raw: Dict[str, Any]):
        cells = [Cell(c) for c in raw.get("cells", [])]
        super().__init__(
            cells=cells,
            metadata=raw.get("metadata", {}),
            nbformat=raw.get("nbformat", 4),
            nbformat_minor=raw.get("nbformat_minor", 5),
        )
        self._raw = raw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "metadata": self.metadata,
            "nbformat": self.nbformat,
            "nbformat_minor": self.nbformat_minor,
        }


def read(fileobj_or_path, as_version: int = 4) -> Notebook:
    """Drop-in-ish replacement for nbformat.read(f, as_version=4).

    Only reads — no schema validation, no version migration (assumes the
    file is already nbformat v4, which is what Colab/Jupyter produce today).
    """
    if hasattr(fileobj_or_path, "read"):
        raw = json.load(fileobj_or_path)
    else:
        with open(fileobj_or_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    if raw.get("nbformat", 4) != 4:
        raise ValueError(
            f"ipynb_lite only supports nbformat v4, got {raw.get('nbformat')}. "
            "Install the real `nbformat` package if you need older notebooks."
        )
    return Notebook(raw)


def write(nb: Notebook, fileobj_or_path):
    data = nb.to_dict()
    if hasattr(fileobj_or_path, "write"):
        json.dump(data, fileobj_or_path, indent=1, ensure_ascii=False)
        fileobj_or_path.write("\n")
    else:
        with open(fileobj_or_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.write("\n")


def new_output(output_type: str, **kwargs) -> Dict[str, Any]:
    """Mimics nbformat.v4.new_output — just builds the plain dict shape."""
    out = {"output_type": output_type}
    out.update(kwargs)
    return out


def new_notebook(**kwargs) -> Notebook:
    raw = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    raw.update(kwargs)
    return Notebook(raw)


def ensure_cell_ids(nb: Notebook):
    """nbformat v4.5+ requires an 'id' on every cell; original code checked
    this per-cell inline. Kept as a helper for callers that want it upfront.
    """
    for cell in nb.cells:
        if not getattr(cell, "id", None):
            cell.id = str(uuid.uuid4())

def new_markdown_cell(source: str = "", **kwargs) -> Cell:
    raw = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }
    raw.update(kwargs)
    return Cell(raw)


def new_code_cell(
    source: str = "", outputs=None, **kwargs
) -> Cell:
    raw = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source,
    }
    raw.update(kwargs)
    return Cell(raw)


# Sub-namespace shim so callers can keep writing `nbformat.v4.new_notebook(...)`
# etc. after `from colab_cli import ipynb_lite as nbformat` -- matches the
# nbformat.v4 module layout closely enough for this codebase's usage.
class _V4Shim(SimpleNamespace):
    pass


v4 = _V4Shim(
    new_notebook=new_notebook,
    new_output=new_output,
    new_markdown_cell=new_markdown_cell,
    new_code_cell=new_code_cell,
)

# Alias so `nbformat.NotebookNode` type-hints (used only as a bare annotation,
# not for isinstance checks, in converter.py) resolve to something real.
NotebookNode = Cell
