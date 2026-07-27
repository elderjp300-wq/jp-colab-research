import io
import json

from colab_cli import ipynb_lite
from colab_cli.converter import convert_history_to_ipynb


def test_notebook_nodes_support_nested_attribute_access():
    nb = ipynb_lite.v4.new_notebook()
    nb.metadata.kernelspec = {"name": "python3"}
    cell = ipynb_lite.v4.new_code_cell("print('ok')")
    nb.cells.append(cell)

    assert nb.metadata.kernelspec.name == "python3"
    assert cell.id

    stream = io.StringIO()
    ipynb_lite.write(nb, stream)
    raw = json.loads(stream.getvalue())
    assert raw["metadata"]["kernelspec"]["name"] == "python3"
    assert raw["cells"][0]["id"] == cell.id


def test_history_export_creates_valid_v4_shape():
    nb = convert_history_to_ipynb(
        [
            {
                "event_type": "execution",
                "timestamp": "2026-01-02T03:04:05",
                "code": "print('hello')",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": "hello\n",
                    }
                ],
            }
        ],
        "demo",
    )

    assert nb.nbformat == 4
    assert nb.nbformat_minor == 5
    assert nb.metadata.kernelspec.name == "python3"
    assert all(cell.id for cell in nb.cells)
    assert nb.cells[-1].outputs[0].text == "hello\n"


def test_read_write_preserves_unknown_fields():
    source = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "abc",
                "metadata": {"custom": {"enabled": True}},
                "outputs": [],
                "source": "1 + 1",
                "vendor_extension": {"x": 1},
            }
        ],
        "metadata": {"colab": {"provenance": []}},
        "nbformat": 4,
        "nbformat_minor": 5,
        "top_level_extension": "kept",
    }
    nb = ipynb_lite.read(io.StringIO(json.dumps(source)), as_version=4)
    output = io.StringIO()
    ipynb_lite.write(nb, output)
    assert json.loads(output.getvalue()) == source
