import ast
from pathlib import Path

import nbformat


def test_notebooks_are_valid_python_and_have_no_saved_outputs():
    for path in Path("notebooks").glob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.outputs == []
                assert cell.execution_count is None
