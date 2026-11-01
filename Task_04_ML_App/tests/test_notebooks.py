"""The training work must live in executed Jupyter notebooks (outputs saved, no errors)."""
import nbformat
import pytest

from src.data_utils import ROOT

NOTEBOOKS = ["01_data_audit_and_eda.ipynb", "02_preparation_and_features.ipynb",
             "03_modeling_tuning_and_saving.ipynb"]


@pytest.fixture(scope="module", params=NOTEBOOKS)
def notebook(request):
    path = ROOT / "notebooks" / request.param
    assert path.exists(), f"{request.param} is missing"
    return request.param, nbformat.read(str(path), as_version=4)


def test_notebook_has_code_and_explanations(notebook):
    _, nb = notebook
    kinds = [c.cell_type for c in nb.cells]
    assert kinds.count("code") >= 5
    assert kinds.count("markdown") >= 5          # explanations between the code cells


def test_notebook_was_executed_and_has_no_error_output(notebook):
    name, nb = notebook
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in code_cells), f"{name}: some cells were never executed"
    errors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{name} contains error output"
    assert any(c.get("outputs") for c in code_cells), f"{name} has no saved outputs"


def test_training_code_is_in_the_notebook_not_a_script():
    nb = nbformat.read(str(ROOT / "notebooks" / "03_modeling_tuning_and_saving.ipynb"), as_version=4)
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for needle in ["GridSearchCV", "RandomizedSearchCV", "cross_val_score", "joblib.dump", "permutation_importance"]:
        assert needle in source, f"{needle} not found in the training notebook"
    assert not (ROOT / "src" / "train.py").exists()
