# -*- coding: utf-8 -*-
"""Every experiment script must import cleanly.

The scripts are not a package and are only ever run one at a time, so a name
left behind by an edit -- a constant moved to `common/`, an import dropped --
stays invisible until the run that needed it is started, which for the model
runs means hours later. Importing the module executes everything above
`main()`, which is where those names live.

Modules that need a model package installed are skipped rather than failed:
the three virtualenvs are mutually incompatible by construction, so no single
interpreter can import all of them.
"""
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "experiments").rglob("*.py"))
MODEL_PACKAGES = {"tabpfn", "tabicl", "tabdpt", "tabfm", "exaone", "torch",
                  "xgboost", "catboost", "exaone_tabular"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_imports(path):
    spec = importlib.util.spec_from_file_location(f"exp_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # A script is run as `python experiments/<h>/<name>.py`, which puts its own
    # directory first on the path. Two scripts import a sibling module that way.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in MODEL_PACKAGES:
            pytest.skip(f"needs {e.name}, which lives in another virtualenv")
        raise
    finally:
        sys.path.remove(str(path.parent))
