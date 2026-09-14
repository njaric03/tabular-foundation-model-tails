# -*- coding: utf-8 -*-
"""Every experiment script must import cleanly.

A name left behind by an edit stays invisible until the run that needs it starts,
hours later for the model runs. Importing a script executes everything above `main()`.
Scripts that need a model package are skipped: the virtualenvs are mutually
incompatible, so no interpreter can import all of them.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "experiments").rglob("*.py"))
MODEL_PACKAGES = {"tabpfn", "tabicl", "tabdpt", "tabfm", "exaonetabular", "torch",
                  "xgboost", "catboost"}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_imports(path):
    spec = importlib.util.spec_from_file_location(f"exp_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in MODEL_PACKAGES:
            pytest.skip(f"needs {e.name}, which lives in another virtualenv")
        raise


def test_every_knob_the_scripts_read_is_recorded_in_provenance():
    """A knob read through `common.env` but missing from ENV_KNOBS would change a
    result without a trace. Uses the knobs the imports above registered."""
    from common import env, provenance
    if not env.READ:
        pytest.skip("run together with test_script_imports")
    missing = sorted(env.READ - set(provenance.ENV_KNOBS))
    assert not missing, f"add to provenance.ENV_KNOBS: {missing}"
