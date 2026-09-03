# -*- coding: utf-8 -*-
"""The resume and header guards in `common/append.py`.

Every one of these corresponds to a failure that cost measurements: shifted
columns when a script gained a parameter, a resume key missing a knob so a
re-run silently did nothing, and two processes writing one output.
"""
import sys

import pandas as pd
import pytest

from common import append


@pytest.fixture()
def out(tmp_path, monkeypatch):
    """An output name that resolves inside tmp_path instead of results/."""
    from common import paths
    monkeypatch.setattr(paths, "result", lambda name: tmp_path / name)
    monkeypatch.setattr(append, "_path", lambda name: tmp_path / str(name))
    return "t.csv"


def test_write_then_done_round_trips(out):
    append.write(out, dict(model="GBM", seed=1, v=0.5), ["model", "seed", "v"])
    assert append.done(out, ["model", "seed"]) == {("GBM", 1.0)}


def test_resume_key_matches_across_int_and_float(out):
    append.write(out, dict(model="GBM", seed=1, v=0.5), ["model", "seed", "v"])
    done = append.done(out, ["model", "seed"])
    assert append.key(dict(model="GBM", seed=1.0), ["model", "seed"]) in done


def test_a_new_column_widens_the_file_instead_of_shifting_it(out):
    append.write(out, dict(a=1, b=2), ["a", "b"])
    append.write(out, dict(a=3, b=4, c=5), ["a", "b", "c"])
    d = pd.read_csv(append._path(out))
    assert list(d.columns) == ["a", "b", "c"]
    assert pd.isna(d.c.iloc[0]) and d.c.iloc[1] == 5


def test_a_dropped_column_is_refused(out):
    append.write(out, dict(a=1, b=2), ["a", "b"])
    with pytest.raises(append.Shifted):
        append.write(out, dict(a=3), ["a"])


def test_done_refuses_a_key_column_the_file_does_not_have(out):
    """The failure mode where a re-run with a new value finds 'already done'."""
    append.write(out, dict(model="GBM", seed=1), ["model", "seed"])
    with pytest.raises(append.Shifted):
        append.done(out, ["model", "seed", "n_est"])


def test_a_second_writer_waits_rather_than_interleaving(out, monkeypatch):
    monkeypatch.setattr(append, "WAIT_S", 0)
    p = append._path(out)
    lock = append._Lock(p)
    lock.__enter__()
    try:
        with pytest.raises(TimeoutError):
            append.write(out, dict(a=1), ["a"])
    finally:
        lock.__exit__()


def test_provenance_is_recorded_for_the_output(out, tmp_path):
    append._provenance_seen = set()
    from common import provenance
    provenance._recorded.clear()
    append.write(out, dict(a=1), ["a"])
    d = pd.read_csv(tmp_path / "provenance.csv")
    assert d.output.iloc[0] == "t.csv"
    assert set(["git_sha", "v_numpy", "python", "venv"]) <= set(d.columns)


def test_a_new_result_lands_in_the_subfolder_of_the_script_writing_it(monkeypatch):
    """results/ mirrors experiments/, so a first write must not land in its root."""
    from common import paths
    monkeypatch.setattr(
        sys, "argv", [str(paths.ROOT / "experiments" / "h1_shape_vs_scale" / "x.py")])
    p = paths.result("a_name_that_does_not_exist_yet.csv")
    assert p.parent == paths.ROOT / "results" / "h1_shape_vs_scale"
