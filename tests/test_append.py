# -*- coding: utf-8 -*-
"""The resume and header guards of `common/append.py`, each after a failure that cost
measurements: shifted columns, a key missing a knob, two writers on one file."""
import sys

import pandas as pd
import pytest

from common import append, files


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


def test_done_refuses_a_key_column_that_is_empty_in_every_row(out):
    """The worse half of the same failure: the column exists, so the check above
    passes, but nothing writes it. NaN does not equal itself, so every lookup
    misses and a re-run duplicates the file instead of resuming. This cost the
    wide leverage sweep a silent double-measurement."""
    append.write(out, dict(model="GBM", seed=1, n_est=None), ["model", "seed", "n_est"])
    with pytest.raises(append.Shifted):
        append.done(out, ["model", "seed", "n_est"])


def test_a_second_writer_waits_rather_than_interleaving(out, monkeypatch):
    monkeypatch.setattr(files, "WAIT_S", 0)
    with files.FileLock(append._path(out)):
        with pytest.raises(TimeoutError):
            append.write(out, dict(a=1), ["a"])


def test_an_empty_key_cell_matches_on_resume(out):
    """None is written as an empty cell and read back as NaN, which equals nothing.
    Without normalising both to None, the cell is measured again on every run."""
    cols = ["model", "seed", "dose"]
    append.write(out, dict(model="GBM", seed=1, dose=None), cols)
    append.write(out, dict(model="GBM", seed=2, dose=3), cols)
    assert append.key(dict(model="GBM", seed=1, dose=None), cols) in append.done(out, cols)


def test_widening_copies_values_as_text(out):
    """A pandas round trip turned an all-digit git sha into an int and dropped its zero."""
    append.write(out, dict(sha="0123456", v="1.10"), ["sha", "v"])
    append.write(out, dict(sha="0000001", v="2", c=1), ["sha", "v", "c"])
    assert "0123456,1.10," in append._path(out).read_text(encoding="utf-8")


def test_provenance_is_recorded_for_the_output(out, tmp_path):
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
