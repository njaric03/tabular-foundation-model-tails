# -*- coding: utf-8 -*-
"""Loading the real tabular datasets, in one place.

Four variants of this loader existed across the scripts, differing in how they
handle a non-numeric target, categorical columns and non-positive y. The
canonical one is the ScoringBench survey loader, kept here.

freMTPL2sev is special: severity comes in one table and the covariates in
another, so the two have to be joined on the policy id and the claims summed
per policy. Three scripts carried a copy of that join.

    from common import datasets

    X, y = datasets.load("diamonds")
    names = datasets.scoringbench_names()
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from common import paths

# Names that live under a different name, or only under an id, on OpenML.
ALIASES = {"Ele2": 42362}

# What the last call to `load` actually loaded. Reset on every call.
#
# Two fields here are not decoration. `target_source` says whether the dataset's
# own target was used or the fallback below picked the last numeric column,
# which is a silent choice that feeds the survey of 99 datasets and therefore
# the base rate of leverage quoted in the findings. `n_categorical` says how
# many columns were ordinal-coded from categories; the foundation models are
# never told which those are, and freMTPL2sev, which carries the whole real-data
# leverage result, has several.
LAST: dict = {}

FINGERPRINTS = "dataset_fingerprints.json"

PMLB_URL = ("https://github.com/EpistasisLab/pmlb/raw/master/datasets/"
            "{name}/{name}.tsv.gz")


def scoringbench_names() -> list[str]:
    """The ScoringBench suite, about 102 dataset names."""
    return paths.load_json("sb_datasets.json")


def openml_ids() -> dict:
    """Name to OpenML id, for the datasets whose name does not resolve."""
    return paths.load_json("sb_openml_ids.json")


def selected_names() -> list[str]:
    """The hand-picked heavy-tailed datasets that pass the identification test."""
    return paths.lines("selected_datasets.txt")


def _load_pmlb(name):
    """PMLB datasets, named like 215_2dplanes, are not on OpenML under that name."""
    df = pd.read_csv(PMLB_URL.format(name=name), sep="\t", compression="gzip")
    y = pd.to_numeric(df.pop("target"), errors="coerce").to_numpy(dtype=float)
    return df, y


def _load_fremtpl2():
    """Claim severity joined to its covariates, summed per policy."""
    sev = fetch_openml("freMTPL2sev", as_frame=True, parser="auto").data
    freq = fetch_openml("freMTPL2freq", as_frame=True, parser="auto").data
    sev = sev.groupby("IDpol", as_index=False).ClaimAmount.sum()
    df = sev.merge(freq.drop(columns=["ClaimNb"]), on="IDpol", how="inner")
    y = df.pop("ClaimAmount").to_numpy(dtype=float)
    return df.drop(columns=["IDpol"]), y


def prepare(X, y):
    """Categories to codes, missing values filled, invalid targets dropped.

    Two properties of this step are limitations rather than choices, and both
    are recorded in `LAST` so they can be quoted instead of discovered:

    * categorical columns become integer codes and are then handed to every
      model as plain numbers. TabPFN and TabICL have their own categorical
      handling, and it is bypassed here. Changing that changes every real-data
      number, so it is recorded and left for a re-run, not switched silently.
    * the median that fills missing values is computed on the whole table,
      before any split. The leakage is small -- a median over thousands of rows
      barely moves when a few thousand are held out -- but it is leakage.
    """
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        raise RuntimeError("target is not numeric")
    X = X.copy()
    cat = list(X.select_dtypes(exclude="number").columns)
    for c in cat:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    ok = np.isfinite(y)
    LAST["n_categorical"] = len(cat)
    LAST["categorical"] = [X.columns.get_loc(c) for c in cat]
    return X.to_numpy(dtype=float)[ok], y[ok]


def fingerprint(X, y) -> dict:
    """Shape and a hash of the target, enough to notice that a source moved.

    `fetch_openml` is called by name, without a version, so the table it returns
    is whatever OpenML serves that day. Nothing in a result CSV would show that
    a dataset changed underneath a measurement. This is the cheapest guard: the
    first load records the fingerprint in `data/dataset_fingerprints.json`, and
    every later load compares against it.
    """
    y = np.asarray(y, dtype=float)
    return dict(n_rows=int(len(y)), n_cols=int(np.shape(X)[1]),
                y_sha1=hashlib.sha1(np.ascontiguousarray(y)).hexdigest()[:16],
                y_min=float(np.min(y)), y_max=float(np.max(y)))


def check_fingerprint(name: str, X, y) -> None:
    """Compare with the recorded fingerprint, or record it the first time.

    A mismatch prints loudly and, under TFM_STRICT=1, raises: a measurement
    started against a table that is not the one earlier rows were measured on is
    a run that should not quietly continue.
    """
    got = fingerprint(X, y)
    LAST["fingerprint"] = got
    try:
        path = paths.data(FINGERPRINTS)
    except Exception:
        path = paths.path(FINGERPRINTS, "data")
    known = {}
    if path.exists():
        known = json.loads(path.read_text(encoding="utf-8"))

    if name not in known:
        known[name] = got
        path.write_text(json.dumps(known, indent=1, sort_keys=True), encoding="utf-8")
        print(f"  [datasets] fingerprint recorded for {name}: "
              f"{got['n_rows']} x {got['n_cols']}, y {got['y_sha1']}", flush=True)
        return

    if known[name] != got:
        message = (f"{name}: the table is not the one that was measured before.\n"
                   f"  recorded: {known[name]}\n"
                   f"  now     : {got}\n"
                   f"Results measured before and after this change are not comparable.")
        if os.environ.get("TFM_STRICT", "") not in ("", "0", "false"):
            raise RuntimeError(message)
        print("  [datasets] WARNING " + message, flush=True)


def load(name, ids=None, positive_only=False):
    """Load a dataset by name. Tries OpenML by name, then by id, then PMLB.

    positive_only drops non-positive targets, which the tail estimators need but
    which changes n, so it is off by default and requested explicitly.

    What was loaded, and how, is left in `LAST`: the source, whether the
    dataset's own target was used or the fallback picked a column, how many
    columns were categorical, and a fingerprint of the table. See `LAST` above
    for why each of those matters to a published number.
    """
    LAST.clear()
    LAST.update(name=name, source="", target_source="default",
                n_categorical=0, categorical=[])
    X, y = _load(name, ids)
    check_fingerprint(name, X, y)
    if positive_only:
        keep = y > 0
        LAST["dropped_non_positive"] = int((~keep).sum())
        X, y = X[keep], y[keep]
    return X, y


def _load(name, ids=None):
    ids = openml_ids() if ids is None else ids

    if name == "freMTPL2sev":
        LAST["source"] = "openml-joined"
        return prepare(*_load_fremtpl2())

    from common import external
    if name in external.REGISTRY:
        LAST["source"] = "external"
        return prepare(*external.load(name))

    if re.match(r"^\d+_", name):
        LAST["source"] = "pmlb"
        return prepare(*_load_pmlb(name))

    # Every candidate's failure is kept. Keeping only the last one reports the
    # error of the as_frame=False fallback, which for a table with string columns
    # is always "try as_frame=True" and hides why the as_frame=True attempt, made
    # first, actually failed. That cost one dataset a silent drop from the wide
    # leverage sweep.
    failures, d = [], None
    candidates = [{"name": name}]
    if name in ALIASES:
        candidates.insert(0, {"data_id": ALIASES[name]})
    if name in ids:
        candidates.append({"data_id": ids[name]})
    candidates.append({"name": name, "as_frame": False})   # rare ARFF
    for kw in candidates:
        tried = dict(kw)
        try:
            as_frame = kw.pop("as_frame", True)
            d = fetch_openml(as_frame=as_frame, parser="auto", **kw)
            LAST["source"] = "openml-id" if "data_id" in kw else "openml-name"
            if not as_frame:
                raw = d.data.todense() if hasattr(d.data, "todense") else d.data
                d.data = pd.DataFrame(np.asarray(raw))
            break
        except Exception as e:
            failures.append(f"{tried}: {type(e).__name__}: {e}")
            d = None
    if d is None:
        try:
            LAST["source"] = "pmlb"
            return prepare(*_load_pmlb(name))
        except Exception as e:
            failures.append(f"pmlb: {type(e).__name__}: {e}")
            raise RuntimeError(
                "cannot load %s; every source failed:\n  %s"
                % (name, "\n  ".join(failures)))

    X = d.data.copy()
    y = pd.to_numeric(pd.Series(np.asarray(d.target).ravel()),
                      errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).any():
        # The default target is not numeric, e.g. a date or a category. Take the
        # last numeric column instead: ScoringBench runs regression, so a numeric
        # target has to exist somewhere in the frame.
        #
        # This is a silent choice of what is being predicted, and it reaches a
        # headline number: the survey of 99 datasets, from which the base rate of
        # leverage comes, runs over whatever this picked. `LAST["target_source"]`
        # records it so the survey can report how many datasets went this way.
        num = X.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise RuntimeError("no numeric column")
        LAST["target_source"] = f"last-numeric-column:{num[-1]}"
        y = X.pop(num[-1]).to_numpy(dtype=float)
    return prepare(X, y)


def target_groups() -> dict:
    """Dataset name to the identity of its target vector.

    Two OpenML names can serve the same target under different feature sets.
    `218_house_8L` and `house_16H` are the pair in this repository: 22.784 rows,
    the same `y_sha1`, 8 columns against 16. A per-dataset sign test that counts
    both is counting one free unit twice, which is rule 6 one level up from the
    seed level where it was first caught.

    The grouping key is the recorded target hash, so a name with no fingerprint
    yet maps to itself and forms its own group. That keeps the function usable
    on result frames written before the fingerprints existed.
    """
    try:
        path = paths.data(FINGERPRINTS)
    except Exception:
        path = paths.path(FINGERPRINTS, "data")
    if not path.exists():
        return {}
    known = json.loads(path.read_text(encoding="utf-8"))
    return {name: fp["y_sha1"] for name, fp in known.items() if "y_sha1" in fp}


def target_group(name: str) -> str:
    """The target identity of one dataset, or its own name when unknown."""
    return target_groups().get(name, name)


def duplicate_targets() -> dict:
    """Target hash to the dataset names that share it, for hashes shared by more
    than one name. Empty when every recorded dataset has its own target."""
    out: dict = {}
    for name, key in target_groups().items():
        out.setdefault(key, []).append(name)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}
