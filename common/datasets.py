# -*- coding: utf-8 -*-
"""Real tabular datasets: loading, preparation and fingerprints.

    from common import datasets

    X, y = datasets.load("diamonds")
    X, y = datasets.load("freMTPL2sev", positive_only=True)

`load` reads the external tables converted by `common/external.py`, tries OpenML by
name and by id, and falls back to PMLB. What it did is left in `LAST`: the source,
whether the dataset's own target was used, which columns were categorical, and a
fingerprint checked against `data/dataset_fingerprints.json`.
"""
from __future__ import annotations

import functools
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from common import env, files, paths

# Names that resolve on OpenML only under an id.
ALIASES = {"Ele2": 42362}

# The ten tables of the real-data dissociation (H1). `218_house_8L` and `house_16H`
# serve one target vector under two feature sets.
DISSOCIATION = ["OnlineNewsPopularity", "diamonds", "particulate-matter-ukair-2017",
                "Buzzinsocialmedia_Twitter", "CPS1988", "218_house_8L",
                "superconduct", "houses", "Allstate_Claims_Severity", "house_16H"]

# What the last `load` did, reset on every call. `target_source` says whether the
# dataset's own target was used or the last numeric column; `categorical` lists the
# columns coded from categories, which reach the models as plain numbers.
LAST: dict = {}

FINGERPRINTS = "dataset_fingerprints.json"

PMLB_URL = ("https://github.com/EpistasisLab/pmlb/raw/master/datasets/"
            "{name}/{name}.tsv.gz")


def scoringbench_names() -> list[str]:
    """The ScoringBench suite, about 102 dataset names."""
    return paths.load_json("sb_datasets.json")


def openml_ids() -> dict:
    """OpenML ids of the ScoringBench names that do not resolve by name."""
    return paths.load_json("sb_openml_ids.json")


def selected_names() -> list[str]:
    """Heavy-tailed tables that passed `experiments/datasets/dataset_selection.py`."""
    return paths.lines("selected_datasets.txt")


def external_names() -> list[str]:
    """External tables that passed `experiments/h2_leverage/external_selection.py`."""
    return paths.lines("external_selected.txt")


def _load_pmlb(name):
    """PMLB tables, named like 215_2dplanes, which OpenML does not carry under that name."""
    df = pd.read_csv(PMLB_URL.format(name=name), sep="\t", compression="gzip")
    y = pd.to_numeric(df.pop("target"), errors="coerce").to_numpy(dtype=float)
    return df, y


def _load_fremtpl2():
    """Claim severity summed per policy and joined to the policy covariates."""
    sev = fetch_openml("freMTPL2sev", as_frame=True, parser="auto").data
    freq = fetch_openml("freMTPL2freq", as_frame=True, parser="auto").data
    sev = sev.groupby("IDpol", as_index=False).ClaimAmount.sum()
    df = sev.merge(freq.drop(columns=["ClaimNb"]), on="IDpol", how="inner")
    y = df.pop("ClaimAmount").to_numpy(dtype=float)
    return df.drop(columns=["IDpol"]), y


def prepare(X, y):
    """Categories to integer codes, missing values filled, non-finite targets dropped.

    Two limitations, recorded rather than hidden: categorical columns reach the models
    as plain codes unless CATEGORICAL=native passes their indices on, and the median
    that fills missing values is taken over the whole table, before any split.
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
    """Shape and a hash of the target: enough to notice that a source changed.

    `fetch_openml` is called without a version, so the table is whatever OpenML
    serves that day.
    """
    y = np.asarray(y, dtype=float)
    return dict(n_rows=int(len(y)), n_cols=int(np.shape(X)[1]),
                y_sha1=hashlib.sha1(np.ascontiguousarray(y)).hexdigest()[:16],
                y_min=float(np.min(y)), y_max=float(np.max(y)))


def check_fingerprint(name: str, X, y) -> None:
    """Record the fingerprint on the first load, compare with it on every later one.

    A mismatch prints a warning, and raises under TFM_STRICT=1.
    """
    got = fingerprint(X, y)
    LAST["fingerprint"] = got
    path = paths.data(FINGERPRINTS)
    # Locked: two processes recording new datasets at once would each write the file
    # back without the other's entry.
    with files.FileLock(path):
        known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
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
        if env.flag("TFM_STRICT"):
            raise RuntimeError(message)
        print("  [datasets] WARNING " + message, flush=True)


def load(name, ids=None, positive_only=False):
    """Load a dataset by name.

    `positive_only` drops non-positive targets, which the tail estimators need but which
    changes n, so it is asked for explicitly.
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


@functools.lru_cache(maxsize=1)
def load_cached(name, positive_only=False):
    """`load`, remembering the last table, for loops that visit one dataset at a time.
    The arrays are shared between calls, so callers must not modify them in place."""
    return load(name, positive_only=positive_only)


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

    # Every candidate's failure is kept. The as_frame=False fallback always fails with
    # the same hint, so keeping only the last error hid why the first attempt failed.
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
        # The default target is not numeric, so take the last numeric column. That is a
        # choice of what is predicted, and it feeds the survey behind the base rate of
        # leverage, hence the record in LAST.
        num = X.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise RuntimeError("no numeric column")
        LAST["target_source"] = f"last-numeric-column:{num[-1]}"
        y = X.pop(num[-1]).to_numpy(dtype=float)
    return prepare(X, y)


def target_groups() -> dict:
    """Dataset name to the hash of its target vector.

    Two names can serve one target under different feature sets (`218_house_8L` and
    `house_16H`), and a sign test over names would count that unit twice (rule 6).
    """
    path = paths.data(FINGERPRINTS)
    if not path.exists():
        return {}
    known = json.loads(path.read_text(encoding="utf-8"))
    return {name: fp["y_sha1"] for name, fp in known.items() if "y_sha1" in fp}


def target_group(name: str) -> str:
    """The target hash of one dataset, or its name when no fingerprint is recorded."""
    return target_groups().get(name, name)


def duplicate_targets() -> dict:
    """Target hash to the names that share it, for hashes shared by more than one name."""
    out: dict = {}
    for name, key in target_groups().items():
        out.setdefault(key, []).append(name)
    return {k: sorted(v) for k, v in out.items() if len(v) > 1}
