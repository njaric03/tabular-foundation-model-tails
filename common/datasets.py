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

import re

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from common import paths

# Names that live under a different name, or only under an id, on OpenML.
ALIASES = {"Ele2": 42362}

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
    """Categories to codes, missing values filled, invalid targets dropped."""
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any():
        raise RuntimeError("target is not numeric")
    X = X.copy()
    for c in X.select_dtypes(exclude="number").columns:
        X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    ok = np.isfinite(y)
    return X.to_numpy(dtype=float)[ok], y[ok]


def load(name, ids=None, positive_only=False):
    """Load a dataset by name. Tries OpenML by name, then by id, then PMLB.

    positive_only drops non-positive targets, which the tail estimators need but
    which changes n, so it is off by default and requested explicitly.
    """
    ids = openml_ids() if ids is None else ids

    if name == "freMTPL2sev":
        X, y = prepare(*_load_fremtpl2())
        return (X[y > 0], y[y > 0]) if positive_only else (X, y)

    if re.match(r"^\d+_", name):
        X, y = prepare(*_load_pmlb(name))
        return (X[y > 0], y[y > 0]) if positive_only else (X, y)

    last, d = None, None
    candidates = [{"name": name}]
    if name in ALIASES:
        candidates.insert(0, {"data_id": ALIASES[name]})
    if name in ids:
        candidates.append({"data_id": ids[name]})
    candidates.append({"name": name, "as_frame": False})   # rare ARFF
    for kw in candidates:
        try:
            as_frame = kw.pop("as_frame", True)
            d = fetch_openml(as_frame=as_frame, parser="auto", **kw)
            if not as_frame:
                raw = d.data.todense() if hasattr(d.data, "todense") else d.data
                d.data = pd.DataFrame(np.asarray(raw))
            break
        except Exception as e:
            last, d = e, None
    if d is None:
        try:
            X, y = prepare(*_load_pmlb(name))
            return (X[y > 0], y[y > 0]) if positive_only else (X, y)
        except Exception:
            raise RuntimeError(f"cannot load {name} ({type(last).__name__}: {last})")

    X = d.data.copy()
    y = pd.to_numeric(pd.Series(np.asarray(d.target).ravel()),
                      errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).any():
        # The default target is not numeric, e.g. a date or a category. Take the
        # last numeric column instead: ScoringBench runs regression, so a numeric
        # target has to exist somewhere in the frame.
        num = X.select_dtypes(include=[np.number]).columns
        if len(num) == 0:
            raise RuntimeError("no numeric column")
        y = X.pop(num[-1]).to_numpy(dtype=float)
    X, y = prepare(X, y)
    return (X[y > 0], y[y > 0]) if positive_only else (X, y)


def subsample(X, y, n, seed=0):
    """Deterministic subsample, for datasets too large to fit in context."""
    if len(y) <= n:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]
