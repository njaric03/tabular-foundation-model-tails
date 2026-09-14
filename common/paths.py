# -*- coding: utf-8 -*-
"""Paths of inputs, results and figures, resolved against the repository root.

    from common import paths

    ids = paths.load_json("sb_openml_ids.json")
    df.to_csv(paths.result("shape_of_x_20.csv"), index=False)

A bare file name is searched recursively within its kind, so a caller need not know the
subfolder; a name found twice raises rather than picking one. A result that does not
exist yet goes into the `results/` subfolder named after the script writing it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """The first directory upwards that contains `.git`."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists():
            return candidate
    return here.parents[1]


ROOT = _repo_root()

# Where each kind of file is searched, in order. The root comes last.
SEARCH = {
    "data": ["data", "."],
    "result": ["results", "."],
    "figure": ["figures", "."],
}


def _find(name, kind: str) -> Path:
    p = Path(str(name))
    if p.is_absolute():
        return p
    if len(p.parts) > 1:
        for d in SEARCH[kind]:
            if (ROOT / d / p).exists():
                return ROOT / d / p
        return ROOT / SEARCH[kind][0] / p

    for d in SEARCH[kind]:
        base = ROOT / d
        if not base.is_dir():
            continue
        if (base / p).exists():
            return base / p
        hits = sorted(base.rglob(p.name))
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{p.name!r} matches {len(hits)} files under {d}/: "
                + ", ".join(str(h.relative_to(ROOT)) for h in hits)
                + "\nPass the path with its subfolder.")
        if hits:
            return hits[0]

    # Not found: where it should be written.
    first = ROOT / SEARCH[kind][0]
    if not first.is_dir():
        return ROOT / p
    if kind == "result" and sys.argv and sys.argv[0]:
        sub = Path(sys.argv[0]).resolve().parent.name
        if sub and (first / sub).is_dir():
            return first / sub / p
    return first / p


def data(name) -> Path:
    """An input under data/."""
    return _find(name, "data")


def result(name) -> Path:
    """A result CSV under results/, for reading and for writing."""
    return _find(name, "result")


def figure(name) -> Path:
    """A figure under figures/."""
    return _find(name, "figure")


def load_json(name):
    with open(data(name), encoding="utf-8") as f:
        return json.load(f)


def lines(name) -> list[str]:
    """The non-empty, stripped lines of a text file under data/."""
    with open(data(name), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
