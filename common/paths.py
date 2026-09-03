# -*- coding: utf-8 -*-
"""Repo-root anchor for every file the experiments read or write.

Scripts used to resolve paths relative to the working directory, so everything
had to live in one folder and be launched from it. Here paths are resolved
against the repo root, found by walking up from ``__file__`` to the directory
holding ``.git``.

    from common import paths

    ids = paths.load_json("sb_openml_ids.json")
    df.to_csv(paths.result("h1_shape_vs_scale/shape_of_x_20.csv"), index=False)

Lookup is recursive within each category, so a bare basename is enough and the
caller does not need to know which subfolder a file sits in. A basename that
matches in two places raises rather than picking one, because silently reading
the wrong CSV is how measurements get mixed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    """First directory upwards that contains `.git`."""
    p = Path(__file__).resolve()
    for candidate in p.parents:
        if (candidate / ".git").exists():
            return candidate
    return p.parents[1]


ROOT = _repo_root()

# Where each category is searched, in order. The repo root is always last so a
# file that has not been moved yet is still found.
SEARCH = {
    "module":  ["experiments", "common", "."],
    "data":    ["data", "."],
    "result":  ["results", "."],
    "finding": ["findings", "."],
    "figure":  ["figures", "."],
}

_module_cache: dict[str, object] = {}


def _find(name: str, kind: str) -> Path:
    name = str(name)
    p = Path(name)
    # An absolute path, or one carrying a subfolder, is taken as given.
    if p.is_absolute():
        return p
    if len(p.parts) > 1:
        for d in SEARCH[kind]:
            candidate = ROOT / d / p
            if candidate.exists():
                return candidate
        return ROOT / SEARCH[kind][0] / p

    for d in SEARCH[kind]:
        base = ROOT / d
        if not base.is_dir():
            continue
        direct = base / name
        if direct.exists():
            return direct
        hits = sorted(base.rglob(name))
        if len(hits) > 1:
            raise FileNotFoundError(
                f"{name!r} matches {len(hits)} files under {d}/: "
                + ", ".join(str(h.relative_to(ROOT)) for h in hits)
                + "\nBasenames must be unique; pass a path with its subfolder.")
        if hits:
            return hits[0]

    # Does not exist yet: return where it should be written.
    first = ROOT / SEARCH[kind][0]
    if not first.is_dir():
        return ROOT / name
    # results/ mirrors experiments/, so the first write of a new CSV belongs in the
    # subfolder named after the script writing it, not in the root of results/.
    # Without this every new measurement lands one level too high and has to be
    # moved by hand; coverage.csv and influence_sd.csv both did.
    if kind == "result":
        argv0 = sys.argv[0] if sys.argv else ""
        if argv0:
            sub = Path(argv0).resolve().parent.name
            if sub and (first / sub).is_dir():
                return first / sub / name
    return first / name


def path(name: str, kind: str = "result") -> Path:
    """Full path for a file of the given kind: module | data | result | finding | figure."""
    return _find(name, kind)


def data(name: str) -> Path:
    """Input data (sb_*.json, selected_datasets.txt)."""
    return _find(name, "data")


def result(name: str) -> Path:
    """Result CSV, for both reading and writing."""
    return _find(name, "result")


def finding(name: str) -> Path:
    """A findings/*.md document."""
    return _find(name, "finding")


def figure(name: str) -> Path:
    """A figure for the thesis."""
    return _find(name, "figure")


def load_json(name: str):
    with open(_find(name, "data"), encoding="utf-8") as f:
        return json.load(f)


def lines(name: str) -> list[str]:
    """Non-empty stripped lines of a text input (e.g. selected_datasets.txt)."""
    with open(_find(name, "data"), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def module(name: str):
    """Load a repo script as a module, independent of the working directory. Cached.

    Only needed for the few scripts that reuse another experiment's helpers;
    shared code belongs in `common`.
    """
    p = _find(name, "module")
    key = str(p)
    if key in _module_cache:
        return _module_cache[key]
    if not p.exists():
        raise FileNotFoundError(f"module {name} not found (searched {SEARCH['module']})")
    mod_name = "repo_" + p.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, p)
    m = importlib.util.module_from_spec(spec)
    # Register before exec so a module importing itself does not loop.
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    _module_cache[key] = m
    return m
