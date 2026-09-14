# -*- coding: utf-8 -*-
"""Row-by-row append to a result CSV, resumable and guarded.

    from common import append

    COLUMNS = ["xi", "model", "seed", "n_est", "xi_implied", "reason"]
    KEY = ["xi", "model", "seed", "n_est"]

    done = append.done(OUTPUT, KEY)
    if append.key(row, KEY) not in done:
        append.write(OUTPUT, row, COLUMNS)

Each guard here was added after its absence cost a measurement:

* the header on disk must match the columns sent. A pure superset widens the file,
  anything else raises `Shifted`;
* rows are written under a file lock, so two processes cannot interleave them;
* `done` raises when a key column is missing, or empty in every row, since a re-run
  with another value would then skip or duplicate the work without a word.
"""
from __future__ import annotations

import pandas as pd

from common import files, paths, provenance


class Shifted(Exception):
    """The header on disk does not match the columns the script sends."""


def _path(output):
    return paths.result(output)


def _reconcile(p, columns: list[str]) -> None:
    """Check the header and widen the file when the new columns are a superset.

    Old rows get blanks in the new columns, which is true: the parameter was not
    recorded then. A renamed or dropped column cannot be reconciled and raises.
    """
    old = files.csv_header(p)
    if old is None or old == list(columns):
        return
    missing = [c for c in old if c not in columns]
    if missing:
        raise Shifted(
            f"{p.name}: the file has columns the script does not send: {missing}\n"
            f"  on disk: {old}\n"
            f"  script : {list(columns)}\n"
            f"Appending would shift the columns. Restore them, or write a new OUTPUT.")
    n = files.rewrite_csv(p, columns)
    added = [c for c in columns if c not in old]
    if added:
        print(f"  [append] {p.name}: added columns {added}, {n} existing rows left blank",
              flush=True)


def write(output, row: dict, columns: list[str]) -> None:
    """Append one row under a lock. The first write of a process also records provenance."""
    p = _path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    provenance.record(p.name)
    with files.FileLock(p):
        _reconcile(p, columns)
        pd.DataFrame([row], columns=list(columns)).to_csv(
            p, mode="a", index=False, header=files.csv_header(p) is None)


def replace(output, frame: pd.DataFrame) -> None:
    """Write a whole result file at once, for scripts that recompute everything in
    minutes. Records provenance like `write`."""
    p = _path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    provenance.record(p.name)
    with files.FileLock(p):
        frame.to_csv(p, index=False)


def key(row: dict, columns: list[str]) -> tuple:
    """Resume key of a row. 1 and 1.0 match, and so do None and an empty cell."""
    return tuple(_norm(row.get(c)) for c in columns)


def _norm(v):
    if v is None or v is pd.NA or isinstance(v, bool):
        return None if v is pd.NA else v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    # An empty cell reads back as NaN, and NaN equals nothing, not even itself.
    return None if f != f else round(f, 9)


def done(output, columns: list[str]) -> set[tuple]:
    """Keys already present in the file.

    Raises when a key column is absent, or present but empty in every row. Either
    way the old rows were written without the parameter, so a re-run cannot tell
    them apart from new ones.
    """
    p = _path(output)
    if files.csv_header(p) is None:
        return set()
    d = pd.read_csv(p, low_memory=False)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        raise Shifted(
            f"{p.name}: key columns {missing} are absent, so it is unknown which "
            f"parameters the existing rows were measured with.\n"
            f"Either write a new OUTPUT, or fill the column in if the values are known.")
    empty = [c for c in columns if len(d) and d[c].isna().all()]
    if empty:
        raise Shifted(
            f"{p.name}: key columns {empty} exist but are empty in every row, so the "
            f"script keys on a parameter it never writes.\n"
            f"Write the column, then backfill the existing rows with the value they "
            f"were measured at.")
    return {key(r, columns) for r in d[list(columns)].to_dict("records")}
