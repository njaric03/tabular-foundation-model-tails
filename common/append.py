# -*- coding: utf-8 -*-
"""Safe row-by-row append to a result CSV.

Long measurements are resumable: they append one row per (model, seed, ...) cell
and skip cells already present. Hand-rolled `to_csv(mode="a")` had three holes,
each of which cost measurements in this project:

1. SHIFTED COLUMNS. `header=not exists` writes no header when the file exists,
   without checking that the column list still matches. When a script gained a
   parameter, the file stayed on the old columns while the script started
   sending more fields.
2. TWO PROCESSES RACING. Two streams writing one output interleave rows and
   formats.
3. SILENT SKIPPING. When the resume key is typed out by hand, a newly added
   knob that is not in the key makes a re-run with a different value find
   "already done" and do nothing, without a message.

    from common import append

    COLUMNS = ["xi", "model", "dose", "seed", "position", "n_train", "n_est", ...]
    KEY     = ["xi", "model", "dose", "seed", "position", "n_train", "n_est"]

    done = append.done(OUTPUT, KEY)
    ...
    if append.key(row, KEY) in done:
        continue
    append.write(OUTPUT, row, COLUMNS)

`done` and `write` share the key list, so a parameter that reaches the CSV but
not the key can no longer pass unnoticed: `done` raises if a key column is
missing from the file.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pandas as pd

from common import paths

# How long to wait for someone else's lock before giving up.
WAIT_S = 120


class Shifted(Exception):
    """The header on disk does not match the columns the script is sending."""


def _path(p) -> Path:
    return Path(p) if len(Path(p).parts) > 1 and Path(p).is_absolute() else paths.result(p)


class _Lock:
    """File lock. `O_EXCL` is atomic on both Windows and POSIX."""

    def __init__(self, target: Path):
        self.path = target.with_suffix(target.suffix + ".lock")

    def __enter__(self):
        deadline = time.time() + WAIT_S
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"lock {self.path.name} held by someone else for {WAIT_S}s. "
                        f"Most likely a second process writing the same OUTPUT; each "
                        f"stream needs its own. If a process died, delete the lock file.")
                time.sleep(0.1)

    def __exit__(self, *_):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def _header(p: Path) -> list[str] | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    with open(p, encoding="utf-8", newline="") as f:
        return next(csv.reader(f), None)


def _reconcile(p: Path, columns: list[str]) -> None:
    """Check the header; widen the file when the new columns are a pure superset.

    Widening is safe: old rows get blanks in the new columns, which is the truth
    (the parameter was not recorded then). Any other difference is an error, as a
    renamed, dropped or reordered column cannot be reconciled automatically.
    """
    old = _header(p)
    if old is None or old == list(columns):
        return

    missing = [c for c in old if c not in columns]
    if missing:
        raise Shifted(
            f"{p.name}: the file has columns the script does not send: {missing}\n"
            f"  on disk: {old}\n"
            f"  script : {list(columns)}\n"
            f"Appending would shift the columns. Restore them, or write a new OUTPUT.")

    d = pd.read_csv(p)
    for c in columns:
        if c not in d.columns:
            d[c] = pd.NA
    d[list(columns)].to_csv(p, index=False)
    added = [c for c in columns if c not in old]
    print(f"  [append] {p.name}: added columns {added}, {len(d)} existing rows "
          f"filled with blanks", flush=True)


def write(output, row: dict, columns: list[str]) -> None:
    """Append one row, under a lock, after reconciling the header."""
    p = _path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _Lock(p):
        _reconcile(p, columns)
        empty = not p.exists() or p.stat().st_size == 0
        pd.DataFrame([row], columns=list(columns)).to_csv(
            p, mode="a", index=False, header=empty)


def key(row: dict, columns: list[str]) -> tuple:
    """Resume key from a row. Numbers are normalised so 1 and 1.0 match."""
    return tuple(_norm(row.get(c)) for c in columns)


def _norm(v):
    if isinstance(v, bool) or v is None:
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return round(f, 9)


def done(output, columns: list[str]) -> set[tuple]:
    """Keys already present in the file.

    Raises when a key column is absent: that means earlier rows were written
    without the parameter, so old and new rows cannot be told apart. Failing is
    better than skipping silently.
    """
    p = _path(output)
    if not p.exists() or p.stat().st_size == 0:
        return set()
    d = pd.read_csv(p)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        raise Shifted(
            f"{p.name}: key columns {missing} are absent, so it is unknown which "
            f"parameters the existing rows were measured with.\n"
            f"Either write a new OUTPUT, or fill the column in if the values are known.")
    return {key(r, columns) for r in d[list(columns)].to_dict("records")}
