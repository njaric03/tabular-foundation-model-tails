# -*- coding: utf-8 -*-
"""The resumable measurement loop the experiments share.

    COLUMNS = ["xi", "model", "seed", "n_est", "xi_implied", "seconds", "reason"]
    KEY = ["xi", "model", "seed", "n_est"]

    def measure(cell):
        ...
        return dict(xi_implied=...)

    cells = runner.grid(xi=XI, model=MODELS, seed=SEEDS, n_est=[N_EST])
    runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
               show=lambda row: f"xi {row['xi_implied']:+.3f}")

A cell whose key is already in the file is skipped. An exception inside `measure` is
written as that row's `reason`, so one failing cell does not stop the run.
"""
from __future__ import annotations

import itertools
import time

from common import append

REASON_CHARS = 110


def grid(**axes) -> list[dict]:
    """Every combination of the axes, the first axis outermost."""
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


def reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:REASON_CHARS]


def run(output, columns, key, cells, measure, show=None) -> float:
    """Measure and append every cell not yet in `output`. Returns the elapsed seconds.

    `measure(cell)` returns the measured columns; the cell's own fields are added to
    the row afterwards, except those starting with `_`, which only carry data to
    `measure`. `show(row)` formats the progress line of a successful cell.
    """
    done = append.done(output, key)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    t0 = time.time()
    for cell in cells:
        if append.key(cell, key) in done:
            continue
        t1 = time.time()
        try:
            row = dict(measure(cell))
            row.setdefault("reason", "")
        except Exception as e:
            row = dict(reason=reason(e))
        row.update({k: v for k, v in cell.items() if not k.startswith("_")},
                   seconds=round(time.time() - t1, 1))
        append.write(output, row, columns)
        label = "  ".join(f"{k}={cell[k]}" for k in key if k in cell)
        detail = row["reason"] or (show(row) if show else "")
        print(f"  {label}  {detail}  [{row['seconds']}s]", flush=True)
    return time.time() - t0
