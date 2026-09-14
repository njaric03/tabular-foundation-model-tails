# -*- coding: utf-8 -*-
"""How far the seeding fix moved TabPFN's numbers.

Run after `run/tabpfn_reseed.sh`. The old files gave TabPFN `random_state=0`, so its
ensemble permutations were frozen across seeds; the new ones pass the seed. Two things
per measure: the share of the true change, and its spread over seeds, which the freeze
suppressed most directly.

The scale comparison at n_est=1 is not a seeding comparison: `scale_of_x.csv` was
measured at n_est=4, so that row measures the ensemble size and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import paths

MODEL = "TabPFN-V3"
# (label, old file, new file, measure, note)
PAIRS = [
    ("shape n_est=4", "shape_of_x_20.csv", "shape_of_x_tabpfn_reseed_20.csv", "shape", ""),
    ("shape n_est=1", "shape_of_x_nest1.csv", "shape_of_x_tabpfn_reseed_nest1.csv", "shape", ""),
    ("scale n_est=4", "scale_of_x.csv", "scale_of_x_tabpfn_reseed_20.csv", "scale", ""),
    ("scale n_est=1", "scale_of_x.csv", "scale_of_x_tabpfn_reseed_nest1.csv", "scale",
     "the old file is n_est=4, so this row measures the ensemble size, not the seeding"),
]


def shares(name: str, measure: str) -> pd.Series | None:
    """The share of the true change per seed, or None when the file is missing."""
    path = paths.result(name)
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d = d[d.model == MODEL]
    if d.empty:
        return None
    if measure == "shape":
        p = d.pivot_table(index="seed", columns="tercile", values=["xi_implied", "xi_true"])
        model = p[("xi_implied", 3)] - p[("xi_implied", 1)]
        true = p[("xi_true", 3)] - p[("xi_true", 1)]
    else:
        p = d.pivot_table(index="seed", columns="tercile", values=["median_model", "median_true"])
        model = np.log(p[("median_model", 3)] / p[("median_model", 1)])
        true = np.log(p[("median_true", 3)] / p[("median_true", 1)])
    return (model / true).replace([np.inf, -np.inf], np.nan).dropna()


def main() -> int:
    print(f"=== {MODEL}: random_state=0 (old) against random_state=seed (new) ===\n")
    print(f"{'measure':16s}{'old share':>11s}{'new share':>11s}{'change':>9s}"
          f"{'old sd':>9s}{'new sd':>9s}")
    notes = []
    for label, old_name, new_name, measure, note in PAIRS:
        old, new = shares(old_name, measure), shares(new_name, measure)
        if old is None or new is None:
            print(f"{label:16s}  {'missing: ' + (old_name if old is None else new_name)}")
            continue
        mark = ""
        if note:
            notes.append(note)
            mark = f"  ({len(notes)})"
        print(f"{label:16s}{old.mean():>10.0%}{new.mean():>11.0%}{new.mean() - old.mean():>+9.0%}"
              f"{old.std():>9.2f}{new.std():>9.2f}{mark}")
    for i, note in enumerate(notes, 1):
        print(f"\n  ({i}) {note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
