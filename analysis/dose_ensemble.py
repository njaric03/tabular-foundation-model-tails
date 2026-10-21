# -*- coding: utf-8 -*-
"""Verdicts of the predictions of 22.10.2026: the dose curve of `clip_context.py` at one
and eight members for TabPFN-V3 and TabPFN-3.5, on the expected pinball.

Reads the four outputs of `run/dose_ensemble.sh`. Written with the predictions in
PREREGISTRATION.md, before any of the runs; a file that is not there yet is skipped.
A task is one (xi, seed) pair, 40 in all.

    python -m analysis.dose_ensemble
"""
import pandas as pd

from analysis.confirm_ensemble_log import verdict
from common import paths, tables

FILES = {("V3", 1): "clip_context_dose_v3_nest1.csv",
         ("V3", 8): "clip_context_dose_v3_nest8.csv",
         ("3.5", 1): "clip_context_dose_tabpfn35_nest1.csv",
         ("3.5", 8): "clip_context_dose_tabpfn35_nest8.csv"}


def load(name):
    path = paths.result(f"h3_repair/{name}")
    if not path.exists():
        print(f"{name}: not there yet")
        return None
    return tables.ok_rows(pd.read_csv(path))


def wide(d, col):
    """(xi, seed) by (sd_shift_target, variant)."""
    return d.pivot_table(index=["xi", "seed"], columns=["sd_shift_target", "variant"],
                         values=col)


def rise(w, shift):
    """raw at `shift` over raw on the clean context, minus one, per task."""
    return w[(shift, "raw")] / w[(1.0, "raw")] - 1


def arm(w, shift, a, b):
    return w[(shift, a)] / w[(shift, b)] - 1


def main():
    d = {k: load(f) for k, f in FILES.items()}
    w99 = {k: wide(v, "epb99") for k, v in d.items() if v is not None}
    w999 = {k: wide(v, "epb999") for k, v in d.items() if v is not None}

    if ("V3", 1) in w999:
        r = rise(w999[("V3", 1)], 50.0)
        verdict("D1", (r > 0).sum() >= 32 and r.median() >= 0.20,
                f"rise at 0.999 in {(r > 0).sum()} of {len(r)} tasks, median {r.median():+.1%}")
    if ("V3", 1) in w99 and ("V3", 8) in w99:
        for s in (20.0, 50.0):
            r1, r8 = rise(w99[("V3", 1)], s), rise(w99[("V3", 8)], s)
            smaller = int((r8 < r1.reindex(r8.index)).sum())
            verdict("D2", smaller >= 30, f"shift {s:g}: smaller at eight in {smaller} of "
                                         f"{len(r8)} tasks ({r1.median():+.1%} and "
                                         f"{r8.median():+.1%})")
    if ("3.5", 8) in w99:
        w = w99[("3.5", 8)]
        for s in (4.0, 20.0):
            for xi, r in rise(w, s).groupby(level="xi"):
                verdict("D3", abs(r.median()) < 0.03,
                        f"shift {s:g}, xi {xi}: median {r.median():+.2%}")
    for k in (("V3", 8), ("3.5", 8)):
        if k in w99:
            r = arm(w99[k], 20.0, "log", "clip")
            verdict("D4", (r < 0).sum() >= 28, f"{k[0]}: log below clip_200 in "
                                               f"{(r < 0).sum()} of {len(r)} tasks, "
                                               f"median {r.median():+.1%}")
    if ("V3", 8) in w999:
        r = arm(w999[("V3", 8)], 50.0, "clip", "raw")
        verdict("D5", (r < 0).sum() >= 30,
                f"{(r < 0).sum()} of {len(r)} tasks, median {r.median():+.1%}")


if __name__ == "__main__":
    main()
