# -*- coding: utf-8 -*-
"""Verdicts of the predictions of 20.10.2026: TabPFN-V3's default ensemble and a log arm.

Reads the four outputs of `run/confirm_ensemble_log.sh` and, for the reproduction check,
the outputs of 1.10.2026. Written with the predictions in PREREGISTRATION.md, before any
of the runs; a file that is not there yet is skipped.

    python -m analysis.confirm_ensemble_log
"""
import pandas as pd

from common import paths, tables

FM = ("TabPFN-V3", "TabICLv2")
CLEAN, SHIFT = 1.0, 20.0


def load(name):
    path = paths.result(f"h3_repair/{name}")
    if not path.exists():
        print(f"{name}: not there yet")
        return None
    return tables.ok_rows(pd.read_csv(path))


def unit_rel(d, level, context, a, b):
    """Per (dataset, model): the median over repeats of a's loss over b's, minus one."""
    w = d.pivot_table(index=["dataset", "model", "repeat"], columns=["context", "arm"],
                      values=level)
    return (w[(context, a)] / w[(context, b)] - 1).groupby(level=["dataset", "model"]).median()


def seg_rel(d, col, M, shift, a, b, shift_b=None):
    """Per (model, seed): a's expected loss over b's, minus one."""
    w = d[d.seg_scale == M].pivot_table(index=["model", "seed"],
                                        columns=["sd_shift_target", "variant"], values=col)
    return w[(shift, a)] / w[(shift if shift_b is None else shift_b, b)] - 1


def verdict(name, ok, detail):
    print(f"{name:4s} {'HOLDS' if ok else 'FAILS'}  {detail}")


def same(new, old, key, cols, label):
    """Largest relative difference between the rows two runs share."""
    j = new.merge(old, on=key, suffixes=("", "_old"))
    gap = max(float(((j[c] - j[c + "_old"]).abs() / j[c + "_old"].abs()).max()) for c in cols)
    print(f"R0   {label}: {len(j)} shared rows, largest relative difference {gap:.2e}")


def main():
    u1, u8 = load("unit_error_confirm_log.csv"), load("unit_error_confirm_nest8.csv")
    s1, s8 = load("legit_segment_log.csv"), load("legit_segment_nest8.csv")

    if u1 is not None:
        same(u1[u1.arm != "log"], load("unit_error_confirm.csv"),
             ["dataset", "model", "repeat", "context", "arm", "clip_c"], ["pb99", "pb999"],
             "unit error, raw and clip_200 against 1.10.2026")
    if s1 is not None:
        same(s1[s1.variant != "log"], load("legit_segment.csv"),
             ["model", "seg_scale", "sd_shift_target", "variant", "clip_c", "seed"],
             ["epb99_all", "epb999_all"], "segment, raw and clip_200 against 1.10.2026")

    if s8 is not None:
        r8 = seg_rel(s8, "epb99_all", 1.0, SHIFT, "raw", "raw", CLEAN).xs("TabPFN-V3")
        verdict("E1", (r8 > 0).sum() >= 15 and r8.median() >= 0.02,
                f"rise {(r8 > 0).sum()} of {len(r8)} seeds, median {r8.median():+.2%}")
        if s1 is not None:
            r1 = seg_rel(s1, "epb99_all", 1.0, SHIFT, "raw", "raw", CLEAN).xs("TabPFN-V3")
            smaller = int((r8 < r1.reindex(r8.index)).sum())
            verdict("E2", smaller >= 14, f"{smaller} of {len(r8)} seeds smaller at eight "
                                         f"members (one member: median {r1.median():+.2%})")
        r = seg_rel(s8, "epb99_all", 1.0, SHIFT, "clip", "raw").xs("TabPFN-V3")
        verdict("E3a", (r < 0).sum() >= 18,
                f"{(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
    if u8 is not None:
        u = unit_rel(u8, "pb99", "visible", "clip", "raw").xs("TabPFN-V3", level="model")
        verdict("E3b", (u < 0).sum() >= 4, f"{(u < 0).sum()} of {len(u)} tables: "
                + ", ".join(f"{k} {v:+.2%}" for k, v in u.items()))

    if s1 is not None:
        for m in FM:
            r = seg_rel(s1, "epb99_all", 1.0, CLEAN, "log", "raw").xs(m)
            verdict("L1", (r < 0).sum() >= 15,
                    f"{m}: {(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
        r = seg_rel(s1, "epb99_all", 1.0, SHIFT, "log", "raw").xs("TabPFN-V3")
        verdict("L2", (r < 0).sum() >= 18,
                f"{(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
    if u1 is not None:
        u = unit_rel(u1, "pb99", "visible", "log", "raw").xs("TabICLv2", level="model")
        verdict("L3", (u < 0).sum() >= 3, f"{(u < 0).sum()} of {len(u)} tables: "
                + ", ".join(f"{k} {v:+.2%}" for k, v in u.items()))
        for m in FM:
            u = unit_rel(u1, "pb99", "visible", "log", "clip").xs(m, level="model")
            verdict("L4a", (u < 0).sum() >= 3, f"{m}: {(u < 0).sum()} of {len(u)} tables: "
                    + ", ".join(f"{k} {v:+.2%}" for k, v in u.items()))
    if s1 is not None:
        for m in FM:
            r = seg_rel(s1, "epb99_seg", 20.0, CLEAN, "log", "clip").xs(m)
            verdict("L4b", (r < 0).sum() >= 14,
                    f"{m}: {(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")


if __name__ == "__main__":
    main()
