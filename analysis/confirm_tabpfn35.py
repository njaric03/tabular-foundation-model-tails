# -*- coding: utf-8 -*-
"""Verdicts of the predictions of 21.10.2026 for TabPFN-3.5 on the confirmation draws.

Reads the outputs of `run/confirm_tabpfn35.sh`. Written with the predictions in
PREREGISTRATION.md, before any of the runs; a file that is not there yet is skipped.

    python -m analysis.confirm_tabpfn35
"""
from analysis.confirm_ensemble_log import CLEAN, SHIFT, load, seg_rel, unit_rel, verdict

M = "TabPFN-v3.5"


def tables(u):
    return ", ".join(f"{k} {v:+.2%}" for k, v in u.items())


def main():
    s1, s8 = load("legit_segment_tabpfn35_nest1.csv"), load("legit_segment_tabpfn35_nest8.csv")
    u8 = load("unit_error_confirm_tabpfn35_nest8.csv")

    rise = {k: seg_rel(s, "epb99_all", 1.0, SHIFT, "raw", "raw", CLEAN).xs(M)
            for k, s in ((1, s1), (8, s8)) if s is not None}
    if 1 in rise:
        r = rise[1]
        verdict("F1", (r > 0).sum() >= 15 and r.median() >= 0.05,
                f"rise {(r > 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
    if 1 in rise and 8 in rise:
        smaller = int((rise[8] < rise[1].reindex(rise[8].index)).sum())
        verdict("F2", smaller >= 14, f"{smaller} of {len(rise[8])} seeds smaller at eight "
                                     f"members (median {rise[8].median():+.2%})")
    if s8 is not None:
        r = seg_rel(s8, "epb99_all", 1.0, SHIFT, "clip", "raw").xs(M)
        verdict("F3a", (r < 0).sum() >= 15,
                f"{(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
    if u8 is not None:
        u = unit_rel(u8, "pb99", "visible", "clip", "raw").xs(M, level="model")
        verdict("F3b", (u < 0).sum() >= 4, f"{(u < 0).sum()} of {len(u)} tables: {tables(u)}")
        u = unit_rel(u8, "pb99", "visible", "log", "clip").xs(M, level="model")
        verdict("F4a", (u < 0).sum() >= 4, f"{(u < 0).sum()} of {len(u)} tables: {tables(u)}")
    if s8 is not None:
        r = seg_rel(s8, "epb99_all", 1.0, CLEAN, "log", "raw").xs(M)
        verdict("F4b", (r < 0).sum() >= 15,
                f"{(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
        q = s8[(s8.model == M) & (s8.seg_scale == 20.0) & (s8.sd_shift_target == CLEAN)
               & (s8.variant == "log")].qratio999_seg
        verdict("F5", q.median() > 1.25, f"median {q.median():.2f} of the truth, {len(q)} seeds")


if __name__ == "__main__":
    main()
