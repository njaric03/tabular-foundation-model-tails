# -*- coding: utf-8 -*-
"""Verdicts of the predictions of 29.10.2026: the conditional cap.

Reads the two outputs of `run/conditional_cap.sh` and, for the reproduction check and the
log arm, the one-member outputs of 20.10.2026 on the same draws and seeds. Written with
the predictions in PREREGISTRATION.md, before any of the runs; a file that is not there
yet is skipped.

    python -m analysis.conditional_cap
"""
import pandas as pd

from analysis.confirm_ensemble_log import load, same, seg_rel, unit_rel, verdict

FM = ("TabPFN-V3", "TabICLv2")
CLEAN, SHIFT = 1.0, 20.0
SEG_KEY = ["model", "seg_scale", "sd_shift_target", "variant", "clip_c", "seed"]
UNIT_KEY = ["dataset", "model", "repeat", "context", "arm", "clip_c"]


def touched(d, arm_col, arm, **where):
    """Per model: in how many rows `arm` lowered the corrupted or added row, of how many."""
    m = d[arm_col] == arm
    for k, v in where.items():
        m &= d[k] == v
    t = d[m].row_touched.astype(str) == "True"
    return t.groupby(d[m].model).agg(["sum", "count"])


def listing(s):
    return ", ".join(f"{k} {v:+.2%}" for k, v in s.items())


def main():
    sc, s1 = load("legit_segment_cclip.csv"), load("legit_segment_log.csv")
    uc, u1 = load("unit_error_confirm_cclip.csv"), load("unit_error_confirm_log.csv")

    if sc is not None and s1 is not None:
        same(sc[sc.variant != "cclip"], s1[s1.variant != "log"], SEG_KEY,
             ["epb99_all", "epb999_all"], "segment, raw and clip_200 against 20.10.2026")
    if uc is not None and u1 is not None:
        same(uc[uc.arm != "cclip"], u1[u1.arm != "log"], UNIT_KEY, ["pb99", "pb999"],
             "unit error, raw and clip_200 against 20.10.2026")

    if sc is not None:
        seg = pd.concat([sc, s1[s1.variant == "log"]]) if s1 is not None else sc
        for m in FM:
            r = seg_rel(sc, "epb99_all", 1.0, CLEAN, "cclip", "raw").xs(m)
            verdict("U0", abs(r.median()) < 0.02,
                    f"{m}: median {r.median():+.2%}, lower in {(r < 0).sum()} of {len(r)}")
        low = sc[(sc.variant == "cclip") & (sc.seg_scale == 20.0)
                 & (sc.sd_shift_target == CLEAN)]
        for m in FM:
            n = low[low.model == m][["n_clipped", "n_clipped_seg"]].median()
            verdict("U1", n.n_clipped_seg <= 1,
                    f"{m}: median {n.n_clipped_seg:g} segment rows of {n.n_clipped:g} lowered")
        for m in FM:
            r = seg_rel(sc, "epb999_seg", 20.0, CLEAN, "cclip", "clip").xs(m)
            verdict("U2", (r < 0).sum() >= 14,
                    f"{m}: {(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
        for M in (1.0, 20.0):
            t = touched(sc, "variant", "cclip", seg_scale=M, sd_shift_target=SHIFT)
            for m in FM:
                verdict("U3", t.loc[m, "sum"] >= 18,
                        f"{m}, M = {M:g}: added row lowered in {t.loc[m, 'sum']} of "
                        f"{t.loc[m, 'count']} seeds")
        for label, M, need in (("U4a", 1.0, 18), ("U4b", 20.0, 15)):
            r = seg_rel(sc, "epb99_all", M, SHIFT, "cclip", "raw").xs("TabPFN-V3")
            verdict(label, (r < 0).sum() >= need,
                    f"M = {M:g}: {(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")
        for m in FM:
            r = seg_rel(sc, "epb999_seg", 20.0, SHIFT, "cclip", "clip").xs(m)
            verdict("U5", (r < 0).sum() >= 14,
                    f"{m}: {(r < 0).sum()} of {len(r)} seeds, median {r.median():+.2%}")

        print("\nReported without a prediction, segment, medians over seeds:")
        for col in ("epb99_all", "epb999_all", "epb99_seg", "epb999_seg"):
            for M in (1.0, 5.0, 20.0):
                for shift in (CLEAN, SHIFT):
                    for b in ("raw", "clip", "log"):
                        if b not in set(seg.variant):
                            continue
                        r = seg_rel(seg, col, M, shift, "cclip", b)
                        print(f"  {col:11s} M {M:4g} shift {shift:4g} cclip/{b:4s} "
                              + "  ".join(f"{k} {v:+.2%}" for k, v
                                          in r.groupby(level="model").median().items()))

    if uc is not None:
        for m in FM:
            r99 = unit_rel(uc, "pb99", "clean", "cclip", "raw").xs(m, level="model")
            r999 = unit_rel(uc, "pb999", "clean", "cclip", "raw").xs(m, level="model")
            verdict("U6", bool((r99.abs() <= 0.02).all() and (r999 <= 0.10).all()),
                    f"{m}: 0.99 {listing(r99)}; 0.999 {listing(r999)}")
        u = unit_rel(uc, "pb99", "visible", "cclip", "raw").xs("TabPFN-V3", level="model")
        verdict("U7", (u < 0).sum() >= 4, f"{(u < 0).sum()} of {len(u)} tables: {listing(u)}")

        print("\nReported without a prediction, unit error:")
        for cx in ("visible", "random"):
            for b in ("clip",) + (("log",) if u1 is not None else ()):
                d = pd.concat([uc, u1[u1.arm == "log"]]) if b == "log" else uc
                r = unit_rel(d, "pb99", cx, "cclip", b)
                print(f"  {cx:7s} cclip/{b:4s} 0.99: {listing(r)}")
        for cx in ("visible", "random"):
            for a in ("clip", "cclip"):
                t = uc[(uc.arm == a) & (uc.context == cx)]
                share = (t.row_touched.astype(str) == "True").groupby(
                    [t.dataset, t.model]).mean()
                print(f"  {cx:7s} {a:5s} corrupted row lowered: "
                      + ", ".join(f"{k[0]}/{k[1]} {v:.0%}" for k, v in share.items()))
        clean = uc[uc.context == "clean"]
        fired = (clean.n_touched > 0).groupby([clean.dataset, clean.arm]).mean().unstack()
        print("\n  share of clean contexts touched, by arm:")
        print(fired.round(2).to_string())


if __name__ == "__main__":
    main()
