# -*- coding: utf-8 -*-
"""Verdicts of rules Y0 to Y3 of 28.10.2026: TabPFN-V3 at eight members with every member
on the untransformed target, against one member and eight at the default. Three measures
at a shift or dose of 20: Q, the median Q(0.99) on the test rows; R, raw's expected
pinball at 0.99 under the generator's law; G, the global change of Q(0.99).

Reads the outputs of `run/tabpfn_y_2810.sh`. Written with the rules, before any of the
runs; a file that is not there yet is skipped.

    python -m analysis.tabpfn_y
"""
import pandas as pd

from analysis.confirm_ensemble_log import verdict
from analysis.dose_ensemble import generator_law, load, rise, wide
from common import paths, tables

LOSS = {"1": "clip_context_ytf_nest1.csv", "8": "clip_context_ytf_nest8.csv",
        "8n": "clip_context_ytf_nest8_none.csv"}
LEVERAGE = {"1": "influence_sd_pos_centre.csv", "8": "influence_sd_tabpfn8.csv",
            "8n": "influence_sd_tabpfn8_none.csv"}


def loss_measures():
    """Per arm, the per-seed changes at a shift of 20 for Q, R and R at 0.999."""
    out = {}
    for a, f in LOSS.items():
        d = load(f)
        if d is None:
            continue
        if not {1.0, 20.0} <= set(d.sd_shift_target):
            print(f"{f}: not every shift yet")
            continue
        g = generator_law(d)
        q = wide(g, "q99")
        w99, w999 = wide(g, "epb99"), wide(g, "epb999")
        out[a] = dict(Q=q[(20.0, "raw")] / q[(1.0, "raw")] - 1, R=rise(w99, 20.0),
                      R999=rise(w999, 20.0), seconds=d.seconds.median(), n=len(q))
    return out


def leverage_measures():
    out = {}
    for a, f in LEVERAGE.items():
        path = paths.result(f"h2_leverage/{f}")
        if not path.exists():
            print(f"{f}: not there yet")
            continue
        d = tables.ok_rows(pd.read_csv(path))
        d = d[(d.model == "TabPFN-V3") & (d.xi == 0.7)]
        out[a] = {dose: d[d.dose == dose].global_q99 for dose in (4.0, 20.0)}
    return out


def share(x1, x8, x8n):
    return (x8n - x8) / (x1 - x8)


def judge(rule, name, x, sign):
    """Y0 for one measure, then its share; `sign` is the direction of the one-member
    effect (-1 for a fall, +1 for a rise). Returns the share or None."""
    m = {a: v.median() for a, v in x.items()}
    print(f"\n{name}: " + ", ".join(f"{a} {m[a]:+.1%} ({(v * sign > 0).sum()} of {len(v)} "
                                     f"in its direction, worst {(v * sign).min() * sign:+.1%})"
                                     for a, v in x.items()))
    if not {"1", "8", "8n"} <= set(m):
        print("     not every arm yet")
        return None
    gap = m["1"] * sign > 0 and m["8"] * sign < m["1"] * sign / 2
    verdict("Y0", gap, f"{name}: eight at the default {m['8']:+.1%} against "
                       f"{m['1'] / 2:+.1%}, half of one member")
    if not gap:
        print(f"     {name} has no gap to explain; no verdict")
        return None
    s = share(m["1"], m["8"], m["8n"])
    verdict(rule, s >= 0.5, f"{name}: s = {s:.2f}, eight without the transform "
                            f"{m['8n']:+.1%}")
    return s


def main():
    shares = []
    loss = loss_measures()
    if loss:
        shares.append(judge("Y1", "Q", {a: v["Q"] for a, v in loss.items()}, -1))
        shares.append(judge("Y2", "R", {a: v["R"] for a, v in loss.items()}, +1))
        print("\nR at 0.999: " + ", ".join(f"{a} {v['R999'].median():+.1%}"
                                         for a, v in loss.items()))
        print("seeds and median seconds per fit: " + ", ".join(
            f"{a} {v['n']} / {v['seconds']:.0f}s" for a, v in loss.items()))
    lev = leverage_measures()
    if lev:
        shares.append(judge("Y3", "G", {a: v[20.0] for a, v in lev.items()}, -1))
        print("G at a dose of 4: " + ", ".join(f"{a} {v[4.0].median():+.1%}"
                                              for a, v in lev.items()))

    shares = [s for s in shares if s is not None]
    if len(loss) == 3 and len(lev) == 3 and shares:
        if all(s >= 0.5 for s in shares):
            print("\nexplanation SUPPORTED: shares " + ", ".join(f"{s:.2f}" for s in shares))
        elif all(s <= 0.25 for s in shares):
            print("\nexplanation REJECTED: shares " + ", ".join(f"{s:.2f}" for s in shares))
        else:
            print("\nexplanation PARTIAL: shares " + ", ".join(f"{s:.2f}" for s in shares))


if __name__ == "__main__":
    main()
