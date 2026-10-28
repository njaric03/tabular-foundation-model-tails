# -*- coding: utf-8 -*-
"""Verdicts of rules T1 to T5 of 28.10.2026: TabICLv2 at its default ensemble of eight,
the leverage on the generator and on freMTPL2sev, and the repairs' loss at 2000 and
10 000 rows on the expected pinball under the generator's law.

Reads the outputs of `run/tabicl8_2810.sh`. Written with the rules, before any of the
runs; a file that is not there yet is skipped.

    python -m analysis.tabicl8
"""
import pandas as pd

from analysis.confirm_ensemble_log import verdict
from analysis.dose_ensemble import arm, generator_law, load, rise, wide
from common import paths, tables

# One-member medians of the global change of Q(0.99) at XI=0.7 and a dose of 20.
LEVERAGE = {2000: ("influence_sd_tabicl8.csv", -0.151),
            10000: ("influence_sd_tabicl8_n10k.csv", -0.387)}
LOSS = {(t, n): f"clip_context_tabicl_n{t // 1000}k_nest{n}.csv"
        for t in (2000, 10000) for n in (1, 8)}


def h2(name):
    path = paths.result(f"h2_leverage/{name}")
    if not path.exists():
        print(f"{name}: not there yet")
        return None
    return tables.ok_rows(pd.read_csv(path))


def leverage():
    for t, (f, one) in LEVERAGE.items():
        d = h2(f)
        if d is None:
            continue
        g = d[d.dose == 20.0].global_q99
        verdict("T1", g.median() <= one / 2,
                f"{t} rows: {g.median():+.1%} against {one:+.1%} at one member, "
                f"{len(g)} seeds, from {g.min():+.1%} to {g.max():+.1%}")
        print(f"     dose 4: {d[d.dose == 4.0].global_q99.median():+.1%}")

    d = load("clip_context_real_tabicl8.csv")
    if d is None:
        return
    d = d[(d.arm != "clip") | (d.clip_c == 200)].copy()
    d.loc[d.arm == "clip", "arm"] = "clip_200"
    m = d[d.arm == "raw"].groupby("bin").d_q99.median()
    if "4.0-inf" not in m.index:
        print("clip_context_real_tabicl8.csv: the bin of 4 sd and above is not there yet")
        return
    verdict("T2", m["4.0-inf"] <= -0.05,
            "raw d_q99 by bin " + ", ".join(f"{b} {v:+.1%}" for b, v in m.items()))
    print("     d_q99 by bin and arm, median")
    print(d.pivot_table(index="bin", columns="arm", values="d_q99", aggfunc="median")
          .round(3).to_string())
    w = d[d.bin.isin(["2.0-4.0", "4.0-inf"])].pivot_table(index=["bin", "repeat"],
                                                           columns="arm", values="pb99")
    for a in ("clip_200", "log"):
        r = w[a] / w["raw"] - 1
        print(f"     {a} over raw, pinball at 0.99, 2 sd and above: {(r < 0).sum()} of "
              f"{len(r)} below, median {r.median():+.1%}, sum "
              f"{w[a].sum() / w['raw'].sum() - 1:+.1%}, worst cell {r.max():+.1%}")


def loss():
    w99, w999 = {}, {}
    for k, f in LOSS.items():
        d = load(f)
        if d is None:
            continue
        g = generator_law(d)
        a, b = wide(g, "epb99"), wide(g, "epb999")
        full = [(s, v) for s in (1.0, 20.0) for v in ("raw", "clip", "log")]
        if any(c not in a.columns for c in full):
            print(f"{f}: no seed has every arm yet")
            continue
        w99[k], w999[k] = a.dropna(subset=full), b.dropna(subset=full)

    for (t, n), w in w99.items():
        print(f"\n{t} rows, {n} member(s), {len(w)} complete seeds")
        r = rise(w, 20.0)
        if n == 8:
            one = rise(w99[(t, 1)], 20.0).median() if (t, 1) in w99 else float("nan")
            verdict("T3", r.median() >= 0.02 and r.median() >= one / 2,
                    f"raw at shift 20 over clean {r.median():+.1%} against "
                    f"{one:+.1%} at one member")
        else:
            print(f"     raw at shift 20 over clean {r.median():+.1%}")
        for rule, v in (("T4", "clip"), ("T5", "log")):
            r, c = arm(w, 20.0, v, "raw"), arm(w, 1.0, v, "raw")
            s = w[(20.0, v)].sum() / w[(20.0, "raw")].sum() - 1
            verdict(rule, (r < 0).sum() >= 8 and c.median() <= 0.02,
                    f"{v} below raw at shift 20 in {(r < 0).sum()} of {len(r)}, median "
                    f"{r.median():+.1%}, sum {s:+.1%}, worst seed {r.max():+.1%}; "
                    f"clean context {c.median():+.2%}")
            z = arm(w999[(t, n)], 20.0, v, "raw")
            print(f"     {v} at 0.999, shift 20: {(z < 0).sum()} of {len(z)} below, "
                  f"median {z.median():+.1%}")
        print(f"     raw at 0.999, shift 20 over clean "
              f"{rise(w999[(t, n)], 20.0).median():+.1%}")


def main():
    leverage()
    loss()


if __name__ == "__main__":
    main()
