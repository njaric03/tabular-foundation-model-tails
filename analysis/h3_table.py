"""Table 6.4 of the thesis: what the cap and the log cost on a clean context and gain
under a unit error, per insurance table and configuration, with the break-even rate.

From the unit-error confirmations of 1, 21 and 22.10.2026 (same draws), with the
conditional cap of 29.10.2026 beside them where it ran, at one member. Per unit, from
mean pinball at 0.99 over 20 repeats: D0 = arm - raw on the clean context, D1 the same
under the random x100 row (the error a table would actually carry), and the visible row
beside it. With p the share of contexts carrying such an error, the arm pays when
(1 - p) D0 + p D1 < 0:

    D0 > 0 > D1   pays above  p* = D0 / (D0 - D1)
    D0 < 0 < D1   pays below  p* = -D0 / (D1 - D0)   (the inverse case of TODO P2)
    both <= 0     pays at every p;   both >= 0   never;   both exactly 0   no effect.

The worst repeat on the clean context is the largest relative rise, so a median does not
hide it. Writes thesis/figures/tabela_6_4.csv, which is not tracked.

    python -m analysis.h3_table
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import paths, tables

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [  # label, file, model
    ("TabPFN-3, 1", "unit_error_confirm_log.csv", "TabPFN-V3"),
    ("TabPFN-3, 8", "unit_error_confirm_nest8.csv", "TabPFN-V3"),
    ("TabPFN-3.5, 1", "unit_error_confirm_tabpfn35_nest1.csv", "TabPFN-v3.5"),
    ("TabPFN-3.5, 8", "unit_error_confirm_tabpfn35_nest8.csv", "TabPFN-v3.5"),
    ("TabICLv2, 1", "unit_error_confirm_log.csv", "TabICLv2"),
    ("GBM, 1", "unit_error_confirm_log.csv", "GBM"),
]
CCLIP = "unit_error_confirm_cclip.csv"


def regime(d0, d1):
    # A cap that fires on no context reuses raw's prediction, so both differences are 0.
    if d0 == 0 and d1 == 0:
        return "none", np.nan
    if d0 <= 0 and d1 <= 0:
        return "always", np.nan
    if d0 >= 0 and d1 >= 0:
        return "never", np.nan
    if d0 > 0:
        return "above", d0 / (d0 - d1)
    return "below", -d0 / (d1 - d0)


def wide(d):
    return d.pivot_table(index=["dataset", "repeat"], columns=["context", "arm"], values="pb99")


def units(label, arm, w):
    out = []
    for table, g in w.groupby(level="dataset"):
        m = g.mean()
        d0 = m[("clean", arm)] - m[("clean", "raw")]
        d1 = m[("random", arm)] - m[("random", "raw")]
        rel = lambda cx: g[(cx, arm)] / g[(cx, "raw")] - 1
        kind, p = regime(d0, d1)
        out.append(dict(config=label, arm=arm, table=table,
                        clean_median=rel("clean").median(),
                        clean_worst=rel("clean").max(),
                        random_median=rel("random").median(),
                        visible_median=rel("visible").median(),
                        d0=d0, d1=d1, pays=kind, p_star=p))
    return out


def rows():
    out = []
    cclip = tables.ok_rows(pd.read_csv(paths.result(f"h3_repair/{CCLIP}")))
    for label, name, model in CONFIGS:
        d = tables.ok_rows(pd.read_csv(paths.result(f"h3_repair/{name}")))
        d = d[d.model == model]
        w = wide(d)
        for arm in ("clip", "log"):
            out += units(label, arm, w)
        # Against the raw rows of its own file, which equal those of 20.10.2026 (R0).
        c = cclip[cclip.model == model]
        if (d.n_est == 1).all() and len(c):
            out += units(label, "cclip", wide(c))
    return pd.DataFrame(out)


def main():
    t = rows()
    out = ROOT / "thesis/figures"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "tabela_6_4.csv", index=False)
    pct = lambda v: f"{v:+.1%}"
    for arm, name in (("clip", "clip_200"), ("log", "log"), ("cclip", "cclip")):
        print(f"\n=== {name} against raw at 0.99: clean median / clean worst / random / visible,"
              " and when it pays")
        part = t[t.arm == arm]
        for (cfg, table), r in part.set_index(["config", "table"]).iterrows():
            when = {"always": "always", "never": "never", "none": "no effect",
                    "above": f"if p > {r.p_star:.2f}", "below": f"if p < {r.p_star:.2f}"}[r.pays]
            print(f"  {cfg:13s} {table:16s} {pct(r.clean_median):>7} {pct(r.clean_worst):>7}"
                  f" {pct(r.random_median):>7} {pct(r.visible_median):>7}   {when}")
        print("\n  units by regime:")
        print(part.pivot_table(index="config", columns="pays", values="table", aggfunc="count",
                               fill_value=0).to_string())


if __name__ == "__main__":
    main()
