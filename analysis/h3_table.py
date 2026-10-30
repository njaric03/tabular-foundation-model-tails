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
hide it. Writes thesis/figures/tabela_6_4.csv, and for \\input the body of table 6.4
(tabela_6_4.tex, one row per configuration and repair, summarised over the five tables)
and the full table of appendix C (tabela_c_h3.tex, one row per table), none tracked.

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


LABELS = {"TabPFN-3, 1": "TabPFN-3, 1 član", "TabPFN-3, 8": "TabPFN-3, 8 članova",
          "TabPFN-3.5, 1": "TabPFN-3.5, 1 član", "TabPFN-3.5, 8": "TabPFN-3.5, 8 članova",
          "TabICLv2, 1": "TabICLv2, 1 član", "GBM, 1": "GBM (kontrola)"}
ARMS = {"clip": "Odsecanje", "log": "Logaritam", "cclip": "Uslovno odsecanje"}
PAYS = ["always", "above", "never", "none"]


def pc(v):
    """A relative change as LaTeX, with a decimal comma: $+1,5$\\,\\%."""
    s = "0,0" if abs(v) < 5e-4 else f"{100 * v:+.1f}".replace(".", ",")
    return f"${s}$\\,\\%"


def latex(t, out):
    """Table 6.4 summarised over the five tables, and the full table for appendix C."""
    assert not (t.pays == "below").any(), "the inverse case appeared; add a column for it"
    cols = len(PAYS) + 5
    body = [r"\begin{tabular}{@{}l r r r r r r r r@{}}", r"\toprule",
            r"& \multicolumn{2}{c}{Cena na čistom} & \multicolumn{2}{c}{Korist pod greškom}"
            r" & \multicolumn{4}{c}{Tabele: isplati se} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-9}",
            r"Konfiguracija & medijana & najgori & nasumična & vidljiva"
            r" & uvek & iznad $p^\ast$ & nikad & bez efekta \\"]
    # Obrazac 3 puts every caption below, so it sits in the last foot.
    head = [r"\toprule", r"Konfiguracija & Tabela & medijana & najgori & nasumična"
            r" & vidljiva & Isplati se \\", r"\midrule"]
    full = [r"\setlength{\LTcapwidth}{\textwidth}",
            r"\begin{longtable}{@{}l l r r r r l@{}}", *head, r"\endfirsthead",
            *head, r"\endhead", r"\bottomrule", r"\endfoot", r"\bottomrule",
            r"\caption{Cena i korist korekcija po tabeli osiguranja i konfiguraciji, nivo"
            r" $0,99$. Kolone kao u tabeli~\ref{tab:h3}; poslednja kaže pri kom udelu $p$"
            r" konteksta sa nasumičnom greškom se korekcija isplati.}\label{tab:h3-puna}",
            r"\endlastfoot"]
    for arm, name in ARMS.items():
        part = t[t.arm == arm]
        body += [r"\midrule", f"\\multicolumn{{{cols}}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\"]
        full += [f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\"]
        for cfg, g in part.groupby("config", sort=False):
            n = g.pays.value_counts()
            body.append(" & ".join([LABELS[cfg], pc(g.clean_median.median()),
                                    pc(g.clean_worst.max()), pc(g.random_median.median()),
                                    pc(g.visible_median.median())]
                                   + [str(n.get(k, 0)) for k in PAYS]) + r" \\")
            for r in g.itertuples():
                when = {"always": "uvek", "never": "nikad", "none": "bez efekta",
                        "above": f"$p > {r.p_star:.2f}$".replace(".", ",")}[r.pays]
                full.append(" & ".join([LABELS[cfg], r.table, pc(r.clean_median),
                                        pc(r.clean_worst), pc(r.random_median),
                                        pc(r.visible_median), when]) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    full += [r"\end{longtable}"]
    (out / "tabela_6_4.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    (out / "tabela_c_h3.tex").write_text("\n".join(full) + "\n", encoding="utf-8")


def main():
    t = rows()
    out = ROOT / "thesis/figures"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "tabela_6_4.csv", index=False)
    latex(t, out)
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
