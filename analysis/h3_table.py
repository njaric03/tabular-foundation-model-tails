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
hide it. Writes thesis/figures/tabela_6_4.csv, and for \\input the bodies of the cost and
benefit table of chapter 6 (tabela_6_4.tex, one row per configuration and repair,
summarised over the five tables), of the break-even table beside it (tabela_prag.tex)
and of the three tables of appendix C, one per repair and one row per table
(tabela_c_h3_<arm>.tex), none tracked.

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
        # The change in mean loss carries the sign of D0 and D1, so the decision reads off it.
        out.append(dict(config=label, arm=arm, table=table,
                        clean_median=rel("clean").median(),
                        clean_mean=d0 / m[("clean", "raw")],
                        clean_worst=rel("clean").max(),
                        random_median=rel("random").median(),
                        random_mean=d1 / m[("random", "raw")],
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
    """A relative change in percent as LaTeX, with a decimal comma and no unit, which the
    header carries: $+1,5$. An exact zero (the repair left the context as it was) is $0$,
    and a change that rounds to zero keeps its sign, $-0,0$."""
    return "$0$" if v == 0 else "$" + f"{100 * v:+.1f}".replace(".", ",") + "$"


def p(v):
    return f"{v:.2f}".replace(".", ",")


# Header of the cost and benefit tables: what every number is and which way is better.
LOSS = r"Promena gubitka u \%, negativna vrednost je manji gubitak"
WHEN = {"always": r"za svako $p$", "never": r"ni za jedno $p$", "none": "bez razlike"}


def group(name, cols, first):
    """A configuration or repair heading inside a table, with air above all but the first."""
    return ([] if first else [r"\addlinespace[4pt]"]) + [
        f"\\multicolumn{{{cols}}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\"]


def latex(t, out):
    """The cost and benefit table of chapter 6 summarised over the five tables, the
    break-even table beside it and one table per repair for appendix C."""
    assert not (t.pays == "below").any(), "the inverse case appeared; add a column for it"
    body = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
            f"& \\multicolumn{{4}}{{c}}{{{LOSS}}} \\\\", r"\cmidrule(l){2-5}",
            r"& \multicolumn{2}{c}{Čist kontekst} & \multicolumn{2}{c}{Greška jedinice $\times 100$}"
            r" \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
            r"Konfiguracija & medijana & najveći rast & nasumičan red & vidljiv red \\"]
    # Fixed widths, so the long header sits over the four counts and not over the last.
    pays = [r"\begin{tabular}{@{}l *{4}{>{\centering\arraybackslash}p{2.4cm}}@{}}",
            r"\toprule",
            r"& \multicolumn{4}{c}{Broj tabela (od pet) na kojima korekcija smanjuje"
            r" očekivani gubitak} \\", r"\cmidrule(l){2-5}",
            r"Konfiguracija & za svako $p$ & za $p > p^\ast$ ($p^\ast$) & ni za jedno $p$"
            r" & bez razlike \\"]
    for arm, name in ARMS.items():
        part = t[t.arm == arm]
        body += [r"\midrule", *group(name, 5, True)]
        pays += [r"\midrule", *group(name, 5, True)]
        full = [r"\begin{tabular}{@{}l r r r r r r l@{}}", r"\toprule",
                f"& \\multicolumn{{6}}{{c}}{{{LOSS}}} & \\\\", r"\cmidrule(lr){2-7}",
                r"& \multicolumn{3}{c}{Čist kontekst} & \multicolumn{2}{c}{Nasumičan red}"
                r" & \multicolumn{1}{c}{Vidljiv red} & Korekcija \\",
                r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
                r"Tabela & medijana & prosek & najveći rast & medijana & prosek & medijana"
                r" & se isplati \\", r"\midrule"]
        for i, (cfg, g) in enumerate(part.groupby("config", sort=False)):
            n = g.pays.value_counts()
            body.append(" & ".join([LABELS[cfg], pc(g.clean_median.median()),
                                    pc(g.clean_worst.max()), pc(g.random_median.median()),
                                    pc(g.visible_median.median())]) + r" \\")
            above = str(n.get("above", 0))
            if n.get("above", 0):
                above += " (" + "; ".join(p(v) for v in sorted(g.p_star.dropna())) + ")"
            pays.append(" & ".join([LABELS[cfg], str(n.get("always", 0)), above,
                                    str(n.get("never", 0)), str(n.get("none", 0))]) + r" \\")
            full += group(LABELS[cfg], 8, i == 0)
            for r in g.itertuples():
                when = WHEN.get(r.pays) or f"za $p > {p(r.p_star)}$"
                full.append(" & ".join([r"\hspace{1em}" + r.table, pc(r.clean_median),
                                        pc(r.clean_mean), pc(r.clean_worst),
                                        pc(r.random_median), pc(r.random_mean),
                                        pc(r.visible_median), when]) + r" \\")
        full += [r"\bottomrule", r"\end{tabular}"]
        (out / f"tabela_c_h3_{arm}.tex").write_text("\n".join(full) + "\n", encoding="utf-8")
    body += [r"\bottomrule", r"\end{tabular}"]
    pays += [r"\bottomrule", r"\end{tabular}"]
    (out / "tabela_6_4.tex").write_text("\n".join(body) + "\n", encoding="utf-8")
    (out / "tabela_prag.tex").write_text("\n".join(pays) + "\n", encoding="utf-8")


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
