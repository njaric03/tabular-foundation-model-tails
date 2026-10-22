"""The claim ledger: every number the thesis quotes, recomputed from results/ in one run.

One row per claim: the value, the file and the condition it holds under. Writing from this
ledger, not from memory or old notes, is what keeps a stale number out of the text; +83%
and +19% both survived for days in places nobody recomputed. Writes
thesis/notes/knjiga_tvrdnji.md, which is not tracked, and prints the same table.

    python -m analysis.thesis_numbers
"""
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.dose_ensemble import generator_law
from analysis.h3_table import rows as h3_rows
from common import paths, tables

ROOT = Path(__file__).resolve().parents[1]
LEDGER = []


def R(name):
    return tables.ok_rows(pd.read_csv(paths.result(name), low_memory=False))


def claim(part, text, value, source, caveat=""):
    LEDGER.append(dict(part=part, claim=text, value=value, source=source, caveat=caveat))


def pct(v, digits=1):
    return f"{100 * v:+.{digits}f}%"


def h1():
    sc = tables.scale_share_by_model(R("scale_of_x.csv")).share
    sh1 = tables.slope_by_model(R("shape_of_x_nest1.csv")).share
    sh8 = tables.slope_by_model(R("shape_of_x_20.csv")).share
    for m in ("TabPFN-V3", "TabICLv2", "GBM"):
        claim("H1", f"share of the true change tracked, {m}, one member: scale / shape",
              f"{sc[m]:.3f} / {sh1[m]:.3f}", "scale_of_x.csv, shape_of_x_nest1.csv",
              "GBM is the control" if m == "GBM" else "")
    claim("H1", "TabPFN-V3 shape share at four members", f"{sh8['TabPFN-V3']:.3f}",
          "shape_of_x_20.csv", "N_EST=4 in this file, not the package default of 8")
    s35 = tables.scale_share_by_model(R("scale_of_x_tabpfn35.csv")).share["TabPFN-v3.5"]
    h35 = tables.slope_by_model(R("shape_of_x_tabpfn35.csv")).share["TabPFN-v3.5"]
    claim("H1", "TabPFN-3.5, one member: scale / shape", f"{s35:.3f} / {h35:.3f}",
          "scale_of_x_tabpfn35.csv, shape_of_x_tabpfn35.csv", "GPU")
    o = R("oracle_xi_of_x.csv")
    g = o[o.n == 2000]
    t = (g.true_2 - g.true_0).mean()
    claim("H1", "EVT oracle on the same 2000 rows, share of the shape change",
          f"{(g.oracle_r_2 - g.oracle_r_0).mean() / t:.3f}", "oracle_xi_of_x.csv",
          "the oracle knows the shape feature; it does not rule out sample size")
    ks = pd.concat([R(f"dissociation_real_slope_kshare_{s}.csv") for s in ("tabicl", "tabpfn")])
    for m in ("TabPFN-V3", "TabICLv2"):
        v = ks[ks.model == m].groupby("k_share").slope_shape_corrected.median()
        claim("H1", f"real data, corrected shape slope across Hill k 0.05-0.20, {m}",
              f"{v.min():.2f} to {v.max():.2f}", "dissociation_real_slope_kshare_*.csv",
              "direction holds at every k; the number does not, so it is qualitative")


def h2():
    pv = R("prevalence_data.csv")
    col = next(c for c in ("max_shift", "sd_shift_max", "shift_max") if c in pv)
    claim("H2", "public datasets whose subsamples reach an sd shift of 4 / 10",
          f"{int((pv[col] >= 4).sum())} / {int((pv[col] >= 10).sum())} of {len(pv)}",
          "prevalence_data.csv", "sampled contexts, not a rate of data errors")
    files = {("V3", 1): "clip_context_dose_v3_nest1.csv", ("V3", 8): "clip_context_dose_v3_nest8.csv",
             ("3.5", 1): "clip_context_dose_tabpfn35_nest1.csv",
             ("3.5", 8): "clip_context_dose_tabpfn35_nest8.csv"}
    for (m, n), f in files.items():
        d = generator_law(R(f"h3_repair/{f}"))
        raw = d[d.variant == "raw"]
        r = {}
        for col in ("epb99", "epb999", "pb999"):
            w = raw.pivot_table(index=["xi", "seed"], columns="sd_shift_target", values=col)
            r[col] = (w[50.0] / w[1.0] - 1).median()
        claim("H2", f"one added row, sd shift 50, TabPFN-{m} at {n} member(s): rise at 0.99 / 0.999",
              f"{pct(r['epb99'])} / {pct(r['epb999'])}", f"h3_repair/{f}",
              "expected loss under the generator's law" + (
                  f"; drawn test {pct(r['pb999'], 0)} at 0.999" if (m, n) == ("V3", 1) else ""))
    nat = R("h3_repair/clip_context_real.csv")
    nat = pd.concat([nat, R("h3_repair/clip_context_real_tabpfn35.csv"),
                     R("h3_repair/clip_context_real_causilo.csv"),
                     R("h3_repair/clip_context_real_limix2.csv"),
                     R("h3_repair/clip_context_real_v25v26.csv")])
    raw = nat[(nat.arm == "raw") & (nat.bin == "4.0-inf")]
    ratio = (1 + raw.d_q99).groupby(raw.model).median()
    claim("H2", "freMTPL2sev, Q(0.99) with the largest claim over without it, sd shift >= 4",
          ", ".join(f"{m} {v:.2f}" for m, v in ratio.sort_values().items()),
          "clip_context_real*.csv", "5 subsamples sharing few distinct largest claims; one member")
    cv = R("h1_shape_vs_scale/coverage.csv")
    clean = cv[cv.dose == cv.dose.min()].groupby("model").cov99.median()
    claim("H2", "empirical coverage at 0.99 on a clean context",
          ", ".join(f"{m} {v:.4f}" for m, v in clean.items()), "coverage.csv", "")


def h3():
    nat = pd.concat([R("h3_repair/clip_context_real.csv"),
                     R("h3_repair/clip_context_real_tabpfn35.csv")])
    nat = nat[nat.bin.isin(["2.0-4.0", "4.0-inf"])]
    w = nat.pivot_table(index=["model", "bin", "repeat"], columns=["arm", "clip_c"], values="pb99")
    g = (w[("clip", 200.0)] / w[("raw", 0.0)] - 1) < 0
    counts = g.groupby(level="model").sum().astype(int)
    claim("H3", "freMTPL2sev, sd shift >= 2: subsamples where clip_200 lowers the 0.99 loss",
          ", ".join(f"{m} {v}/10" for m, v in counts.items()), "clip_context_real*.csv",
          "development result; the cap was chosen after it")
    u = R("h3_repair/unit_error_confirm.csv")
    w = u[u.context == "visible"].pivot_table(index=["model", "dataset", "repeat"], columns="arm",
                                              values="pb99")
    lower = ((w["clip"] / w["raw"] - 1).groupby(level=["model", "dataset"]).median() < 0)
    claim("H3", "unit error (visible row): tables where clip_200 lowers the 0.99 loss, one member",
          ", ".join(f"{m} {v}/5" for m, v in lower.groupby(level="model").sum().astype(int).items()),
          "unit_error_confirm.csv", "confirmation of 1.10.2026 on fresh draws")
    t = h3_rows()
    for arm in ("clip", "log"):
        a = t[t.arm == arm]
        claim("H3", f"{arm}: worst repeat on a clean context, over tables and configurations",
              pct(a.clean_worst.max()), "unit_error_confirm*.csv via analysis/h3_table.py",
              f"at {a.loc[a.clean_worst.idxmax(), 'config']}, "
              f"{a.loc[a.clean_worst.idxmax(), 'table']}")
    s = R("h3_repair/legit_segment.csv")
    c = s[(s.variant == "clip") & (s.seg_scale == 20) & (s.sd_shift_target == 1)]
    claim("H3", "segment x20, clean: clipped context rows / of them in the segment, median",
          f"{c.n_clipped.median():.0f} / {c.n_clipped_seg.median():.0f}", "legit_segment.csv", "")
    for f, m, label in (("legit_segment_log.csv", "TabPFN-V3", "TabPFN-V3, 1"),
                        ("legit_segment_nest8.csv", "TabPFN-V3", "TabPFN-V3, 8"),
                        ("legit_segment_tabpfn35_nest8.csv", "TabPFN-v3.5", "TabPFN-3.5, 8"),
                        ("legit_segment_log.csv", "TabICLv2", "TabICLv2, 1")):
        s = R(f"h3_repair/{f}")
        s = s[(s.model == m) & (s.seg_scale == 20) & (s.sd_shift_target == 1)]
        q = s.groupby("variant").qratio999_seg.median()
        claim("H3", f"segment x20, clean, predicted / true Q(0.999) in the segment, {label}: "
              "raw / clip / log", f"{q['raw']:.2f} / {q['clip']:.2f} / {q['log']:.2f}",
              f"h3_repair/{f}", "the log's overshoot of a legitimate segment")
    rc = R("h3_repair/recalibration.csv")
    rc = rc[(rc.model == "TabPFN-V3") & (rc.seg_scale == 1) & (rc.sd_shift_target == 20)]
    gaps = []
    for col in ("epb99_all_none", "epb99_all_mul"):
        w = rc.pivot_table(index="seed", columns="variant", values=col)
        gaps.append((w["clip"] / w["raw"] - 1).median())
    claim("H3", "recalibration K2: cap against raw at M = 1, shift 20, TabPFN-V3, without / with",
          f"{pct(gaps[0])} / {pct(gaps[1])}", "recalibration.csv",
          "the calibration rows are new clean information")


def main():
    h1()
    h2()
    h3()
    led = pd.DataFrame(LEDGER)
    out = ROOT / "thesis/notes"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Knjiga tvrdnji", "",
             "Generisano skriptom `analysis/thesis_numbers.py`; ne menjati rukom. Svaki broj u "
             "tekstu teze treba da postoji ovde.", ""]
    for part, g in led.groupby("part", sort=False):
        lines += [f"## {part}", "", "| Tvrdnja | Vrednost | Izvor | Uslov |", "|---|---|---|---|"]
        lines += [f"| {r.claim} | {r.value} | `{r.source}` | {r.caveat} |" for r in g.itertuples()]
        lines.append("")
    (out / "knjiga_tvrdnji.md").write_text("\n".join(lines), encoding="utf-8")
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(led[["part", "claim", "value"]].to_string(index=False))
    print(f"\n{len(led)} claims -> {out / 'knjiga_tvrdnji.md'}")


if __name__ == "__main__":
    main()
