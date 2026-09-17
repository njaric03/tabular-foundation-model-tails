# -*- coding: utf-8 -*-
"""Every number a poster abstract may quote, recomputed from the committed CSVs.

No model is called. Each block prints the aggregate, the unit it is counted over and the
filter, so a sentence of the abstract can be traced to one row of the table this prints.

    venv-tfmp/Scripts/python.exe analysis/poster_claims.py > claims.md
"""
import numpy as np
import pandas as pd

from common import paths, tables

BINS_LEV = ["2.0-4.0", "4.0-inf"]
BINS_LOW = ["0.0-1.2", "1.2-2.0"]
TFM = ["TabPFN-v2.5", "TabPFN-v2.6", "TabPFN-V3", "TabICLv2", "EXAONE", "TabDPT"]
TREES = ["GBM", "XGB", "CB"]
KEY_REAL = ["dataset", "model", "bin", "repeat", "arm", "clip_c", "n_est", "n_fit", "n_test"]


def pct(x):
    return f"{100 * x:+.1f}%"


def arm_name(d):
    return np.where(d.arm == "clip", "clip_" + d.clip_c.map("{:g}".format), d.arm)


def load_natural():
    frames = [tables.ok_rows(pd.read_csv(paths.result(f)))
              for f in ("clip_context_real.csv", "clip_context_real_v25v26.csv")]
    d = pd.concat(frames)
    # The v2.5 and v2.6 rows of the main file are the separate file's, digit for digit.
    d = d.drop_duplicates(KEY_REAL, keep="last")
    d["arm_c"] = arm_name(d)
    return d


def prevalence():
    p = tables.ok_rows(pd.read_csv(paths.result("prevalence_data.csv")))
    top = p.loc[p.sd_shift_max.idxmax()]
    listed = len(pd.read_csv(paths.result("prevalence_data.csv")))
    print("## C1. Prevalence of leverage (no model)\n")
    print(f"- datasets loaded: {len(p)} (of {listed} listed); subsamples of "
          f"{int(p.n_fit.iloc[0])} rows, {int(p.n_subsamples.iloc[0])} per dataset")
    print(f"- largest sd shift over subsamples >= 4: {(p.sd_shift_max >= 4).sum()} datasets "
          f"({', '.join(sorted(p[p.sd_shift_max >= 4].dataset))}); >= 10: "
          f"{(p.sd_shift_max >= 10).sum()}; maximum {top.sd_shift_max:.1f} ({top.dataset})")
    fp = paths.load_json("dataset_fingerprints.json")
    sha = p.dataset.map(lambda n: fp.get(n, {}).get("y_sha1"))
    shared = p[sha.duplicated(keep=False)].groupby(sha).dataset.apply(sorted).tolist()
    print(f"- {sha.nunique()} distinct target vectors among the {len(p)} names (shared: "
          f"{'; '.join(' = '.join(g) for g in shared)}); none of them reaches 4")
    print("- unit: dataset name; source `results/h2_leverage/prevalence_data.csv`\n")


def natural(d):
    raw = d[d.arm_c == "raw"]
    print("## C2. Natural maximum, freMTPL2sev, raw against without_max\n")
    shifts = raw[raw.bin == "4.0-inf"].drop_duplicates("repeat").sd_shift
    print(f"- context 2000 rows, 1000 held-out; bin 4.0-inf shifts "
          f"{shifts.min():.1f} to {shifts.max():.1f}; 5 subsamples per bin; one dataset\n")
    print("| model | median d_q99, shift 2-4 | range | median d_q99, shift >=4 | range | "
          "falls in (>=4) |")
    print("|---|---|---|---|---|---|")
    for m in TFM + TREES:
        row = [m]
        for b in BINS_LEV:
            x = raw[(raw.model == m) & (raw.bin == b)].d_q99
            row += [pct(x.median()), f"{pct(x.min())} to {pct(x.max())}"]
        x = raw[(raw.model == m) & (raw.bin == "4.0-inf")].d_q99
        row.append(f"{(x < 0).sum()} of {len(x)}")
        print("| " + " | ".join(row) + " |")
    nat = tables.ok_rows(pd.read_csv(paths.result("clip_context_real_native.csv")))
    nat = nat[(nat.arm == "raw") & (nat.bin == "4.0-inf")]
    print("\nThe same subsamples with categorical columns passed natively (CATEGORICAL=native), "
          "median d_q99 at >=4: " + ", ".join(
              f"{m} {pct(g.d_q99.median())}" for m, g in nat.groupby("model")) + "\n")


def natural_repair(d):
    print("## C3. Repairs on the natural maximum: pinball at 0.99 against raw\n")
    print("- paired per subsample; leverage = bins 2-4 and >=4 (10 subsamples), "
          "clean = bins below 2 (10); median relative change, count lower\n")
    print("| model | arm | under leverage: median | lower in | clean: median | lower in |")
    print("|---|---|---|---|---|---|")
    idx = ["model", "bin", "repeat"]
    raw = d[d.arm_c == "raw"].set_index(idx).pb99
    for m in TFM + ["GBM"]:
        for a in ("clip_200", "sd_cap", "rank_gpd"):
            x = d[(d.model == m) & (d.arm_c == a)].set_index(idx).pb99
            if x.empty:
                continue
            r = (x / raw.reindex(x.index) - 1).dropna()
            lev = r[r.index.get_level_values("bin").isin(BINS_LEV)]
            low = r[r.index.get_level_values("bin").isin(BINS_LOW)]
            print(f"| {m} | {a} | {pct(lev.median())} | {(lev < 0).sum()} of {len(lev)} | "
                  f"{pct(low.median())} | {(low < 0).sum()} of {len(low)} |")
    print()


def unit_error():
    u = tables.ok_rows(pd.read_csv(paths.result("unit_error_real.csv")))
    u["arm_c"] = arm_name(u)
    print("## C4. A unit error: one context target times 100, seven tables\n")
    e = u[(u.arm_c == "raw") & (u.factor == 100)]
    shift = e.groupby("dataset").sd_shift.median()
    print("- context 2000 rows, 1000 held-out, 10 repeats per table; the corrupted row is drawn "
          "among rows whose error reaches an sd shift of 2 (a conditional stress test)")
    print(f"- median sd shift under the error, by table: {shift.min():.1f} to {shift.max():.1f}; "
          f"share of rows whose error would be visible: "
          f"{u.groupby('dataset').share_visible.median().min():.0%} to "
          f"{u.groupby('dataset').share_visible.median().max():.0%}\n")
    idx = ["dataset", "model", "repeat"]
    clean = u[(u.arm_c == "raw") & (u.factor == 1)].set_index(idx).pb99
    err = e.set_index(idx).pb99
    cost = (err / clean.reindex(err.index) - 1).groupby(["dataset", "model"]).median()
    print("Cost of the error, raw context: median over repeats of pb99(error)/pb99(clean) - 1, "
          "range over the seven tables\n")
    print("| model | median over tables | range |")
    print("|---|---|---|")
    for m in ["TabPFN-v2.5", "TabPFN-v2.6", "TabPFN-V3", "TabICLv2", "GBM"]:
        c = cost.xs(m, level="model")
        print(f"| {m} | {pct(c.median())} | {pct(c.min())} to {pct(c.max())} |")
    print("\nRepairs against raw at 0.99 (units: table x model; each unit is the median over "
          "10 repeats of the paired relative change)\n")
    print("| models | arm | under error: units lower | median | worst unit | clean: median | "
          "worst unit | 0.999 under error: cells >100% worse |")
    print("|---|---|---|---|---|---|---|---|")
    groups = {"V3, TabICLv2, GBM": ["TabPFN-V3", "TabICLv2", "GBM"],
              "v2.5, v2.6": ["TabPFN-v2.5", "TabPFN-v2.6"]}
    for gname, ms in groups.items():
        for a in ("clip_200", "sd_cap", "rank_exp", "rank_gpd", "rank_emp"):
            row = [gname, a]
            for f in (100, 1):
                base = u[(u.arm_c == "raw") & (u.factor == f) & u.model.isin(ms)].set_index(idx)
                x = u[(u.arm_c == a) & (u.factor == f) & u.model.isin(ms)].set_index(idx)
                if x.empty:
                    row = None
                    break
                r = (x.pb99 / base.pb99.reindex(x.index) - 1)
                unit = r.groupby(["dataset", "model"]).median()
                if f == 100:
                    r999 = x.pb999 / base.pb999.reindex(x.index) - 1
                    row += [f"{(unit < 0).sum()} of {len(unit)}", pct(unit.median()),
                            pct(unit.max())]
                    blow = f"{(r999 > 1).mean():.1%}"
                else:
                    row += [pct(unit.median()), pct(unit.max())]
            if row:
                print("| " + " | ".join(row + [blow]) + " |")
    print()


def mechanism():
    c = tables.ok_rows(pd.read_csv(paths.result("clip_context.csv")))
    c["arm_c"] = arm_name(c.rename(columns={"variant": "arm"}))
    o = pd.read_csv(paths.result("pinball_oracle.csv"))
    key = ["xi", "seed", "n_train", "n_test"]
    c = c.merge(o[o.family == "gpd"][key + ["pb99", "pb999"]], on=key, suffixes=("", "_or"))
    c["R99"], c["R999"] = c.pb99 / c.pb99_or, c.pb999 / c.pb999_or
    print("## C5. TabPFN-V3 on the generator: mechanism against cost\n")
    print(f"- GPD, xi 0.7 and 0.9, {c.seed.nunique()} seeds each, 2000 context rows, 900 test; one "
          f"row added at the centre of x at the named sd shift; medians over the "
          f"{c.seed.nunique() * 2} cells\n")
    print("| arm | shift | implied xi (xi 0.7 / 0.9) | borders over the data | "
          "pinball/oracle 0.99 | pinball/oracle 0.999 |")
    print("|---|---|---|---|---|---|")
    for a in ("raw", "sd_cap", "clip_200"):
        for s in (1.0, 4.0, 20.0, 50.0):
            x = c[(c.arm_c == a) & (c.sd_shift_target == s)]
            xi = x.groupby("xi").xi_implied.median()
            print(f"| {a} | {s:g} | {xi.get(0.7, np.nan):+.2f} / {xi.get(0.9, np.nan):+.2f} | "
                  f"{x.borders_in_data.median():.0f} | {x.R99.median():.2f} | "
                  f"{x.R999.median():.2f} |")
    print()


def finetune():
    t = tables.ok_rows(pd.read_csv(paths.result("clip_context_real_tabicl_ft.csv")))
    t["arm_c"] = arm_name(t)
    t["family"] = t.model.str.replace(r"-s\d+$", "", regex=True)
    print("## C6. TabICLv2 continued pre-training, freMTPL2sev natural maximum\n")
    print("- per subsample the two training seeds are averaged first, then the median over "
          "subsamples; one dataset\n")
    avg = t.groupby(["family", "arm_c", "bin", "repeat"])[["d_q99", "pb99"]].mean()
    print("| model | raw median d_q99, >=4 (5) | clip_200 median d_q99, >=4 | "
          "raw pb99 median, below 2 (10) |")
    print("|---|---|---|---|")
    for f in ("TabICLv2", "TabICLv2-FT-A0", "TabICLv2-FT-A1"):
        r = avg.loc[(f, "raw")]
        cl = avg.loc[(f, "clip_200")]
        print(f"| {f} | {pct(r.loc['4.0-inf'].d_q99.median())} | "
              f"{pct(cl.loc['4.0-inf'].d_q99.median())} | "
              f"{r.loc[BINS_LOW].pb99.median():.0f} |")
    a1 = avg.loc[("TabICLv2-FT-A1", "raw")].loc[BINS_LEV].pb99
    a0 = avg.loc[("TabICLv2-FT-A0", "raw")].loc[BINS_LEV].pb99
    ratio = (a1 / a0.reindex(a1.index)).dropna()
    print(f"\n- FT-A1 against FT-A0, pb99 at shift >= 2: lower in {(ratio < 1).sum()} of "
          f"{len(ratio)}, median ratio {ratio.median():.3f}")
    by_seed = t[(t.arm_c == "raw") & (t.bin == "4.0-inf")].groupby("model").d_q99.median()
    print("- by seed, raw median d_q99 at >=4: " + ", ".join(
        f"{m} {pct(v)}" for m, v in by_seed.items()) + "\n")


def nano():
    folder = paths.result("prior_pretraining_eval.csv").parent
    frames = []
    for f in sorted(folder.glob("prior_pretraining_eval*.csv")):
        x = tables.ok_rows(pd.read_csv(f))
        if "family" not in x:
            x["family"] = "gpd"
        frames.append(x)
    d = pd.concat(frames).drop_duplicates(
        ["model", "family", "xi", "sd_shift_target", "seed", "n_train", "n_test"], keep="last")
    d = d[d.pretrain_seed.isin([92, 93]) & d.arm.isin(["A0B0", "A1B0"]) & (d.n_train == 200)]
    o = pd.read_csv(paths.result("pinball_oracle.csv"))
    key = ["family", "xi", "seed", "n_train", "n_test"]
    d = d.merge(o[key + ["pb99"]], on=key, suffixes=("", "_or"))
    d["R99"] = d.pb99 / d.pb99_or
    d["set"] = np.where(d.family == "gpd", "GPD", "three other families")
    print("## C7. nanoTabPFN, standard prior A0B0 against heavy-tailed and contaminated A1B0\n")
    print("- same architecture, same mean/sd target encoding, same first two training stages; "
          "200 context rows; 10 seeds x 2 xi per test set; seeds 92 / 93\n")
    print("| test set | arm | pinball/oracle 0.99: clean | shift 20 | shift 50 | "
          "borders clean -> 50 |")
    print("|---|---|---|---|---|---|")
    for s in ("GPD", "three other families"):
        for a in ("A0B0", "A1B0"):
            x = d[(d.set == s) & (d.arm == a)]
            cells = []
            for sh in (1.0, 20.0, 50.0):
                v = x[x.sd_shift_target == sh].groupby("pretrain_seed").R99.median()
                cells.append(" / ".join(f"{v.get(k, np.nan):.2f}" for k in (92, 93)))
            b = x.groupby("sd_shift_target").borders_in_data.median()
            print(f"| {s} | {a} | {cells[0]} | {cells[1]} | {cells[2]} | "
                  f"{b.get(1.0, np.nan):.0f} -> {b.get(50.0, np.nan):.0f} |")
    print()


def main():
    print("# Claims for the poster abstract, recomputed from the CSVs\n")
    d = load_natural()
    prevalence()
    natural(d)
    natural_repair(d)
    unit_error()
    mechanism()
    finetune()
    nano()


if __name__ == "__main__":
    main()
