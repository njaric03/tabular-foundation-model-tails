# -*- coding: utf-8 -*-
"""A unit error in one context row, on real heavy-tailed tables.

The external tables of `external_selection.py` have Hill indices from 0.48 to 1.05 and
none reaches an sd shift of 2 in 200 subsamples: a heavy tail alone does not produce the
leverage of part two, an error does. So the test brings one: a context row whose target is
multiplied by 100, a value in cents instead of units. The corrupted row is known, and the
clean context of the same subsample is the reference.

Datasets: freMTPL2sev and `data/external_selected.txt`. Per table, REPEATS subsamples of
N_FIT + N_TEST positive rows and the row to corrupt are drawn up front from seed 31337,
so a resumed run corrupts the same row. The row is drawn among those whose error reaches
an sd shift of 2; `share_visible` is the share of context rows where it would.

    factor 1     the clean context, what each arm costs when nothing is wrong
    factor 100   the chosen row times 100

The arms are those of `common/transforms.py`; `d_q*` is the change against raw on the
clean context, and pinball is on the clean held-out rows.

Outcomes, 12.9.2026, seven tables, three models, ten subsamples, paired over 21
dataset-model units:

  A unit error is visible at 4% of rows on freMTPL2sev and 20 to 33% on the small-claim
  tables, and lifts the shift to 2.8 to 5.5. It costs GBM most (pinball at 0.99 +35% on
  BlogFeedback), against 3 to 14% for the foundation models.

  | arm | pinball 0.99 | worst 0.99 | cells >100% worse at 0.999 | worst 0.999 |
  |---|---|---|---|---|
  | rank_exp | 19/21 (-4.0%) | +0.3% | 5.2% | +84% |
  | rank_gpd_trim | 19/21 (-3.6%) | +3.0% | 11.9% | +10191% |
  | rank_gpd | 17/21 (-1.9%) | +12.3% | 16.2% | +86195% |
  | sd_cap | 17/21 (-2.3%) | +4.9% | 0.5% | +1% |
  | clip_200 | 16/21 (-1.3%) | +4.0% | 0.0% | +1% |

  evt_trim touches the corrupted row in 10% of cells. log repairs as often as clip_200 but
  overflows in 22 of 210 cells. On a clean real context the rank arms buy nothing at 0.99.

Outcomes, 14.9.2026, TabPFN v2.5 and v2.6 on the same seven tables, 14 units: the error costs
them little accuracy, the caps carry over (11 of 14, p = 0.057, median -4.0%), and the rank
transform does not (8 of 14). Under the stronger natural leverage of freMTPL2sev it does help
them at 0.99 (`clip_context_real_v25v26.csv`).

The predictions written before each run and their full outcomes are in PREREGISTRATION.md.

    DATASETS=beMTPL97 MODELS=GBM REPEATS=2 python -u experiments/h3_repair/unit_error_real.py
"""
import time

import numpy as np
import pandas as pd

from common import (
    append,
    datasets,
    env,
    leverage,
    metrics,
    models,
    paths,
    quiet,
    tables,
    transforms,
)

quiet.silence()

DATASETS = env.names("DATASETS", ["freMTPL2sev"] + datasets.external_names())
MODELS = env.models("TabPFN-V3,TabICLv2,GBM")
REPEATS = env.integer("REPEATS", 10)
N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
N_EST = env.integer("N_EST", 1)
FACTORS = [1.0, 100.0]
ARMS = [("raw", 0.0), ("clip", 50.0), ("clip", 200.0), ("log", 0.0),
        ("tail_log", transforms.TAIL_LOG_C), ("evt_trim", 0.0), ("sd_cap", transforms.TAU),
        ("rank_emp", 0.0), ("rank_gpd", 0.0), ("rank_gpd_trim", 0.0), ("rank_exp", 0.0)]
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ("50", "90", "99", "999")
OUTPUT = env.text("OUTPUT", "unit_error_real.csv")

COLUMNS = (["dataset", "model", "repeat", "factor", "arm", "clip_c", "n_est", "n_fit",
            "n_test", "sd_shift", "share_visible", "corrupt_rank", "corrupt_over_next",
            "n_touched", "hit"]
           + [f"d_q{t}" for t in TAGS] + [f"pb{t}" for t in TAGS]
           + ["n_above99", "seconds", "reason"])
KEY = ["dataset", "model", "repeat", "factor", "arm", "clip_c", "n_est", "n_fit", "n_test"]


def draws(y):
    """Subsample, corrupted row and share of visible rows for every repeat, drawn up front."""
    rng = np.random.default_rng(31337)
    out = []
    for _ in range(REPEATS):
        idx = rng.permutation(len(y))[:N_FIT + N_TEST]
        shift = leverage.shift_if_corrupted(y[idx[:N_FIT]], FACTORS[-1])
        visible = np.flatnonzero(shift >= leverage.VISIBLE_SHIFT)
        out.append((idx, int(rng.choice(visible)) if len(visible) else -1,
                    float(len(visible) / N_FIT)))
    return out


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    for name in DATASETS:
        X, y = datasets.load(name)
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        for r, (idx, bad, share) in enumerate(draws(y)):
            if bad < 0:
                print(f"  {name}: #{r} no row at which a unit error is visible", flush=True)
                for model in MODELS:
                    append.write(OUTPUT, dict(dataset=name, model=model, repeat=r, factor=100.0,
                                              arm="raw", clip_c=0.0, n_est=N_EST, n_fit=N_FIT,
                                              n_test=N_TEST, share_visible=0.0,
                                              reason="no visible row"), COLUMNS)
                continue
            Xf, y_clean, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
            contexts = {}
            for f in FACTORS:
                yc = y_clean.copy()
                yc[bad] *= f
                ordered = np.sort(yc)[::-1]
                rank = int((yc > yc[bad]).sum()) + 1
                following = ordered[1] if rank == 1 else ordered[0]
                contexts[f] = (yc, dict(sd_shift=float(metrics.sd_shift(yc)), share_visible=share,
                                        corrupt_rank=rank,
                                        corrupt_over_next=float(yc[bad] / following)))
            for model in MODELS:
                keys = [dict(dataset=name, model=model, repeat=r, factor=f, arm=a, clip_c=c,
                             n_est=N_EST, n_fit=N_FIT, n_test=N_TEST)
                        for f in FACTORS for a, c in ARMS]
                todo = [k for k in keys if append.key(k, KEY) not in done]
                if not todo:
                    continue
                seed = 7000 + r

                def predict(yc):
                    return models.quantiles(model, Xf, yc, Xte, seed=seed, levels=LEVELS,
                                            n_est=N_EST)

                t1 = time.time()
                try:
                    q_ref = predict(y_clean)
                    err = None
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:110]
                for k in todo:
                    t2 = time.time()
                    yc, info = contexts[k["factor"]]
                    row = dict(k, **info)
                    if err:
                        row["reason"] = err
                    else:
                        try:
                            y_fit, back, n = transforms.transform(k["arm"], k["clip_c"], yc)
                            if k["arm"] == "raw" and k["factor"] == 1.0:
                                qa = q_ref
                            else:
                                with np.errstate(over="ignore", invalid="ignore"):
                                    qa = back(predict(y_fit))
                            row.update(n_touched=n, reason="",
                                       hit=bool(y_fit[bad] < yc[bad]) if k["factor"] > 1 else "")
                            row.update(zip([f"d_q{t}" for t in TAGS],
                                           metrics.rel_change_by_level(q_ref, qa)))
                            for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
                                row[f"pb{t}"] = metrics.pinball(yte, qa[:, i], a)
                            row["n_above99"] = int((yte > qa[:, 2]).sum())
                        except Exception as e:
                            row["reason"] = f"{type(e).__name__}: {e}"[:110]
                    row["seconds"] = round(time.time() - t2, 1)
                    append.write(OUTPUT, row, COLUMNS)
                corrupted = contexts[100.0][1]
                print(f"  {name[:16]:16s} {model:10s} #{r} shift x100 {corrupted['sd_shift']:6.2f} "
                      f"rank {corrupted['corrupt_rank']} over next "
                      f"{corrupted['corrupt_over_next']:7.1f}  [{time.time() - t1:.0f}s, "
                      f"{len(todo)} arms]", flush=True)
    report()
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    d["arm_c"] = np.where(d.arm.isin(["clip", "tail_log"]),
                          d.arm + "_" + d.clip_c.map("{:g}".format), d.arm)
    ref = d[(d.arm == "raw") & (d.factor == 1.0)].set_index(["dataset", "model", "repeat"])
    d = d.join(ref[["pb99", "pb999"]], on=["dataset", "model", "repeat"], rsuffix="_clean")
    d["rel99"] = d.pb99 / d.pb99_clean - 1
    d["rel999"] = d.pb999 / d.pb999_clean - 1
    for col in ("rel99", "rel999", "d_q99"):
        print(f"\n=== {col}, median over repeats ===")
        print(d.pivot_table(index=["factor", "dataset", "model"], columns="arm_c", values=col,
                            aggfunc="median").round(3).to_string())
    trim = d[d.arm == "evt_trim"]
    if len(trim):
        hit = trim[trim.factor == 100.0].hit.astype(str).eq("True").mean()
        touched = (trim[trim.factor == 1.0].n_touched > 0).mean()
        print(f"\nevt_trim hit the corrupted row in {hit:.0%} of corrupted cells and touched "
              f"a clean context in {touched:.0%}")


if __name__ == "__main__":
    main()
