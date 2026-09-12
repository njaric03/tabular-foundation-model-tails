# -*- coding: utf-8 -*-
"""A unit error in one row of the context, on real heavy-tailed tables.

WHY
---
`clip_context_real.py` tests the repairs where leverage occurs naturally, and in
the whole pool that is one table, freMTPL2sev. `external_selection.py` added six
heavy-tailed tables from insurance, health expenditure and online popularity,
with Hill indices from 0.48 to 1.05, and none of them reaches an sd shift of 2
in 200 subsamples of 2000; the largest is 1.81. A heavy tail alone does not
produce the leverage that part two measures. What produces it is an error, so
the test on those tables has to bring one: a single context row whose target is
multiplied by 100, a value entered in cents instead of units.

This also gives the repairs a truth that the natural test lacks. The corrupted
row is known, so it can be checked whether an arm touched that row and whether
it touched anything else, and the clean context of the same subsample is the
reference: the prediction the model would have made had the error not been
there.

DESIGN
------
Datasets: freMTPL2sev and `data/external_selected.txt`. On each, REPEATS
random subsamples of N_FIT + N_TEST rows with a positive target, and one row of
the context chosen as the one to corrupt. Both are drawn up front from seed
31337, so a resumed run corrupts the same row.

The row is drawn uniformly among those whose error would reach an sd shift of
at least 2, the level at which part two sees a loss. A first draft drew it
uniformly from all rows; its smoke test on AutoClaims put the corrupted value
at rank 9 with a shift of 1.03, which tests nothing, because a unit error on a
small value is invisible. The condition is the same kind as the bins of the
natural test, and what it conditions away is kept: `share_visible` is the
fraction of context rows at which a unit error would reach that shift.

    factor 1     the clean context; what each arm costs when nothing is wrong
    factor 100   the chosen context row times 100

Arms, all from `clip_context.transform`: raw, clip_50, clip_200, log, tail_log
(C = 50), evt_trim. The reference is raw on the clean context, same subsample,
same seed. `d_q*` is the median relative change of the predicted quantile
against it; pinball is on the clean held-out rows.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  U1. Measured, not predicted: `share_visible`, how often a unit error in a
      random row would be visible at all.
  U2. Under the corruption, raw worsens pinball at 0.99 against the clean
      reference more for the two foundation models than for GBM.
  U3. clip_200 brings pinball at 0.99 back to within 2% of the clean reference
      in most corrupted cells, and costs less than 2% on the clean context.
  U4. evt_trim touches the corrupted row wherever it lands at least ten times
      above the second largest value, and touches a clean context in at most
      5% of subsamples. (The smoke test of `clip_context.py`, seen before this
      was written, showed it missing an injected row at a shift of 20 on the
      generator; the prediction is kept as it was drafted.)
  U5. log removes most of the corruption's effect, and costs more than
      clip_200 at 0.999 on the clean context.

Results in `unit_error_real.csv`.

OUTCOME, 12.9.2026
------------------
Seven tables, three models, ten subsamples each, 2520 rows, none failed. The unit
of the counts below is a dataset-model pair, 21 of them.

  U1 measured. A unit error is visible at 4% of rows on freMTPL2sev, 8 to 11% on
     BlogFeedback, beMTPL97 and MEPS, and 20 to 33% on ausprivauto0405,
     AutoClaims and norauto. Where it lands it lifts the sd shift to 2.8 to 5.5
     and sits 4 to 15 times above the second largest value.
  U2 WRONG. The error costs GBM most, not the foundation models: median pinball
     at 0.99 rises by 35% on BlogFeedback and 25% on MEPS, against TabICLv2's
     9 and 14% and TabPFN-V3's 8 and 3%. On freMTPL2sev it costs nothing
     measurable, because that context's own largest claim already dominates.
  U3 HELD, and it is the strongest argument for the cap that exists. clip_200
     beats raw in 17 of 21 units (p = 0.007, median -2.3%), its worst cell is
     +5.1%, and on a clean context it costs 0.00% at the median and +1.4% at
     worst. C = 200 was picked after a run, but it behaves as a safe default.
  U4 FAILED. evt_trim touches the corrupted row in 10% of cells and nothing else
     in any cell, on any table. Same as on the generator.
  U5 HALF. log repairs as often (17 of 21, -1.8%) and is not safe: 22 of 210
     cells overflow, the worst by 2.8e43, because a quantile over-predicted in
     log space is exponentiated. clip_50 repairs more (-3.7%) and costs more
     (+64% at worst under the error, +85% on a clean context). tail_log is
     neither reliable (15 of 21) nor cheap (+63% at worst).

RUNNING
-------
    python -u experiments/h3_repair/unit_error_real.py
    DATASETS=beMTPL97 MODELS=GBM REPEATS=2 python -u unit_error_real.py
"""
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import append, datasets, metrics, models, paths, quiet

quiet.silence()


def _sibling(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _sibling("clip_context.py")

_default = ["freMTPL2sev"] + paths.lines("external_selected.txt")
DATASETS = (os.environ["DATASETS"].split(",") if os.environ.get("DATASETS")
            else _default)
MODELS = models.parse_list(os.environ.get("MODELS", "TabPFN-V3,TabICLv2,GBM"))
REPEATS = int(os.environ.get("REPEATS", "10"))
N_FIT = int(os.environ.get("N_FIT", "2000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
N_EST = int(os.environ.get("N_EST", "1"))
FACTORS = [1.0, 100.0]
ARMS = [("raw", 0.0), ("clip", 50.0), ("clip", 200.0), ("log", 0.0),
        ("tail_log", CC.TAIL_LOG_C), ("evt_trim", 0.0)]
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ("50", "90", "99", "999")
OUT = os.environ.get("OUTPUT", "unit_error_real.csv")

COLUMNS = (["dataset", "model", "repeat", "factor", "arm", "clip_c", "n_est", "n_fit",
            "n_test", "sd_shift", "share_visible", "corrupt_rank", "corrupt_over_next",
            "n_touched",
            "hit"]
           + [f"d_q{t}" for t in TAGS] + [f"pb{t}" for t in TAGS]
           + ["n_above99", "seconds", "reason"])
KEY = ["dataset", "model", "repeat", "factor", "arm", "clip_c", "n_est", "n_fit",
       "n_test"]


def change(q_ref, q):
    out = []
    for i in range(len(LEVELS)):
        a, b = q_ref[:, i], q[:, i]
        ok = np.abs(a) > 1e-9
        out.append(float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan)
    return out


VISIBLE_SHIFT = 2.0


def shift_if_corrupted(yf, factor):
    """sd shift of the context if row i alone were multiplied by `factor`, for
    every i, without building 2000 copies."""
    n = len(yf)
    s, q = yf.sum(), (yf ** 2).sum()
    new = yf * factor
    s1, q1 = s + new - yf, q + new ** 2 - yf ** 2
    sd_all = np.sqrt(np.maximum(q1 / n - (s1 / n) ** 2, 0))
    top = yf.max()
    # Without the largest value: that is the corrupted row when it became the
    # largest, otherwise the old maximum.
    drop = np.where(new >= top, new, top)
    s2, q2 = s1 - drop, q1 - drop ** 2
    sd_wo = np.sqrt(np.maximum(q2 / (n - 1) - (s2 / (n - 1)) ** 2, 1e-24))
    return sd_all / sd_wo


def draws(y):
    """Subsample, corrupted row and the share of visible rows, for every repeat,
    drawn up front."""
    rng = np.random.default_rng(31337)
    out = []
    for _ in range(REPEATS):
        idx = rng.permutation(len(y))[:N_FIT + N_TEST]
        sh = shift_if_corrupted(y[idx[:N_FIT]], FACTORS[-1])
        visible = np.flatnonzero(sh >= VISIBLE_SHIFT)
        bad = int(rng.choice(visible)) if len(visible) else -1
        out.append((idx, bad, float(len(visible) / N_FIT)))
    return out


def main():
    done = append.done(OUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    t0 = time.time()
    for name in DATASETS:
        X, y = datasets.load(name)
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        for r, (idx, bad, share) in enumerate(draws(y)):
            if bad < 0:
                print(f"  {name}: #{r} no row at which a unit error is visible", flush=True)
                for model in MODELS:
                    append.write(OUT, dict(dataset=name, model=model, repeat=r, factor=100.0,
                                           arm="raw", clip_c=0.0, n_est=N_EST, n_fit=N_FIT,
                                           n_test=N_TEST, share_visible=0.0,
                                           reason="no visible row"), COLUMNS)
                continue
            fi, ti = idx[:N_FIT], idx[N_FIT:]
            Xf, y_clean, Xte, yte = X[fi], y[fi], X[ti], y[ti]
            contexts = {}
            for f in FACTORS:
                yc = y_clean.copy()
                yc[bad] *= f
                srt = np.sort(yc)[::-1]
                rank = int((yc > yc[bad]).sum()) + 1
                nxt = srt[1] if rank == 1 else srt[0]
                contexts[f] = (yc, dict(sd_shift=float(metrics.sd_shift(yc)),
                                        share_visible=share, corrupt_rank=rank,
                                        corrupt_over_next=float(yc[bad] / nxt)))
            for model in MODELS:
                keys = [dict(dataset=name, model=model, repeat=r, factor=f, arm=a,
                             clip_c=c, n_est=N_EST, n_fit=N_FIT, n_test=N_TEST)
                        for f in FACTORS for a, c in ARMS]
                todo = [k for k in keys if append.key(k, KEY) not in done]
                if not todo:
                    continue
                seed = 7000 + r
                q = lambda yc: models.quantiles(model, Xf, yc, Xte, seed=seed,
                                                levels=LEVELS, n_est=N_EST)
                t1 = time.time()
                try:
                    q_ref = q(y_clean)
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
                            y_fit, back, n = CC.transform(k["arm"], k["clip_c"], yc)
                            if k["arm"] == "raw" and k["factor"] == 1.0:
                                qa = q_ref
                            else:
                                with np.errstate(over="ignore", invalid="ignore"):
                                    qa = back(q(y_fit))
                            row.update(n_touched=n,
                                       hit=bool(y_fit[bad] < yc[bad]) if k["factor"] > 1 else "",
                                       reason="")
                            row.update({f"d_q{t}": v for t, v in zip(TAGS, change(q_ref, qa))})
                            for a, t in zip(LEVELS, TAGS):
                                row[f"pb{t}"] = CC.pinball(yte, qa[:, LEVELS.index(a)], a)
                            row["n_above99"] = int((yte > qa[:, LEVELS.index(0.99)]).sum())
                        except Exception as e:
                            row["reason"] = f"{type(e).__name__}: {e}"[:110]
                    row["seconds"] = round(time.time() - t2, 1)
                    append.write(OUT, row, COLUMNS)
                f100 = contexts[100.0][1]
                print(f"  {name[:16]:16s} {model:10s} #{r} shift x100 {f100['sd_shift']:6.2f} "
                      f"rank {f100['corrupt_rank']} over_next {f100['corrupt_over_next']:7.1f}"
                      f"  [{time.time() - t1:.0f}s, {len(todo)} arms]", flush=True)
    report(t0)


def report(t0=None):
    path = paths.result(OUT)
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
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
        print(d.pivot_table(index=["factor", "dataset", "model"], columns="arm_c",
                            values=col, aggfunc="median").round(3).to_string())
    c = d[(d.factor == 100.0) & (d.arm == "evt_trim")]
    if len(c):
        print(f"\nevt_trim hit the corrupted row in {c.hit.astype(str).eq('True').mean():.0%} "
              f"of corrupted cells; touched a clean context in "
              f"{(d[(d.factor == 1.0) & (d.arm == 'evt_trim')].n_touched > 0).mean():.0%}")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
