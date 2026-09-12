# -*- coding: utf-8 -*-
"""Put the tail back on the output instead of cleaning the input.

WHY
---
Every repair measured so far acts on the context: clip, log, tail_log, evt_trim.
The best of them, a cap at median + 200 robust sd, works but cannot tell a data
error from a genuine extreme, and its threshold was picked from three values.
The literature's answer to the same problem is the other side: leave the context
alone, take the model as a black box, and repair its upper quantiles afterwards
by fitting a generalised Pareto tail to its own errors. Two references carry it,
both checked against Crossref and arXiv before this was written:

  * Pasche, Lam and Engelke (2026), "Extreme conformal prediction: reliable
    intervals for high-impact events", Extremes 29:129-155,
    doi 10.1007/s10687-026-00536-9 (arXiv 2505.08578). Fits a GPD to the
    conformity scores and extrapolates the score quantile past the calibration
    data, which is the only way to get a level beyond 1 - 1/(n_calib + 1).
  * Romano, Patterson and Candes (2019), conformalised quantile regression, the
    base method whose empirical score quantile degenerates at exactly that
    level.

This also tests the mechanism from the other end. If a GPD on the scores
restores Q(0.99) under a context error, then the damage is in the model's
quantile and not in anything the repair needs to know about the error: the
output layer never sees which row was wrong.

WHAT IS MEASURED
----------------
The context of 2000 rows is split once, 1500 to fit and 500 to calibrate, and
the test set of 1000 clean rows is the arbiter. Two treatments of the context,
crossed with two error states, four fits per cell:

    raw         the context as it is
    clip_200    capped at median + 200 * robust_sd of the fit part
    factor 1    clean
    factor 100  one row of the fit part times 100, a value in cents

The corrupted row is drawn among those where the error would reach an sd shift
of 2, as in `unit_error_real.py`, and it always lands in the fit part, so the
calibration rows are clean.

For each level a in 0.9, 0.99, 0.999 the model's own quantile is the base, and
the one-sided conformity score on the calibration rows is s_i = y_i - q_a(x_i):

    none        the model's quantile, as shipped
    cqr_emp     q_a(x) + the ceil((n+1)a)-th smallest score; undefined when that
                index exceeds n, which at n = 500 is exactly a = 0.999, and then
                it falls back to the largest score and is marked degenerate
    cqr_gpd     q_a(x) + a GPD quantile of the scores: peaks over the 90th
                percentile of s, shape and scale by maximum likelihood, the
                level converted with the exceedance rate

Reported per cell: coverage of the bound on the test rows (the number a user
reads), pinball at the same level, the median bound, and the fitted shape.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  T1. cqr_emp at 0.999 is degenerate at n_calib = 500 and under-covers.
  T2. cqr_gpd covers closer to nominal than cqr_emp at 0.999, and closer than
      the model alone at 0.99 and 0.999.
  T3. Under factor 100 the model's own coverage at 0.99 falls, and both
      conformal layers bring it back without being told that a row is wrong. If
      they do, an output repair replaces the input repair, and the trade-off
      that the clip cannot avoid disappears.
  T4. clip_200 plus cqr_gpd is not better than cqr_gpd on the raw context: once
      the output carries a tail, cleaning the input adds little.

Results in `tail_splice.csv`.

OUTCOME, 12.9.2026
------------------
840 cells, none failed. The unit of the counts is a dataset-model pair, 21.

  T1 HELD structurally. At n_calib = 500 all 840 cqr_emp cells at 0.999 are
     degenerate, since ceil(501 * 0.999) = 501 > 500. It still covers 0.9985,
     because the largest of 500 heavy-tailed scores is large, but the level is
     not identified by the data.
  T2 HELD. cqr_gpd is closer to nominal coverage than the model alone in 16 of
     21 units at 0.999 (p = 0.027), 15 of 21 at 0.99 (p = 0.078), and 16 of 21
     at 0.99 under the error (p = 0.027).
  T3 HELD where it was needed. At 0.99 the error takes TabPFN-V3's coverage from
     0.9855 to 0.9805 and the patch returns it to 0.9910, without being told
     which row is wrong. GBM goes 0.9845 to 0.9900. TabICLv2 was already at
     0.9915 and stays there.
  T4 HELD except on GBM. Under the error at 0.99, clip_200 plus cqr_gpd gives a
     pinball of 328 against 364 for the patch alone on GBM, but 326 against 327
     on TabICLv2 and 284 against 277 on TabPFN-V3.

The price is sharpness, and it is the opposite trade from the clip's. At 0.99 the
patch costs 3.7% of pinball on a clean context and 6.9% under the error; at 0.999
it gains 6.1% and 2.6%. Coverage is bought with a wider bound, which is what a
conformal bound is for. The clip cannot tell an error from an extreme; the patch
does not need to, and never gets sharper at 0.99.

The shape fitted to the conformity scores is 0.14 to 0.61 by dataset, median
0.34. The model's own errors are heavy-tailed, which is why the empirical score
quantile cannot reach 0.999 and why a GPD on those scores can.

RUNNING
-------
    python -u experiments/h3_repair/tail_splice.py
    DATASETS=AutoClaims MODELS=GBM REPEATS=2 python -u tail_splice.py
"""
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from common import append, datasets, metrics, models, paths, quiet

quiet.silence()


def _sibling(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _sibling("clip_context.py")
UE = _sibling("unit_error_real.py")

_default = ["freMTPL2sev"] + paths.lines("external_selected.txt")
DATASETS = (os.environ["DATASETS"].split(",") if os.environ.get("DATASETS")
            else _default)
MODELS = models.parse_list(os.environ.get("MODELS", "TabPFN-V3,TabICLv2,GBM"))
REPEATS = int(os.environ.get("REPEATS", "10"))
N_FIT = int(os.environ.get("N_FIT", "1500"))
N_CALIB = int(os.environ.get("N_CALIB", "500"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
N_EST = int(os.environ.get("N_EST", "1"))
CLIP_C = float(os.environ.get("CLIP_C", "200"))
U_SHARE = float(os.environ.get("U_SHARE", "0.9"))   # threshold for the GPD, on the scores
FACTORS = [1.0, 100.0]
TREATMENTS = ["raw", "clip_200"]
LEVELS = [0.9, 0.99, 0.999]
OUT = os.environ.get("OUTPUT", "tail_splice.csv")

COLUMNS = ["dataset", "model", "repeat", "factor", "treatment", "arm", "level",
           "n_fit", "n_calib", "n_test", "n_est", "sd_shift", "coverage", "pinball",
           "q_median", "xi_hat", "scale_hat", "u_score", "n_exc", "degenerate",
           "seconds", "reason"]
KEY = ["dataset", "model", "repeat", "factor", "treatment", "arm", "level",
       "n_fit", "n_calib", "n_test", "n_est"]


def score_quantile(scores, level, u_share=U_SHARE):
    """The conformity-score quantile three ways: empirical with the conformal
    correction, and a GPD extrapolation above the u_share threshold."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * level))
    emp = (float(s[-1]), True) if k > n else (float(s[k - 1]), False)

    u = float(np.quantile(s, u_share))
    exc = s[s > u] - u
    zeta = len(exc) / n
    out = dict(u_score=u, n_exc=int(len(exc)), xi_hat=np.nan, scale_hat=np.nan)
    if len(exc) < 20 or level <= 1 - zeta:
        # Not extrapolating: inside the calibration data the empirical quantile is
        # the honest answer, and a GPD fitted to fewer than 20 exceedances is not.
        return emp, (emp[0], emp[1]), out
    try:
        xi, _, scale = genpareto.fit(exc, floc=0.0)
        out.update(xi_hat=float(xi), scale_hat=float(scale))
        r = (1 - level) / zeta
        if abs(xi) < 1e-8:
            q = u + scale * np.log(1 / r)
        else:
            q = u + scale / xi * (r ** (-xi) - 1)
        if not np.isfinite(q):
            raise RuntimeError("non-finite GPD quantile")
        return emp, (float(q), False), out
    except Exception:
        return emp, (emp[0], emp[1]), out


def main():
    done = append.done(OUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    t0 = time.time()
    n_ctx = N_FIT + N_CALIB
    for name in DATASETS:
        X, y = datasets.load(name)
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        rng = np.random.default_rng(31337)
        plan = []
        for _ in range(REPEATS):
            idx = rng.permutation(len(y))[:n_ctx + N_TEST]
            sh = UE.shift_if_corrupted(y[idx[:N_FIT]], FACTORS[-1])
            vis = np.flatnonzero(sh >= UE.VISIBLE_SHIFT)
            plan.append((idx, int(rng.choice(vis)) if len(vis) else -1))
        for r, (idx, bad) in enumerate(plan):
            if bad < 0:
                print(f"  {name}: #{r} no row at which a unit error is visible", flush=True)
                continue
            fi, ci, ti = idx[:N_FIT], idx[N_FIT:n_ctx], idx[n_ctx:]
            Xf, yf0 = X[fi], y[fi]
            Xc, yc, Xte, yte = X[ci], y[ci], X[ti], y[ti]
            Xq = np.vstack([Xc, Xte])                  # one predict for both
            for model in MODELS:
                for factor in FACTORS:
                    yf = yf0.copy()
                    yf[bad] *= factor
                    shift = float(metrics.sd_shift(yf))
                    for treatment in TREATMENTS:
                        keys = [dict(dataset=name, model=model, repeat=r, factor=factor,
                                     treatment=treatment, arm=a, level=lv, n_fit=N_FIT,
                                     n_calib=N_CALIB, n_test=N_TEST, n_est=N_EST)
                                for a in ("none", "cqr_emp", "cqr_gpd") for lv in LEVELS]
                        todo = [k for k in keys if append.key(k, KEY) not in done]
                        if not todo:
                            continue
                        t1 = time.time()
                        y_fit = (yf if treatment == "raw"
                                 else CC.transform("clip", CLIP_C, yf)[0])
                        try:
                            q = models.quantiles(model, Xf, y_fit, Xq, seed=7000 + r,
                                                 levels=LEVELS, n_est=N_EST)
                            err = None
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"[:110]
                        secs = round((time.time() - t1) / max(len(todo), 1), 2)
                        bounds = {}
                        if not err:
                            for i, lv in enumerate(LEVELS):
                                qc, qt = q[:N_CALIB, i], q[N_CALIB:, i]
                                emp, gpd, info = score_quantile(yc - qc, lv)
                                bounds[lv] = dict(none=(qt, 0.0, False, info),
                                                  cqr_emp=(qt, emp[0], emp[1], info),
                                                  cqr_gpd=(qt, gpd[0], gpd[1], info))
                        for k in todo:
                            row = dict(k, sd_shift=shift, seconds=secs)
                            if err:
                                row["reason"] = err
                            else:
                                qt, add, degen, info = bounds[k["level"]][k["arm"]]
                                b = qt + add
                                row.update(info, degenerate=degen, reason="",
                                           coverage=float((yte <= b).mean()),
                                           pinball=CC.pinball(yte, b, k["level"]),
                                           q_median=float(np.median(b)))
                            append.write(OUT, row, COLUMNS)
                        if not err:
                            msg = "  ".join(
                                f"{lv}: {bounds[lv]['none'][0].mean():.0f}"
                                f"->{(bounds[lv]['cqr_gpd'][0] + bounds[lv]['cqr_gpd'][1]).mean():.0f}"
                                f" cov {float((yte <= bounds[lv]['none'][0]).mean()):.3f}"
                                f"/{float((yte <= bounds[lv]['cqr_gpd'][0] + bounds[lv]['cqr_gpd'][1]).mean()):.3f}"
                                for lv in LEVELS)
                        else:
                            msg = err
                        print(f"  {name[:14]:14s} {model:10s} #{r} f{factor:.0f} {treatment:8s} "
                              f"sd={shift:5.2f}  {msg}", flush=True)
    report(t0)


def report(t0=None):
    path = paths.result(OUT)
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        return
    for col in ("coverage", "pinball"):
        print(f"\n=== {col}, median over datasets and repeats ===")
        print(d.pivot_table(index=["level", "factor", "treatment"], columns=["model", "arm"],
                            values=col, aggfunc="median").round(4).to_string())
    print("\n=== coverage by dataset, level 0.99, raw context ===")
    s = d[(d.level == 0.99) & (d.treatment == "raw")]
    print(s.pivot_table(index=["dataset", "factor"], columns=["model", "arm"],
                        values="coverage", aggfunc="median").round(3).to_string())
    g = d[(d.arm == "cqr_gpd") & d.xi_hat.notna()]
    if len(g):
        print(f"\nfitted GPD shape on the scores: median {g.xi_hat.median():.3f}, "
              f"quartiles {g.xi_hat.quantile(0.25):.3f} to {g.xi_hat.quantile(0.75):.3f}, "
              f"exceedances {int(g.n_exc.median())}")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
