# -*- coding: utf-8 -*-
"""Clip the context on real data, on the subsamples that carry the leverage.

WHY
---
`clip_context.py` found a fix on the generator: cap the context at
`median + C * robust_sd` before the fit, and one leverage row no longer stretches
TabPFN's grid off the data. There the true tail index is known, so the question
was whether the reported xi comes back. It does. But that is TabPFN-V3 alone, a
row injected at the centre of x, and a C picked from three values afterwards.

The real-data evidence for part two is one dataset and one model: on
freMTPL2sev TabICLv2 loses about 15% of Q(0.99) on subsamples whose largest
claim shifts the sd by two or more. This asks whether the same clip removes
that loss where it was measured, without a synthetic row and without a known
truth.

WHAT IS MEASURED
----------------
The subsamples are drawn exactly as `prevalence_models.py` draws them: rejection
sampling into four bins of the natural sd shift, seed 31337, N_FIT + N_TEST rows,
five per bin. The raw arm therefore has to reproduce the wide sweep's `d_q99`
on the same cells, and that is the control that the instrument is the same one.

For each subsample, three contexts, one fit each:

    without_max   the largest row removed; the reference the wide sweep uses
    raw           the full context, as shipped
    clip_C        the full context with y capped at median + C * robust_sd

`d_q*` is the median relative change of the predicted quantile against
`without_max`, exactly as in `prevalence_models.py`. If clipping is a fix, a
context that still holds the leverage row behaves like one that never had it:
`d_q99` of clip_C near zero where raw is far from it.

There is no true tail here, so the test set is the arbiter of cost: pinball at
each level on N_TEST held-out rows the context never saw. At 0.99 that is about
ten exceedances per cell; 0.999 is recorded with its count and is not a finding
at this size.

`robust_sd` and the cap are imported from `clip_context.py`, so C means the same
thing in both.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  P1. The raw arm reproduces `prevalence_models_wide.csv` for the same model,
      bin and repeat on freMTPL2sev. Same seed, same sizes, same code path.
  P2. In the bins at two sd and above, TabICLv2's raw `d_q99` is negative and
      clip_200's is near zero.
  P3. Below 1.2 sd the cap rarely binds at C = 200, so clipping costs close to
      nothing in pinball at 0.99. At C = 50 it binds more and may cost.
  P4. GBM moves little under any arm; trees do not standardise the target.

Results in `clip_context_real.csv`.

OUTCOME, 11.9.2026
------------------
  P1. Holds for GBM exactly and for TabICLv2 to 1e-5; `sd_shift` agrees on all
      60 cells. Fails for TabPFN-V3 by up to 0.03 per cell, 0.009 on a bin
      median. The sweep ran before venv-tfm was rebuilt (Python 3.11.0 then,
      3.12.8 now, the same package versions). Within one environment TabPFN is
      deterministic: 6 and 12 threads give identical output.
  P2. Holds. TabICLv2 at 4 sd and above: raw -15.4%, clip_200 +3.6%. At 2 to
      4 sd: -5.2% and +5.8%. TabPFN-V3 at 4 sd and above: -7.8% and +0.4%.
  P3. Holds. Below 2 sd the cap binds on 3 to 5 of 2000 values and pinball at
      0.99 moves by -0.3% to -1.2%, significant for no model. At 2 sd and above
      clip_200 beats raw in 9/10, 9/10 and 10/10 subsamples (TabICLv2,
      TabPFN-V3, GBM; sign test p = 0.021, 0.021, 0.002; medians -6.1%, -2.4%,
      -9.7%). C = 50 costs TabICL 25 to 34% of Q(0.99) in every bin.
  P4. Wrong. GBM's raw d_q99 has a median of +13% at 2 to 4 sd, and GBM gains
      the most from the clip in pinball. Not all of the gain is the grid.

SECOND RUN, 11.9.2026: THE ARMS THAT DO NOT NEED A PICKED C
-----------------------------------------------------------
log, tail_log (C = 50) and evt_trim from `clip_context.transform`, on the same
60 cells. No separate predictions were written here; P5 to P8 of
`clip_context.py` are the ones that apply. Pinball at 0.99 against raw, over
the ten subsamples at 2 sd and above:

    clip_200   TabICLv2 9/10 (p = 0.021)  TabPFN-V3 9/10 (0.021)  GBM 10/10 (0.002)
    log                 8/10 (0.109)               9/10 (0.021)        8/10 (0.109)
    tail_log            9/10 (0.021)               6/10 (0.754)        9/10 (0.021)
    evt_trim   touches no context in any of the 60 cells; identical to raw

log moves TabPFN-V3's Q(0.99) by +20 to +37% against without_max in every
bin, the clean ones included: it gives a different model, not the reference
back. For TabICLv2 at 4 sd and above it makes the loss larger (-19.7% against
raw's -15.4%). tail_log costs TabPFN-V3 a pinball of 854 against 660 at 2 to 4
sd. clip_200 stays the only arm that helps all three models where the leverage
is and moves nothing where it is not.

THIRD RUN, 12.9.2026: SIX MODELS, AND THE MECHANISM AS A NATURAL EXPERIMENT
--------------------------------------------------------------------------
TabPFN-v2.5, TabPFN-v2.6 and EXAONE-Tabular were added on the same 20 subsamples.
EXAONE runs in `venv-tabfm`, which is where its package lives; the run in
`venv-tabfm` is the one `run/run_exaone_a.sh` has always used, and a first attempt
in `venv-tfm` failed outright with ModuleNotFoundError, 240 rows that were dropped.

Raw `d_q99` against a context without the largest row, median by bin:

| model | 2 to 4 sd | 4 sd and above |
|---|---|---|
| TabPFN-v2.5 | -29.7% | **-51.6%** |
| TabPFN-v2.6 | -54.6% | -48.5% |
| TabICLv2 | -5.2% | -15.4% |
| TabPFN-V3 | -6.8% | -7.8% |
| GBM | +13.3% | -1.7% |
| EXAONE | +2.8% | +0.9% |

Two things follow, and the second is the stronger.

First, the loss is not one model's quirk: two TabPFN generations lose about half
of the predicted Q(0.99) on real subsamples whose largest claim shifts the sd, and
V3 is the least affected of the three. The mechanism weakens across generations
without being removed.

Second, EXAONE does not lose anything, and its source says why it is the right
control. `regressor.py:341` centres the target with `selected_y.mean()` and scales
it with `selected_y.std(correction=1)`: the same non-robust standardisation the
other packages use. What differs is the head, a bank of predicted quantiles rather
than a fixed grid of borders in z-space. Under an inflated sd a quantile head
scales affinely and the relative change cancels; a fixed grid loses resolution
over the data. So the damage is in the discretisation, not in the standardisation,
which is what part two argued from the code and can now be read off a comparison
across packages. TabICLv2 is the case that keeps this honest: it also reports
quantiles and still loses 15%, so the head type alone is not the whole story and
its own head has not been read yet.

Repairs scale with the damage, which is the third piece of evidence for the same
mechanism. Pinball at 0.99 against raw on the ten subsamples at 2 sd and above:

    clip_200   v2.6 10/10 (-17.9%)  v2.5 10/10 (-9.5%)  V3 9/10 (-2.4%)  EXAONE 10/10 (-1.8%)
    sd_cap     v2.6 10/10 (-15.9%)  v2.5  9/10 (-9.4%)  V3 7/10 (-1.7%)  EXAONE  9/10 (-1.6%)
    rank_gpd   v2.6  9/10 (-10.8%)  v2.5 10/10 (-11.6%) V3 9/10 (-5.0%)  EXAONE  6/10 (-1.9%)
    evt_trim   0/10 on every one of the six models: it touches nothing, anywhere.

RUNNING
-------
    python -u experiments/h3_repair/clip_context_real.py
    DATASETS=OnlineNewsPopularity MODELS=TabICLv2 python -u clip_context_real.py
"""
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import append, datasets, models, paths, quiet

quiet.silence()


def _sibling(name):
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CC = _sibling("clip_context.py")

DATASETS = os.environ.get("DATASETS", "freMTPL2sev").split(",")
MODELS = models.parse_list(os.environ.get("MODELS", "TabPFN-V3,TabICLv2,GBM"))
CLIP_C = [float(v) for v in os.environ.get("CLIP_C", "50,200").split(",")]
# The same sizes, bins, quota and seed as the wide sweep, so the subsamples are
# the same ones and the raw arm can be checked against it.
N_FIT = int(os.environ.get("N_FIT", "2000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
N_EST = int(os.environ.get("N_EST", "1"))
N_PER_BIN = int(os.environ.get("N_PER_BIN", "5"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "4000"))
BINS = [(0.0, 1.2), (1.2, 2.0), (2.0, 4.0), (4.0, 1e9)]
LEVELS = [0.5, 0.9, 0.99, 0.999]
OUT = os.environ.get("OUTPUT", "clip_context_real.csv")

COLUMNS = ["dataset", "model", "bin", "repeat", "arm", "clip_c", "n_est", "n_fit",
           "n_test", "sd_shift", "leverage", "n_clipped",
           "d_q50", "d_q90", "d_q99", "d_q999",
           "pb50", "pb90", "pb99", "pb999", "n_above99", "seconds", "reason"]
KEY = ["dataset", "model", "bin", "repeat", "arm", "clip_c", "n_est", "n_fit",
       "n_test"]


def bin_name(k):
    lo, hi = BINS[k]
    return f"{lo}-{hi if hi < 1e8 else 'inf'}"


def subsamples(y, rng):
    """`prevalence_models.nadji_poduzorke`, line for line."""
    n = min(N_FIT + N_TEST, len(y))
    found = {k: [] for k in range(len(BINS))}
    for _ in range(MAX_ATTEMPTS):
        if all(len(v) >= N_PER_BIN for v in found.values()):
            break
        idx = rng.permutation(len(y))[:n]
        yf = y[idx[:N_FIT]]
        p = np.sort(yf)
        shift = np.std(yf) / max(np.std(p[:-1]), 1e-12)
        lev = abs(p[-1]) / max(abs(p[-2]), 1e-12)
        for k, (lo, hi) in enumerate(BINS):
            if lo <= shift < hi and len(found[k]) < N_PER_BIN:
                found[k].append((idx, shift, lev))
                break
    return found


def change(q_ref, q):
    """Median relative change per level, as `prevalence_models.promena` computes it."""
    out = []
    for i in range(len(LEVELS)):
        a, b = q_ref[:, i], q[:, i]
        ok = np.abs(a) > 1e-9
        out.append(float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan)
    return out


def arm_row(q, q_ref, yte):
    d = change(q_ref, q)
    r = {f"d_q{s}": v for s, v in zip(("50", "90", "99", "999"), d)}
    for a, s in zip(LEVELS, ("50", "90", "99", "999")):
        r[f"pb{s}"] = CC.pinball(yte, q[:, LEVELS.index(a)], a)
    r["n_above99"] = int((yte > q[:, LEVELS.index(0.99)]).sum())
    return r


def main():
    done = append.done(OUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    ids = paths.load_json("sb_openml_ids.json")
    arms = ([("without_max", 0.0), ("raw", 0.0)] + [("clip", c) for c in CLIP_C]
            + [("log", 0.0), ("tail_log", CC.TAIL_LOG_C), ("evt_trim", 0.0),
               ("sd_cap", CC.TAU), ("rank_emp", 0.0), ("rank_gpd", 0.0),
               ("rank_gpd_trim", 0.0), ("rank_exp", 0.0)])
    t0 = time.time()
    for name in DATASETS:
        X, y = datasets.load(name, ids)
        ok = np.isfinite(y) & (y > 0)            # the wide sweep's filter, same order
        X, y = X[ok], y[ok]
        found = subsamples(y, np.random.default_rng(31337))
        for k, cells in found.items():
            for j, (idx, shift, lev) in enumerate(cells):
                fi, ti = idx[:N_FIT], idx[N_FIT:]
                Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]
                keep = np.ones(len(yf), bool)
                keep[int(np.argmax(yf))] = False
                for model in MODELS:
                    keys = [dict(dataset=name, model=model, bin=bin_name(k), repeat=j,
                                 arm=a, clip_c=c, n_est=N_EST, n_fit=N_FIT,
                                 n_test=N_TEST) for a, c in arms]
                    todo = [kk for kk in keys if append.key(kk, KEY) not in done]
                    if not todo:
                        continue
                    t1 = time.time()
                    seed = 7000 + j                    # as in prevalence_models
                    q = lambda Xc, yc: models.quantiles(model, Xc, yc, Xte, seed=seed,
                                                        levels=LEVELS, n_est=N_EST)
                    try:
                        # The reference is refitted on resume as well; every model
                        # here is deterministic within one environment, so it is the
                        # same reference the earlier rows were measured against.
                        q_ref = q(Xf[keep], yf[keep])
                        rows = {("without_max", 0.0): (q_ref, 0)}
                        for kk in todo:
                            a, c = kk["arm"], kk["clip_c"]
                            if a == "without_max":
                                continue
                            y_fit, back, n = CC.transform(a, c, yf)
                            with np.errstate(over="ignore", invalid="ignore"):
                                rows[(a, c)] = (back(q(Xf, y_fit)), n)
                        err = None
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"[:110]
                    secs = round((time.time() - t1) / (len(todo) + 1), 1)
                    for kk in todo:
                        r = dict(kk, sd_shift=shift, leverage=lev, seconds=secs)
                        if err:
                            r["reason"] = err
                        else:
                            qa, nclip = rows[(kk["arm"], kk["clip_c"])]
                            r.update(arm_row(qa, q_ref, yte), n_clipped=nclip, reason="")
                        append.write(OUT, r, COLUMNS)
                    if err:
                        msg = err
                    else:
                        d = {a if a != "clip" else f"clip_{c:g}": arm_row(v[0], q_ref, yte)["d_q99"]
                             for (a, c), v in rows.items()}
                        msg = "  ".join(f"{a} {v:+6.1%}" for a, v in d.items() if a != "without_max")
                    print(f"  {name[:14]:14s} {model:10s} {bin_name(k):>8} #{j} sd={shift:5.2f}  "
                          f"d_q99: {msg}  [{secs}s/arm]", flush=True)
    report(t0)


def report(t0=None):
    path = paths.result(OUT)
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        return
    d["arm_c"] = np.where(d.arm == "clip", "clip_" + d.clip_c.map("{:g}".format), d.arm)
    for col in ("d_q99", "pb99"):
        print(f"\n=== {col}, median by bin ===")
        print(d.pivot_table(index=["dataset", "model", "bin"], columns="arm_c", values=col,
                            aggfunc="median").round(4).to_string())
    w = paths.result("prevalence_models_wide.csv")
    if os.path.exists(w):
        ref = pd.read_csv(w)
        ref = ref[ref.d_q99.notna()]
        raw = d[d.arm == "raw"]
        k = ["dataset", "model", "bin", "repeat"]
        j = raw.set_index(k)[["d_q99"]].join(ref.set_index(k)[["d_q99"]], rsuffix="_wide",
                                            how="inner")
        if len(j):
            gap = float((j.d_q99 - j.d_q99_wide).abs().max())
            print(f"\nP1 control, raw vs prevalence_models_wide on {len(j)} cells: "
                  f"max |d_q99 gap| = {gap:.2e}  ({'holds' if gap < 1e-3 else 'FAILS'})")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
