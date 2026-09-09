# -*- coding: utf-8 -*-
"""What `dissociation_real_slope.py` reports for a model that is RIGHT.

The ceiling of the measuring instrument, separated from the blindness of the
model.

WHY
---
The slope from `dissociation_real_slope.py` is read in the text as a share:
1.00 means the model tracks the reference gradient, 0.40 that it captures two
fifths of it. That reading assumes a perfect model would score 1.00. The
assumption was never checked, and it need not hold, because the two halves are
measured with different instruments:

    reference   Hill on the residuals y / s_hat, per tercile
    model       inversion of (Q(0.99) - Q(0.5)) / (Q(0.9) - Q(0.5))

The first is an estimate with error, the second is near exact when the quantiles
are. The slope is a regression of the second on the first, and error in the
REGRESSOR pulls the slope -- but here away from zero rather than towards it,
because the noisy quantity is the denominator. If the reference underestimates
the true gradient by a factor c, a perfect model scores 1/c, not 1.

On real data the true xi(x) does not exist, so this cannot be measured there. On
the generator it can: xi(x) is known per row, so the model call can be replaced
by the TRUE conditional quantiles. Anything that then differs from 1.00 is the
instrument.

WHAT IS MEASURED
----------------
The same procedure as in `dissociation_real_slope.py`, with one change: the call
to a model is replaced by oracle quantiles s * G^-1(a; xi(x)), that is, what a
model that knows the distribution would return. Recorded:

    slope_shape, slope_shape_corrected   what a perfect model scores for shape
    slope_scale, slope_scale_corrected   the same for scale
    instrument_slope                     slope of the reference on the TRUE xi
                                         gradient, i.e. how much Hill on the
                                         residuals captures of what it measures

The feature count is a knob and a column, because it is the one quantity on
which the datasets in `dissociation_real_slope.csv` differ sharply: 6 for
diamonds up to 80 for superconduct. The default D list is exactly those values,
so every dataset can be compared against the ceiling at its own width with no
interpolation. The extra features here are pure noise -- their weights are zero
-- which is the worst case; on real data the extra columns carry signal too, so
the ceiling at large d is a lower bound.

WHY THE ESTIMATOR IS IMPORTED AND NOT COPIED
--------------------------------------------
A calibration is worth something only if it measures the SAME procedure. Copies
have already cost this repository 20 model dispatchers and one seeding bug, so
`grad_ref`, `grad_model`, `grad_skale`, `nagib`, `tercili` and
`skala_regresijom` are loaded from `dissociation_real_slope.py` by path. If that
file changes, the ceiling changes with it. N_FIT, N_TEST, QS, N_TERCILA and
MIN_PO_TERCILU come from there for the same reason: written out again here, they
could drift apart.

Those names are Serbian because that file is, like the rest of `experiments/`
written before 7 September. They are used as they stand rather than renamed:
renaming them means editing a script that carries a published claim.

The import also executes that module's top level, which reads MODELS from the
environment. No model is constructed, but `MODELS=` with an unknown name brings
this script down too.

K_SHARE, AND WHY IT IS A PATCH HERE
-----------------------------------
`reference_instrument.py` measured that the reference gradient depends strongly
on Hill's k, and `dissociation_real_slope.py`, which now carries the claim, has
no knob for k. Until it gets one, this script temporarily replaces the name
`metrics` in ITS namespace with a wrapper that forwards k_share, and restores it
in `finally`. The patch is ugly and deliberately visible; the right fix is a
knob over there. The `k_share` column exists and is part of the key, so runs
before and after that knob will not mix.

THE PREDICTION, AND IT FELL
---------------------------
The prediction is not recorded here but in the text of the thesis, which reads a
slope of 1.00 as perfect tracking. So: a perfect model scores 1.00 for both
halves, at every d. Outcome of the first pass (10 seeds per d, k = 0.10):

    P1 PASSED for scale.  slope_scale 1.00 at d=6 down to 0.85 at d=80;
       corrected 0.95 to 1.01 at every d. For scale, reading 1.00 as perfect is
       correct.
    P2 FAILED for shape.  The raw slope falls from 1.23 at d=6 to 0.37 at d=80,
       so it is not comparable between datasets of different width. The
       corrected slope IS independent of d (1.16 to 1.92) but sits around 1.5,
       not around 1.0.
    Consequence: the shape share has to be divided by the ceiling at its own d
       rather than read against 1.00. The 0.85-1.15 band holds for scale and
       does not hold for shape.

HOW TO READ IT
--------------
  slope_scale_corrected ~ 1     the scale instrument is unbiased, the reading stands
  slope_shape_corrected ~ 1.5   divide the shape share from the text by this
  instrument_slope < 1          the reference underestimates the true gradient by
                                that factor
  var_share_shape ~ 1           measurement error eats the whole signal; there the
                                correction hits the guard in `nagib` and the row
                                carries nothing

Results in `instrument_ceiling.csv`. No model is called, so the cost is minutes
rather than hours, and none of the four virtualenvs is needed.

RUNNING
-------
    python -u experiments/h1_shape_vs_scale/instrument_ceiling.py
    D=6,80 SEEDS=3 python -u instrument_ceiling.py
    for k in 0.05 0.10 0.20; do \
      K_SHARE=$k OUTPUT=instrument_ceiling_k$k.csv \
      python -u experiments/h1_shape_vs_scale/instrument_ceiling.py; done
"""
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import append, generator, metrics, paths, quiet

quiet.silence()


def _sibling(name):
    """Load a neighbouring experiment as a module, by path.

    `experiments/` is not a package, so a plain import does not reach it. The
    point is that the ceiling measures the very procedure being calibrated
    rather than a copy of it.
    """
    p = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRS = _sibling("dissociation_real_slope.py")

# Taken, not rewritten: a change there is a change here.
QS = DRS.QS
N_FIT, N_TEST = DRS.N_FIT, DRS.N_TEST
N_TERCILA, MIN_PO_TERCILU = DRS.N_TERCILA, DRS.MIN_PO_TERCILU

# Feature counts of the generator. The default is exactly the values that occur
# in `dissociation_real_slope.csv`, so every dataset meets the ceiling at its
# own width.
D_LIST = [int(v) for v in os.environ.get("D", "6,7,8,14,25,37,51,80").split(",")]
SEEDS = int(os.environ.get("SEEDS", "10"))
N_BOOT = int(os.environ.get("N_BOOT", str(DRS.N_BOOT)))
K_SHARE = float(os.environ.get("K_SHARE", "0.10"))
# Room for later: a second oracle, a deliberately noisy one, enters as a new
# value of this column, without changing the key and without mixing with rows
# already measured.
ARM = os.environ.get("ARM", "oracle")
OUT = os.environ.get("OUTPUT", "instrument_ceiling.csv")

COLUMNS = ["arm", "d", "seed", "k_share", "n_fit", "n_test", "n_boot", "n_features",
           "slope_shape", "slope_shape_corrected", "r_shape", "var_share_shape",
           "slope_scale", "slope_scale_corrected", "r_scale", "var_share_scale",
           "instrument_slope", "instrument_r", "true_grad_median",
           "seconds", "reason"]
KEY = ["arm", "d", "seed", "k_share", "n_fit", "n_test", "n_boot"]


class _HillWithK:
    """A wrapper around `common.metrics` that hands Hill a k_share.

    It exists only because `dissociation_real_slope.py` has no knob for k.
    Everything except `hill` passes straight through, so nothing else in that
    file is touched.
    """

    def __init__(self, k_share):
        self._k = k_share

    def hill(self, y, k_share=None, k_min=30):
        return metrics.hill(y, k_share=self._k if k_share is None else k_share,
                            k_min=k_min)

    def __getattr__(self, name):
        return getattr(metrics, name)


def oracle_quantiles(s, xi):
    """What a model that knows the distribution returns: the true GPD quantiles."""
    return np.column_stack([s * metrics.gpd_quantile(np.full(len(s), a), xi)
                            for a in QS])


def one(d, seed):
    rng = np.random.default_rng(seed)
    # The generator weights padded with zeros: the extra features carry neither
    # scale nor shape, which is the worst case for a slope through the origin.
    w = np.concatenate([generator.W, np.zeros(d - len(generator.W))])
    p = generator.gpd(N_FIT + N_TEST, rng, xi=generator.XI_OF_X, d=d, w=w)
    X, y, s, xi = p.X, p.y, p.s, p.xi

    fi = slice(0, N_FIT)
    ti = slice(N_FIT, N_FIT + N_TEST)
    Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]
    xi_te = xi[ti]

    s_hat = DRS.skala_regresijom(Xf, yf)
    s_te = s_hat(Xte)
    q = oracle_quantiles(s[ti], xi_te)

    ref_sh, mod_sh, var_sh, true_sh = [], [], [], []
    ref_sc, mod_sc, var_sc = [], [], []
    for j in range(X.shape[1]):
        v = Xf[:, j]
        if len(np.unique(v)) < N_TERCILA * 3:
            continue
        edges = np.quantile(v, np.linspace(0, 1, N_TERCILA + 1))
        if len(np.unique(edges)) < N_TERCILA + 1:
            continue
        terc = DRS.tercili(Xte[:, j], edges)
        if min((terc == t).sum() for t in range(N_TERCILA)) < MIN_PO_TERCILU:
            continue

        g_ref, g_mod = DRS.grad_ref(yte, s_te, terc), DRS.grad_model(q, terc)
        if not (np.isfinite(g_ref) and np.isfinite(g_mod)):
            continue
        # The true gradient: mean xi in the top tercile minus in the bottom one.
        # It exists only on the generator, and it is why this script has a point.
        g_true = float(xi_te[terc == N_TERCILA - 1].mean() - xi_te[terc == 0].mean())

        b = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(yte), len(yte))
            b.append(DRS.grad_ref(yte[k], s_te[k], terc[k]))
        b = np.array([x for x in b if np.isfinite(x)])
        if len(b) < N_BOOT // 2:
            continue
        ref_sh.append(g_ref)
        mod_sh.append(g_mod)
        var_sh.append(float(np.var(b)))
        true_sh.append(g_true)

        e1, e3 = (float(np.median(yte[terc == 0])),
                  float(np.median(yte[terc == N_TERCILA - 1])))
        m1, m3 = (float(np.median(q[terc == 0, QS.index(DRS.MID)])),
                  float(np.median(q[terc == N_TERCILA - 1, QS.index(DRS.MID)])))
        gs_ref, gs_mod = DRS.grad_skale(e1, e3), DRS.grad_skale(m1, m3)
        bs = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(yte), len(yte))
            t = terc[k]
            if min((t == u).sum() for u in (0, N_TERCILA - 1)) < 2:
                continue
            bs.append(DRS.grad_skale(float(np.median(yte[k][t == 0])),
                                     float(np.median(yte[k][t == N_TERCILA - 1]))))
        bs = np.array([x for x in bs if np.isfinite(x)])
        if np.isfinite(gs_ref) and np.isfinite(gs_mod) and len(bs) > 2:
            ref_sc.append(gs_ref)
            mod_sc.append(gs_mod)
            var_sc.append(float(np.var(bs)))

    if len(ref_sh) < 3:
        raise RuntimeError(f"only {len(ref_sh)} usable features")

    a_raw, a_cor, a_r, a_v = DRS.nagib(ref_sh, mod_sh, var_sh)
    b_raw, b_cor, b_r, b_v = DRS.nagib(ref_sc, mod_sc, var_sc)
    # Slope of the REFERENCE on the true gradient: how much the instrument itself
    # captures. No correction, because here the regressor is the true gradient
    # and it carries no measurement error.
    i_raw, _, i_r, _ = DRS.nagib(true_sh, ref_sh)

    return dict(n_features=len(ref_sh),
                slope_shape=a_raw, slope_shape_corrected=a_cor,
                r_shape=a_r, var_share_shape=a_v,
                slope_scale=b_raw, slope_scale_corrected=b_cor,
                r_scale=b_r, var_share_scale=b_v,
                instrument_slope=i_raw, instrument_r=i_r,
                true_grad_median=float(np.median(true_sh)), reason="")


def main():
    done = append.done(OUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    t0 = time.time()
    saved = DRS.metrics
    DRS.metrics = _HillWithK(K_SHARE)
    try:
        for d in D_LIST:
            if d < len(generator.W):
                print(f"d={d}: fewer than {len(generator.W)} features, skipping",
                      flush=True)
                continue
            for seed in [7000 + 1000 * i for i in range(SEEDS)]:
                k = dict(arm=ARM, d=d, seed=seed, k_share=K_SHARE,
                         n_fit=N_FIT, n_test=N_TEST, n_boot=N_BOOT)
                if append.key(k, KEY) in done:
                    continue
                t1 = time.time()
                try:
                    r = one(d, seed)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                r.update(k, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, COLUMNS)
                message = r.get("reason") or (
                    f"shape {r['slope_shape']:+.3f} (corr {r['slope_shape_corrected']:+.3f}) "
                    f"| scale {r['slope_scale']:+.3f} (corr {r['slope_scale_corrected']:+.3f}) "
                    f"| instrument {r['instrument_slope']:+.3f}")
                print(f"  d={d:>3} s={seed}  {message}  [{r['seconds']}s]", flush=True)
    finally:
        DRS.metrics = saved
    report(t0)


def report(t0=None):
    path = paths.result(OUT)
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        print("no successful rows")
        return

    print("\n=== CEILING: what a perfect model scores through the same measure ===")
    ceiling = d.groupby("d")[["n_features", "slope_shape", "slope_shape_corrected",
                              "slope_scale", "slope_scale_corrected",
                              "var_share_shape", "instrument_slope"]].median()
    print(ceiling.round(3).to_string())
    print("\n  For scale, 1.00 is the right expectation. For shape it is not: the raw"
          "\n  slope depends on the feature count, the corrected one does not but sits"
          "\n  around 1.5.")

    measured = paths.result("dissociation_real_slope.csv")
    if not os.path.exists(measured):
        if t0 is not None:
            print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")
        return

    m = pd.read_csv(measured)
    m = m[m.n_features.notna()]
    if m.empty:
        return
    m["d"] = m.n_features.astype(int)
    missing = sorted(set(m.d) - set(ceiling.index))
    m = m.join(ceiling.add_prefix("bm_"), on="d")
    m["cal_shape_cor"] = m.slope_shape_corrected / m.bm_slope_shape_corrected
    m["cal_scale_cor"] = m.slope_scale_corrected / m.bm_slope_scale_corrected

    print("\n=== the same numbers from the text, divided by the ceiling at their own"
          " feature count ===")
    t = m.groupby("model").agg(
        shape_as_written=("slope_shape_corrected", "median"),
        shape_calibrated=("cal_shape_cor", "median"),
        scale_as_written=("slope_scale_corrected", "median"),
        scale_calibrated=("cal_scale_cor", "median"))
    print(t.round(3).to_string())
    if missing:
        print(f"\n  no ceiling, left uncalibrated: d = {missing}. "
              f"Run with D={','.join(map(str, missing))}.")
    print("\n  The dissociation is not overturned but widened: scale stays at 1, and the"
          "\n  shape share falls further, because until now it was divided by one rather"
          "\n  than by the ceiling.")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
