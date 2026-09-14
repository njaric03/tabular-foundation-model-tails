# -*- coding: utf-8 -*-
"""Put the tail back on the output instead of cleaning the input.

Every context repair either cannot tell an error from a genuine extreme (the cap) or
risks the 0.999 level (the rank maps). The other side: leave the context alone and repair
the model's upper quantiles afterwards with a GPD fitted to its own conformity scores,
after Pasche, Lam and Engelke (2026, Extremes 29:129-155, doi 10.1007/s10687-026-00536-9)
on top of conformalised quantile regression (Romano, Patterson and Candes, 2019).

The context of N_FIT + N_CALIB rows is split once, fit and calibration, and N_TEST clean
rows judge. Three context treatments (raw, clip_<CLIP_C>, rank_gpd), crossed with a clean
context and one fit row multiplied by 100. For each level, with the one-sided score
s_i = y_i - q_a(x_i) on the calibration rows:

    none           the model's quantile
    cqr_emp        plus the ceil((n+1)a)-th smallest score; past n it is degenerate
    cqr_gpd_<u>    plus a GPD quantile of the scores above their u quantile
    cqr_gpd_ks     u chosen by the Kolmogorov-Smirnov distance of the fitted tail
    cqr_gpd_cv     u chosen by out-of-fold pinball of the extrapolated quantile
    cqr_gpd_norm   the score divided by the row's spread Q(0.9|x) - Q(0.5|x)
    cond_gpd       the splice at the model's own Q(0.9|x), per row

Outcomes, 12.9.2026, 21 dataset-model units, paired:

  cqr_emp at 0.999 is degenerate in every cell (ceil(501 * 0.999) > 500). At 0.99 cqr_gpd
  brings coverage to nominal for the models that miss it, and under the error it brings TabPFN-V3's
  coverage at 0.99 from 0.9805 back to 0.9910
  without knowing which row is wrong. The price is sharpness: +3.7% pinball at 0.99 on a
  clean context, -6.1% at 0.999. The four thresholds differ by at most 0.06% of pinball
  and neither rule beats 0.90, so the threshold can be left to a rule. The locally adaptive
  score and the conditional splice both fail (worse in 14 and 16 of 21 units). The patch
  adds as much on a rank-transformed context as on a raw one, so the two repairs compose.

  At 5000 test rows (13.9.2026, `tail_splice_n5000.csv`, tested per calibration draw) the
  patch reaches nominal coverage at 0.999 for GBM only; TabICLv2 and TabPFN-V3 stay at 0.9955
  to 0.9961. An earlier claim of 16 of 21 units closer to nominal at 0.999 was measured on 1000
  test rows, which cannot resolve that level, and is withdrawn for the foundation models.

The predictions written before each run and their full outcomes are in PREREGISTRATION.md.

    DATASETS=AutoClaims MODELS=GBM REPEATS=2 python -u experiments/h3_repair/tail_splice.py
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto

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
N_FIT = env.integer("N_FIT", 1500)
N_CALIB = env.integer("N_CALIB", 500)
N_TEST = env.integer("N_TEST", 1000)
N_EST = env.integer("N_EST", 1)
# One cap, unlike the grid clip_context.py reads from the same variable.
_clip_c = env.text("CLIP_C", "200")
if "," in _clip_c:
    raise SystemExit(f"CLIP_C={_clip_c!r}: tail_splice.py takes one cap, not a grid")
CLIP_C = float(_clip_c)
U_SHARES = env.floats("U_SHARES", [0.8, 0.85, 0.9, 0.95])
ARMS = (["none", "cqr_emp"] + [f"cqr_gpd_{u:g}" for u in U_SHARES]
        + ["cqr_gpd_ks", "cqr_gpd_cv", "cqr_gpd_norm", "cond_gpd"])
SHOW = "cqr_gpd_0.9" if "cqr_gpd_0.9" in ARMS else ARMS[-1]
FACTORS = [1.0, 100.0]
# The clip treatment carries its cap in its name, since `treatment` is in the key.
CLIP_TREATMENT = f"clip_{CLIP_C:g}"
TREATMENTS = {"raw": ("raw", 0.0), CLIP_TREATMENT: ("clip", CLIP_C),
              "rank_gpd": ("rank_gpd", 0.0)}
# 0.5 gives the spread the normalised score and the conditional splice divide by.
LEVELS = [0.5, 0.9, 0.99, 0.999]
I05, I09 = LEVELS.index(0.5), LEVELS.index(0.9)
OUTPUT = env.text("OUTPUT", "tail_splice.csv")

COLUMNS = ["dataset", "model", "repeat", "factor", "treatment", "clip_c", "arm",
           "level", "n_fit", "n_calib", "n_test", "n_est", "sd_shift", "coverage",
           "pinball", "q_median", "u_share", "xi_hat", "scale_hat", "u_score",
           "n_exc", "degenerate", "seconds", "reason"]
KEY = ["dataset", "model", "repeat", "factor", "treatment", "arm", "level",
       "n_fit", "n_calib", "n_test", "n_est"]


def emp_quantile(s, level):
    """The conformal empirical score quantile, and whether the level lies past the data."""
    s = np.sort(np.asarray(s, dtype=float))
    k = int(np.ceil((len(s) + 1) * level))
    return (float(s[-1]), True) if k > len(s) else (float(s[k - 1]), False)


def pwm_fit(exc):
    """GPD shape and scale by probability-weighted moments (Hosking and Wallis, 1987).

    Only for choosing a threshold, which takes hundreds of fits per cell; the reported arms
    keep maximum likelihood, and the two agree to 0.02 in shape on these scores. A
    non-positive denominator gives a negative scale, so it raises.
    """
    x = np.sort(np.asarray(exc, dtype=float))
    p = (np.arange(1, len(x) + 1) - 0.35) / len(x)
    a0, a1 = float(x.mean()), float(np.mean(x * (1 - p)))
    d = a0 - 2 * a1
    if d <= 1e-12:
        raise RuntimeError("degenerate PWM denominator")
    xi, scale = 2.0 - a0 / d, 2 * a0 * a1 / d
    if not (scale > 0) or not np.isfinite(xi):
        raise RuntimeError(f"PWM fit out of range: xi={xi}, scale={scale}")
    return xi, scale


def gpd_quantile(s, level, u_share, fast=False):
    """Score quantile from a GPD above the u_share threshold, with its fit info.

    None when there is nothing to extrapolate: fewer than 20 exceedances, a level inside
    the calibration data, or a failed fit. `fast` uses probability-weighted moments.
    """
    s = np.sort(np.asarray(s, dtype=float))
    u = float(np.quantile(s, u_share))
    exc = s[s > u] - u
    zeta = len(exc) / len(s)
    info = dict(u_share=u_share, u_score=u, n_exc=int(len(exc)), xi_hat=np.nan,
                scale_hat=np.nan)
    if len(exc) < 20 or level <= 1 - zeta:
        return None, info
    try:
        xi, scale = pwm_fit(exc) if fast else genpareto.fit(exc, floc=0.0)[::2]
        r = (1 - level) / zeta
        q = u + scale * np.log(1 / r) if abs(xi) < 1e-8 else u + scale / xi * (r ** (-xi) - 1)
        if not np.isfinite(q):
            return None, info
        info.update(xi_hat=float(xi), scale_hat=float(scale))
        return float(q), info
    except Exception:
        return None, info


def pick_u_ks(s, grid):
    """The threshold whose fitted GPD sits closest to its exceedances in KS distance,
    as Clauset, Shalizi and Newman (2009) choose x_min for a power law."""
    s = np.sort(np.asarray(s, dtype=float))
    best, best_d = grid[len(grid) // 2], np.inf
    for u_share in grid:
        u = float(np.quantile(s, u_share))
        exc = np.sort(s[s > u] - u)
        if len(exc) < 20:
            continue
        try:
            xi, scale = pwm_fit(exc)
        except Exception:
            continue
        emp = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
        d = float(np.max(np.abs(genpareto.cdf(exc, xi, 0.0, scale) - emp)))
        if d < best_d:
            best, best_d = u_share, d
    return best


def pick_u_cv(s, level, grid, folds=5, seed=0):
    """The threshold with the lowest out-of-fold pinball of its extrapolated quantile. The
    scores are exchangeable, so plain K-fold is valid and no model is refitted."""
    s = np.asarray(s, dtype=float)
    idx = np.random.default_rng(seed).permutation(len(s))
    parts = np.array_split(idx, folds)
    best, best_loss = grid[len(grid) // 2], np.inf
    for u_share in grid:
        losses = []
        for p in parts:
            q, _ = gpd_quantile(s[np.setdiff1d(idx, p)], level, u_share, fast=True)
            if q is not None:
                losses.append(metrics.pinball(s[p], q, level))
        if losses and np.mean(losses) < best_loss:
            best, best_loss = u_share, float(np.mean(losses))
    return best


def bounds(q, yc):
    """Per level, every arm as (test quantile, additive correction, degenerate, fit info)."""
    spread_c = np.maximum(q[:N_CALIB, I09] - q[:N_CALIB, I05], 1e-9)
    spread_t = np.maximum(q[N_CALIB:, I09] - q[N_CALIB:, I05], 1e-9)
    # The conditional splice: exceedances over the model's own Q(0.9 | x), divided by
    # the row's spread, pooled into one GPD.
    u_c, u_t = q[:N_CALIB, I09], q[N_CALIB:, I09]
    over = yc > u_c
    zeta_c = float(over.mean())
    cond = None
    if over.sum() >= 20:
        try:
            cxi, cscale = genpareto.fit((yc[over] - u_c[over]) / spread_c[over], floc=0.0)[::2]
            cond = (float(cxi), float(cscale))
        except Exception:
            cond = None

    out = {}
    for i, level in enumerate(LEVELS):
        qc, qt = q[:N_CALIB, i], q[N_CALIB:, i]
        s = yc - qc
        emp, degenerate = emp_quantile(s, level)
        arms = {"none": (qt, 0.0, False, {}), "cqr_emp": (qt, emp, degenerate, {})}
        picked = {"cqr_gpd_ks": pick_u_ks(s, U_SHARES),
                  "cqr_gpd_cv": pick_u_cv(s, level, U_SHARES)}
        for arm, u in [(f"cqr_gpd_{v:g}", v) for v in U_SHARES] + list(picked.items()):
            gq, info = gpd_quantile(s, level, u)
            # A GPD arm that fell back to the empirical quantile is marked degenerate on
            # its own account: it did not do what its name says.
            arms[arm] = (qt, emp, True, info) if gq is None else (qt, gq, False, info)
        if cond is not None and level > 1 - zeta_c:
            cxi, cscale = cond
            ratio = (1 - level) / zeta_c
            step = (cscale * np.log(1 / ratio) if abs(cxi) < 1e-8
                    else cscale / cxi * (ratio ** (-cxi) - 1))
            arms["cond_gpd"] = (u_t + spread_t * step, 0.0, False,
                                dict(xi_hat=cxi, scale_hat=cscale, u_share=1 - zeta_c,
                                     n_exc=int(over.sum())))
        else:
            arms["cond_gpd"] = (qt, 0.0, True, {})
        normalised = s / spread_c
        gq, info = gpd_quantile(normalised, level, 0.9)
        if gq is None:
            arms["cqr_gpd_norm"] = (qt, emp_quantile(normalised, level)[0] * spread_t, True, info)
        else:
            arms["cqr_gpd_norm"] = (qt, gq * spread_t, False, info)
        out[level] = arms
    return out


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    n_ctx = N_FIT + N_CALIB
    for name in DATASETS:
        X, y = datasets.load(name)
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        rng = np.random.default_rng(31337)
        plan = []
        for _ in range(REPEATS):
            idx = rng.permutation(len(y))[:n_ctx + N_TEST]
            visible = np.flatnonzero(leverage.shift_if_corrupted(y[idx[:N_FIT]], FACTORS[-1])
                                     >= leverage.VISIBLE_SHIFT)
            plan.append((idx, int(rng.choice(visible)) if len(visible) else -1))
        for r, (idx, bad) in enumerate(plan):
            if bad < 0:
                print(f"  {name}: #{r} no row at which a unit error is visible", flush=True)
                continue
            Xf, yf_clean = X[idx[:N_FIT]], y[idx[:N_FIT]]
            yc, yte = y[idx[N_FIT:n_ctx]], y[idx[n_ctx:]]
            Xq = np.vstack([X[idx[N_FIT:n_ctx]], X[idx[n_ctx:]]])   # one predict for both
            for model in MODELS:
                for factor in FACTORS:
                    yf = yf_clean.copy()
                    yf[bad] *= factor
                    shift = float(metrics.sd_shift(yf))
                    for treatment, (arm_t, c_t) in TREATMENTS.items():
                        keys = [dict(dataset=name, model=model, repeat=r, factor=factor,
                                     treatment=treatment, arm=a, level=lv, n_fit=N_FIT,
                                     n_calib=N_CALIB, n_test=N_TEST, n_est=N_EST)
                                for a in ARMS for lv in LEVELS]
                        todo = [k for k in keys if append.key(k, KEY) not in done]
                        if not todo:
                            continue
                        t1 = time.time()
                        y_fit, back, _ = transforms.transform(arm_t, c_t, yf)
                        try:
                            q = models.quantiles(model, Xf, y_fit, Xq, seed=7000 + r,
                                                 levels=LEVELS, n_est=N_EST)
                            with np.errstate(over="ignore", invalid="ignore"):
                                q = back(q)
                            err = None
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"[:110]
                        secs = round((time.time() - t1) / max(len(todo), 1), 2)
                        arms = {} if err else bounds(q, yc)
                        for k in todo:
                            row = dict(k, sd_shift=shift, seconds=secs,
                                       clip_c=c_t if arm_t == "clip" else 0.0)
                            if err:
                                row["reason"] = err
                            else:
                                qt, add, degenerate, info = arms[k["level"]][k["arm"]]
                                b = qt + add
                                row.update(info, degenerate=degenerate, reason="",
                                           coverage=float((yte <= b).mean()),
                                           pinball=metrics.pinball(yte, b, k["level"]),
                                           q_median=float(np.median(b)))
                            append.write(OUTPUT, row, COLUMNS)
                        msg = err or "  ".join(
                            f"{lv}: cov {float((yte <= arms[lv]['none'][0]).mean()):.3f}/"
                            f"{float((yte <= arms[lv][SHOW][0] + arms[lv][SHOW][1]).mean()):.3f}"
                            for lv in LEVELS)
                        print(f"  {name[:14]:14s} {model:10s} #{r} f{factor:.0f} {treatment:8s} "
                              f"sd={shift:5.2f}  {msg}", flush=True)
    report()
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    # A degenerate row is the arm's fallback, the bare model or cqr_emp under the arm's
    # name, so it is counted first and then kept out of every median.
    d = d.assign(degenerate=d.degenerate.fillna(False).astype(bool))
    share = d.pivot_table(index=["level", "arm"], values="degenerate", aggfunc="mean").round(3)
    print("\n=== degenerate share by level and arm ===")
    print(share[share.degenerate > 0].to_string())
    d = d[~d.degenerate]
    for col in ("coverage", "pinball"):
        print(f"\n=== {col}, median, degenerate rows dropped ===")
        print(d.pivot_table(index=["level", "factor", "treatment"], columns=["model", "arm"],
                            values=col, aggfunc="median").round(4).to_string())
    g = d[(d.arm == SHOW) & d.xi_hat.notna()]
    if len(g):
        print(f"\nGPD shape on the scores ({SHOW}): median {g.xi_hat.median():.3f}, quartiles "
              f"{g.xi_hat.quantile(0.25):.3f} to {g.xi_hat.quantile(0.75):.3f}")


if __name__ == "__main__":
    main()
