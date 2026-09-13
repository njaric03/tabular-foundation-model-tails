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

SECOND RUN: THE THRESHOLD AS A GRID AND AS A RULE
-------------------------------------------------
The 90th percentile was picked as freely as C = 200 was, so it is now measured
over 0.80, 0.85, 0.90 and 0.95, and chosen by two rules from the scores
themselves: `ks` by the Kolmogorov-Smirnov distance of the fitted tail, `cv` by
out-of-fold pinball of the extrapolated score quantile. Predictions:
  T5. The conclusion does not depend on the threshold: coverage at 0.99 across
      the four fixed thresholds stays within 0.003 of itself.
  T6. Neither rule beats the fixed 0.90 in coverage by more than that. If one
      does, the fixed choice goes and the rule takes its place.
  T7. The two rules disagree with each other more often than either disagrees
      with 0.90, because the KS distance judges the fit of the whole tail while
      the cross-validated loss judges one quantile of it.

THIRD RUN: SHARPNESS, AND WHETHER THE TWO REPAIRS COMPOSE
--------------------------------------------------------
Two additions, both answers to a literature review of the second run.

The review proposed switching to the empirical conformal quantile at 0.99, where
it is not degenerate, on the assumption that it is sharper there. On 1467 cells
where every arm is present it is not: pinball at 0.99 is 414.9 for cqr_emp
against 394.2 to 394.6 for every GPD variant and 405.0 for no patch at all. The
empirical bound is the widest of the three, because at n = 500 the 496th order
statistic of a heavy-tailed score sample is itself an extreme. So the switch is
not made, and the sharpness question is attacked where it belongs: the score.

    cqr_gpd_norm   the score divided by the model's own spread for that row,
                   Q(0.9 | x) - Q(0.5 | x), so
                   the bound scales with each row's spread instead of being one
                   additive constant for all of them. The standard locally
                   adaptive conformal construction, applied to the GPD
                   extrapolation.

    treatment rank_gpd   the third context treatment: the context target becomes
                   its own normal scores and the prediction is mapped back
                   through the spliced empirical-Pareto inverse. The review
                   argued that this reduces to the output patch with a different
                   body model. It does not: the rank transform removes the
                   leverage before the network sees it (borders inside the data
                   3547 against 421), while the patch leaves the context
                   contaminated and repairs the output. Crossing the two settles
                   whether they compose or are redundant.

Predictions:
  T8. cqr_gpd_norm lowers pinball at 0.99 against cqr_gpd_0.9 at coverage no
      worse than nominal, on the datasets whose predicted spread varies most
      across rows.
  T9. On the rank_gpd context the patch adds less than it adds on the raw
      context, because the leverage is already gone; if instead it adds the same
      amount, the two repairs are independent and should be reported as a pair.

FOURTH RUN: THE SPLICE POINT ITSELF CONDITIONAL
-----------------------------------------------
Every arm above splices at one number: a quantile of the pooled scores. The
conditional version moves the splice with the row. `cond_gpd` takes the model's
own Q(0.9 | x) as the threshold, divides the exceedances over it by that row's
spread Q(0.9 | x) - Q(0.5 | x), pools those, fits a GPD, and extrapolates per
row:

    Q(a | x) = Q(0.9 | x) + spread(x) * (sigma / xi) * (((1 - a) / zeta) ** -xi - 1)

with zeta the measured exceedance rate of the model's own Q(0.9 | x) on the
calibration rows, which also corrects for that quantile being miscalibrated. This
is the classical conditional peaks-over-threshold construction with an
intermediate conditional quantile, the same shape as gradient boosting for
extreme quantile regression and extremal random forests, except that the
intermediate quantile comes from the model under test rather than from a purpose
built quantile regressor. Run with GBM as that model it is a baseline in the
spirit of gbex, built from parts already here; it is not the published
implementation and is not claimed to be.

It differs from the conformal arms in what it gives up: there is no exchangeability
argument behind it, so no coverage guarantee, only an EVT extrapolation. That is
the trade to report.

Predictions:
  T10. cond_gpd is sharper at 0.99 than cqr_gpd_0.9, because the bound moves with
       the row instead of adding one constant to every row.
  T11. Its coverage at 0.999 is closer to nominal than the model alone but less
       reliable than the conformal arms, since it has no finite-sample guarantee.
  T12. Under the unit error it degrades less than the additive arms, because the
       threshold it splices at is itself re-estimated per row.

OUTCOME OF THE SECOND, THIRD AND FOURTH RUNS, 12.9.2026
-------------------------------------------------------
50400 rows, none failed. Every comparison is paired on the same cell, and the
unit is a dataset-model pair, 21 of them. That matters here: a pooled median over
arms measured on different cell sets suggested a 3% gap between thresholds that
the paired test says is 0.06%.

  T5 HELD, strongly. At 0.99 the four fixed thresholds differ by at most 0.06% of
     pinball, paired, and coverage is 0.990 to 0.991 for all of them.
  T6 HELD. Neither rule beats the fixed 0.90: the KS rule is better in 11 of 21
     units (p = 1.0, median -0.01%) and the cross-validated rule in 11 of 21
     (median -0.06%). They cost nothing either, so a rule can replace the hand
     choice for free, which is the useful form of this result.
  T7 STRONGER than predicted. The rules disagree about the threshold, often: KS
     picks 0.80 in 49% of cells and 0.95 in 12%, cross-validation picks 0.90 in
     35% and 0.80 in 25%. The outcome does not move. The threshold is not a knob
     worth arguing about at this calibration size.
  T8 FAILED. Dividing the score by the model's own spread is not sharper: at 0.99
     it is better in 7 of 21 units on a clean context (median +0.5%, worst +25%),
     and at 0.90 it is significantly worse (5 of 21 under the unit error, two
     sided p = 0.027; 4 of 21 on a clean context). Locally adaptive scores buy
     nothing here, and add tail risk.
     What the 0.90 row actually compares, which the arm names hide. At a
     threshold of 0.9 the level 0.90 sits inside the calibration data, so
     `gpd_quantile` returns None for BOTH arms and both fall back -- by design,
     not by failure: 100% of rows at that level carry no fit. So that line is
     locally adaptive EMPIRICAL conformal against plain empirical conformal, with
     no Pareto tail on either side. It is still the textbook test of the locally
     adaptive score, and the verdict stands; it is simply not a statement about
     the GPD. Only the 0.99 row, where both arms really fit, is. A re-run cannot
     move the 0.90 number, and `cqr_gpd_norm` is fixed at a 0.9 threshold, so the
     locally adaptive score is never measured against a live tail at that level.
     Doing that needs a `cqr_gpd_norm_0.8` arm, the way cqr_gpd_0.8 and 0.85 do
     fit at 0.90; it is not measured here.
  T9 NOT HELD, and that settles an argument. The patch adds about as much on a
     rank-transformed context as on a raw one (12 of 21 units better at 0.99,
     median -1.6%, and 12 of 21 at 0.999), so the rank transform is not the patch
     with a different body model: the two act on different objects and compose.
     The rank context is also better on its own at 0.99 (15 of 21, -2.8%) and
     unusable at 0.999 (worst +175232%), which is the instability the fifth run of
     `clip_context.py` addresses.
  T10 and T12 FAILED. The conditional splice is worse than the additive patch:
     at 0.99 it is better in 5 of 21 units on a clean context (p = 0.027, median
     +14.4%) and at 0.999 under the error in 3 of 21 (p = 0.001, +13.5%), with
     worst cells of +72800% and +1.7e8%. Splicing per row multiplies a pooled
     shape by a per-row spread, and where either is large the product is not a
     quantile any more. Conditionality is not free.
  T11 HELD for coverage only. cond_gpd covers 0.990 and 0.991 at 0.99 and 0.998
     to 0.999 at 0.999, as close to nominal as the conformal arms, while being
     the least sharp of them. Coverage alone does not rank these methods.

So the output side keeps exactly one recommendation, the plain conformal patch
with a GPD on the scores, and its threshold can be chosen by a rule instead of by
hand. The two attempts to improve it, locally adaptive scores and a conditional
splice point, both failed, and both are reported.

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
# One value, not the grid `clip_context.py` reads out of the same variable name.
# `CLIP_C=20,50,200` is the documented invocation there and used to die here on an
# opaque `could not convert string to float`, four frames inside an import.
_clip_c = os.environ.get("CLIP_C", "200")
if "," in _clip_c:
    raise SystemExit(
        f"CLIP_C={_clip_c!r}: this script takes ONE cap, not the grid "
        f"clip_context.py takes. The clip treatment is a single control here.")
CLIP_C = float(_clip_c)
# Where the GPD starts, as a share of the calibration scores. The fixed 0.9 of the
# first run is as arbitrary as C = 200 was, so the whole grid is measured as a
# sensitivity, and two rules pick it from the scores themselves: `ks` minimises the
# Kolmogorov-Smirnov distance of the fitted tail, the way Clauset, Shalizi and
# Newman (2009) pick x_min for a power law, and `cv` minimises out-of-fold pinball
# of the extrapolated score quantile at the level being asked for. The EVT
# literature has automatic choices of its own, ordered goodness-of-fit tests with a
# false discovery rate (Bader, Yan and Zhang, Ann. Appl. Stat. 2018), score tests
# (Northrop and Coleman, Extremes 2014) and the double bootstrap (Danielsson et
# al., J. Multivar. Anal. 2001); these two are the cheap end of that family, and
# both are computable from 500 scores without refitting the model.
U_SHARES = [float(v) for v in os.environ.get("U_SHARES", "0.8,0.85,0.9,0.95").split(",")]
ARMS = (["none", "cqr_emp"] + [f"cqr_gpd_{u:g}" for u in U_SHARES]
        + ["cqr_gpd_ks", "cqr_gpd_cv", "cqr_gpd_norm", "cond_gpd"])
SHOW = "cqr_gpd_0.9" if "cqr_gpd_0.9" in ARMS else ARMS[-1]
FACTORS = [1.0, 100.0]
# The clip treatment is named after the cap it actually ran at. It was hard-coded
# `clip_200` while the cap came from CLIP_C, so a run at CLIP_C=50 wrote rows
# labelled clip_200 -- and since `treatment` is in KEY and `clip_c` was in neither
# KEY nor COLUMNS, a later resume at a different cap found "already done" and
# skipped the work. That is rule 2's silent skip. The label carries the value now,
# so the two cannot come apart, and `clip_c` is written alongside it.
CLIP_TREATMENT = f"clip_{CLIP_C:g}"
TREATMENTS = ["raw", CLIP_TREATMENT, "rank_gpd"]
# What each treatment does to the context, in `clip_context.transform` terms. The
# back-transform matters for the third one: its quantiles come out in normal-score
# space and are mapped back through the context's own spliced inverse.
TREATMENT_ARMS = {"raw": ("raw", 0.0), CLIP_TREATMENT: ("clip", CLIP_C),
                  "rank_gpd": ("rank_gpd", 0.0)}
# 0.5 is here for the spread that the locally adaptive score and the conditional
# splice divide by, Q(0.9 | x) - Q(0.5 | x). It is also measured in its own right,
# which is how a repair that moves the whole distribution gets caught.
LEVELS = [0.5, 0.9, 0.99, 0.999]
I05, I09 = LEVELS.index(0.5), LEVELS.index(0.9)
OUT = os.environ.get("OUTPUT", "tail_splice.csv")

COLUMNS = ["dataset", "model", "repeat", "factor", "treatment", "clip_c", "arm",
           "level", "n_fit", "n_calib", "n_test", "n_est", "sd_shift", "coverage",
           "pinball", "q_median", "u_share", "xi_hat", "scale_hat", "u_score",
           "n_exc", "degenerate", "seconds", "reason"]
KEY = ["dataset", "model", "repeat", "factor", "treatment", "arm", "level",
       "n_fit", "n_calib", "n_test", "n_est"]


def emp_quantile(s, level):
    """The conformal empirical score quantile, and whether the level is past it."""
    s = np.sort(np.asarray(s, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * level))
    return (float(s[-1]), True) if k > n else (float(s[k - 1]), False)


def pwm_fit(exc):
    """Probability-weighted moments for the GPD, in closed form.

    Hosking and Wallis (1987). Used only where a threshold is being chosen, which
    needs hundreds of fits per cell: `genpareto.fit` costs about 0.3 s each and
    made the threshold rules five times more expensive than the model fit they
    were selecting for. The reported arms keep maximum likelihood, and the two
    agree to 0.02 in shape on these scores.
    """
    x = np.sort(np.asarray(exc, dtype=float))
    n = len(x)
    p = (np.arange(1, n + 1) - 0.35) / n
    a0 = float(x.mean())
    a1 = float(np.mean(x * (1 - p)))
    d = a0 - 2 * a1
    # d <= 0, not just |d| ~ 0. A negative denominator gives a NEGATIVE scale, and
    # `gpd_quantile` then returns u + scale/xi * (r**-xi - 1) < u for r < 1 and
    # xi > 0 -- a score quantile below its own threshold. That only has to be
    # finite to pass the check there, so it would be handed to `pick_u_ks` (which
    # feeds it straight to genpareto.cdf) and `pick_u_cv` as a legitimate fit.
    if d <= 1e-12:
        raise RuntimeError("degenerate PWM denominator")
    xi, scale = 2.0 - a0 / d, 2 * a0 * a1 / d
    if not (scale > 0) or not np.isfinite(xi):
        raise RuntimeError(f"PWM fit out of range: xi={xi}, scale={scale}")
    return xi, scale


def gpd_quantile(s, level, u_share, fast=False):
    """Score quantile from a GPD fitted above the u_share threshold.

    Returns None when there is nothing to extrapolate: fewer than 20 exceedances,
    a level already inside the calibration data, or a fit that does not converge.
    `fast` swaps maximum likelihood for probability-weighted moments.
    """
    s = np.sort(np.asarray(s, dtype=float))
    n = len(s)
    u = float(np.quantile(s, u_share))
    exc = s[s > u] - u
    zeta = len(exc) / n
    info = dict(u_share=u_share, u_score=u, n_exc=int(len(exc)), xi_hat=np.nan,
                scale_hat=np.nan)
    if len(exc) < 20 or level <= 1 - zeta:
        return None, info
    try:
        xi, scale = pwm_fit(exc) if fast else genpareto.fit(exc, floc=0.0)[::2]
        r = (1 - level) / zeta
        q = (u + scale * np.log(1 / r) if abs(xi) < 1e-8
             else u + scale / xi * (r ** (-xi) - 1))
        if not np.isfinite(q):
            return None, info
        info.update(xi_hat=float(xi), scale_hat=float(scale))
        return float(q), info
    except Exception:
        return None, info


def pick_u_ks(s, grid):
    """The threshold whose fitted GPD sits closest to its exceedances in KS distance."""
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
    """The threshold with the lowest out-of-fold pinball for the quantile it
    extrapolates. The scores are exchangeable, so plain K-fold is valid on them,
    and no model is refitted: this costs GPD fits, not TabPFN fits."""
    s = np.asarray(s, dtype=float)
    idx = np.random.default_rng(seed).permutation(len(s))
    parts = np.array_split(idx, folds)
    best, best_loss = grid[len(grid) // 2], np.inf
    for u_share in grid:
        loss, used = 0.0, 0
        for p in parts:
            keep = np.setdiff1d(idx, p)
            q, _ = gpd_quantile(s[keep], level, u_share, fast=True)
            if q is None:
                continue
            d = s[p] - q
            loss += float(np.mean(np.maximum(level * d, (level - 1) * d)))
            used += 1
        if used and loss / used < best_loss:
            best, best_loss = u_share, loss / used
    return best


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
                                for a in ARMS for lv in LEVELS]
                        todo = [k for k in keys if append.key(k, KEY) not in done]
                        if not todo:
                            continue
                        # Loud, because a shadowed loop variable once turned this
                        # into a float and TabICL accepted it silently.
                        assert float(r).is_integer(), f"repeat is not an integer: {r!r}"
                        t1 = time.time()
                        arm_t, c_t = TREATMENT_ARMS[treatment]
                        y_fit, back, _ = CC.transform(arm_t, c_t, yf)
                        try:
                            q = models.quantiles(model, Xf, y_fit, Xq, seed=7000 + r,
                                                 levels=LEVELS, n_est=N_EST)
                            with np.errstate(over="ignore", invalid="ignore"):
                                q = back(q)
                            err = None
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"[:110]
                        secs = round((time.time() - t1) / max(len(todo), 1), 2)
                        bounds = {}
                        if not err:
                            # The scale a locally adaptive score divides by: the
                            # model's own Q(0.9), which is positive here and carries
                            # its per-row spread. Scores become dimensionless, so a
                            # row with a wide distribution no longer sets the bound
                            # for a row with a narrow one.
                            sc_c = np.maximum(q[:N_CALIB, I09] - q[:N_CALIB, I05], 1e-9)
                            sc_t = np.maximum(q[N_CALIB:, I09] - q[N_CALIB:, I05], 1e-9)
                            # The conditional splice. The threshold is not a
                            # quantile of the pooled scores but the model's own
                            # Q(0.9 | x), so it moves with the row; the exceedances
                            # over it are pooled only after being divided by that
                            # row's spread. This is the classical conditional POT
                            # construction with the intermediate quantile supplied
                            # by the model under test, and with GBM as that model it
                            # is a baseline in the spirit of gbex.
                            u_c, u_t = q[:N_CALIB, I09], q[N_CALIB:, I09]
                            over = yc > u_c
                            zeta_c = float(over.mean())
                            cond = None
                            if over.sum() >= 20:
                                try:
                                    cxi, cscale = genpareto.fit(
                                        (yc[over] - u_c[over]) / sc_c[over], floc=0.0)[::2]
                                    cond = (float(cxi), float(cscale))
                                except Exception:
                                    cond = None
                            for i, lv in enumerate(LEVELS):
                                qc, qt = q[:N_CALIB, i], q[N_CALIB:, i]
                                s = yc - qc
                                emp, degen = emp_quantile(s, lv)
                                arms = {"none": (qt, 0.0, False, {}),
                                        "cqr_emp": (qt, emp, degen, {})}
                                picked = {"cqr_gpd_ks": pick_u_ks(s, U_SHARES),
                                          "cqr_gpd_cv": pick_u_cv(s, lv, U_SHARES)}
                                for arm, u in ([(f"cqr_gpd_{v:g}", v) for v in U_SHARES]
                                               + list(picked.items())):
                                    gq, info = gpd_quantile(s, lv, u)
                                    # `True`, not `degen`: a GPD arm that fell back
                                    # to the empirical quantile did not do what its
                                    # name says, and that is a degeneracy of its own
                                    # -- separate from the empirical quantile being
                                    # out of range, which is what `degen` records.
                                    # Recording the fallback as False made 12600 rows
                                    # of the last run byte-identical copies of
                                    # cqr_emp under seven arm names, all of them
                                    # flagged as genuine fits.
                                    arms[arm] = ((qt, emp, True, info) if gq is None
                                                 else (qt, gq, False, info))
                                if cond is not None and lv > 1 - zeta_c:
                                    cxi, cscale = cond
                                    # Not `r`: that is the repeat index of the loop
                                    # this sits inside, and shadowing it sent a
                                    # float seed into every later fit of the run.
                                    ratio = (1 - lv) / zeta_c
                                    step = (cscale * np.log(1 / ratio) if abs(cxi) < 1e-8
                                            else cscale / cxi * (ratio ** (-cxi) - 1))
                                    arms["cond_gpd"] = (u_t + sc_t * step, 0.0, False,
                                                        dict(xi_hat=cxi, scale_hat=cscale,
                                                             u_share=1 - zeta_c,
                                                             n_exc=int(over.sum())))
                                else:
                                    arms["cond_gpd"] = (qt, 0.0, True, {})
                                sn = (yc - qc) / sc_c
                                gq, info = gpd_quantile(sn, lv, 0.9)
                                if gq is None:
                                    e2, _ = emp_quantile(sn, lv)   # fallback: degenerate
                                    arms["cqr_gpd_norm"] = (qt, e2 * sc_t, True, info)
                                else:
                                    arms["cqr_gpd_norm"] = (qt, gq * sc_t, False, info)
                                bounds[lv] = arms
                        for k in todo:
                            row = dict(k, sd_shift=shift, seconds=secs,
                                       clip_c=c_t if arm_t == "clip" else 0.0)
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
                                f"->{(bounds[lv][SHOW][0] + bounds[lv][SHOW][1]).mean():.0f}"
                                f" cov {float((yte <= bounds[lv]['none'][0]).mean()):.3f}"
                                f"/{float((yte <= bounds[lv][SHOW][0] + bounds[lv][SHOW][1]).mean()):.3f}"
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

    # A degenerate row is the arm's fallback, not the arm: cqr_emp past the reach
    # of 500 calibration scores, a cqr_gpd_* whose GPD declined to fit, cond_gpd
    # below its own conditional threshold. Those rows are the BARE MODEL wearing
    # the arm's name, and pooling them into the arm's median compares a method
    # against a mixture of itself and no method at all. The share is not small:
    # in the run of 12.9.2026 cond_gpd was degenerate in 1260 of 1260 cells at
    # level 0.5 and 331 of 1260 at 0.9. Printed first, so that a table below with
    # a thin denominator is visible as such, then excluded from every median.
    deg = d.assign(degenerate=d.degenerate.fillna(False).astype(bool))
    share = deg.pivot_table(index=["level", "arm"], values="degenerate",
                            aggfunc="mean").round(3)
    share = share[share.degenerate > 0]
    print("\n=== degenerate share (arm fell back to the bare model or to cqr_emp) ===")
    print(share.to_string() if len(share) else "  none")
    d = deg[~deg.degenerate]
    if d.empty:
        return
    for col in ("coverage", "pinball"):
        print(f"\n=== {col}, median over datasets and repeats, degenerate rows dropped ===")
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
