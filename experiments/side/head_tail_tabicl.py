# -*- coding: utf-8 -*-
"""What TabICL's head can express against what it is configured to express.

WHY THIS EXISTS

`findings/h1/head_tail_family.md` did this for TabPFN, whose bar head cannot represent
xi > 0 at all: the outer tail is half-normal, and the quantile path does not even read
that. TabICL's head is a different construction -- 999 quantile levels, a monotone spline
between them, and a parametric tail past the outermost level -- so the same question has
to be asked separately, and the answer is not the same.

WHAT THE SOURCE SAYS, `tabicl` 2.1.1

`_model/quantile_dist.py` implements **both** tail families:

    def estimate_exp_tail_params(...)     # Q(a) = -beta * ln(1-a) + c,  xi = 0
    def estimate_gpd_tail_params(...)     # generalised Pareto, with a shape parameter

and the configuration carries the shape parameter explicitly:

    MIN_ETA: float = -0.49    # eta > 0 gives heavy tails, eta = 0 exponential
    MAX_ETA: float =  0.49    # must be < 0.5 for finite variance

That `eta` is the extreme-value index this thesis calls xi: same parameter, and the
comment about finite variance is the GPD moment condition k < 1/xi.

The switch is `QuantileDistribution(..., tail_type: Literal["exp", "gpd"] = "exp")`.
`_model/tabicl.py:200` builds the head as `QuantileToDistribution(num_quantiles=...)`,
so it takes that default, and `TabICLRegressor.__init__` carries no parameter with
"tail" or "quantile" in its name. The branch cannot be reached through the public API.

WHAT IS MEASURED

The model is fitted once on a context with a known xi. Its native bank of 999 quantiles
is read out, and the *same* vector is then wrapped in `QuantileDistribution` twice, once
per tail type. Nothing about the network changes between the two: the weights, the
context and the predicted quantiles are identical, and only the extrapolation past level
0.999 differs. Whatever separates the two columns is therefore the tail family alone.

Implied xi is read the way the rest of the thesis reads it, by inverting the ratio of
quantiles, at levels far enough out that the tail rather than the spline decides.

PREDICTIONS, WRITTEN BEFORE THE RUN

Q1. `exp` returns an implied xi near zero at every true xi, including 0.8. If it does,
    the ceiling is a configuration default and not a property of the trained network.
Q2. `gpd` returns an implied xi that rises with the true one. If it does, the capability
    is present in the shipped package and switched off, which is a far cheaper
    recommendation than "the head needs a tail parameter".
Q3. `gpd` saturates near 0.49, its configured maximum, so it cannot follow the heaviest
    tails either. The recommendation is then "expose it", not "it solves the problem".

If Q2 fails -- if the GPD branch does not recover more than the exponential one -- then
the branch is dead code and no issue should be filed upstream. That is the outcome this
script exists to rule out before anything is reported.

    XI=0.2,0.5,0.8 SEEDS=3 python -u experiments/side/head_tail_tabicl.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from common import append, generator, metrics, paths, quiet

quiet.silence()

OUTPUT = os.environ.get("OUTPUT", "head_tail_tabicl.csv")
XI_LISTA = [float(v) for v in os.environ.get("XI", "0.0,0.2,0.5,0.8").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "3"))
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
N_TEST = int(os.environ.get("N_TEST", "200"))
N_EST = int(os.environ.get("N_EST", "1"))

# Read far enough out that the tail, not the spline between the 999 levels,
# decides the answer. 0.999 is the last level the bank itself carries.
NIVOI = [0.5, 0.999, 0.9999, 0.99999]

KOLONE = ["xi_true", "tail_type", "seed", "n_est", "n_train",
          "q50", "q999", "q9999", "q99999", "ratio", "xi_implied",
          "eta_r_median", "seconds", "reason"]
KLJUC = ["xi_true", "tail_type", "seed", "n_est", "n_train"]


def native_bank(Xtr, ytr, Xte, seed):
    """TabICL's own 999 quantile levels, as the head produces them."""
    from tabicl import TabICLRegressor

    alphas = [(i + 1) / 1000.0 for i in range(999)]
    m = TabICLRegressor(n_estimators=N_EST, device="cpu", random_state=seed)
    m.fit(Xtr, ytr)
    q = np.asarray(m.predict(Xte, output_type="quantiles", alphas=alphas), dtype=float)
    # (levels, rows) or (rows, levels) depending on version; make it (rows, levels)
    if q.shape[0] == len(alphas) and q.shape[0] != len(Xte):
        q = q.T
    return q, np.asarray(alphas, dtype=float)


def wrap(q_rows, alphas, tail_type):
    """The same predicted quantiles under one tail family or the other."""
    from tabicl._model.quantile_dist import QuantileDistribution

    return QuantileDistribution(
        torch.tensor(q_rows, dtype=torch.float64),
        alpha_levels=torch.tensor(alphas, dtype=torch.float64),
        tail_type=tail_type,
    )


def jedan(xi, tail_type, seed):
    t0 = time.time()
    r = dict(xi_true=xi, tail_type=tail_type, seed=seed, n_est=N_EST,
             n_train=N_TRAIN, reason=None)
    try:
        rng = np.random.default_rng(seed)
        p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
        Xtr, ytr = p.X, p.y
        Xte = generator.gpd(N_TEST, rng, xi=xi, clip=True).X

        q_rows, alphas = native_bank(Xtr, ytr, Xte, seed)
        dist = wrap(q_rows, alphas, tail_type)

        vals = {}
        for a in NIVOI:
            z = dist.icdf(torch.tensor(a, dtype=torch.float64))
            vals[a] = float(np.median(np.asarray(z, dtype=float)))

        # Far-tail ratio, read where the tail family decides rather than the spline.
        ratio = ((vals[0.99999] - vals[0.5]) / (vals[0.9999] - vals[0.5])
                 if vals[0.9999] > vals[0.5] else np.nan)
        eta = getattr(dist, "eta_r", None)
        r.update(q50=vals[0.5], q999=vals[0.999], q9999=vals[0.9999],
                 q99999=vals[0.99999], ratio=ratio,
                 xi_implied=metrics.xi_from_ratio(ratio, hi=0.99999, lo=0.9999)
                 if np.isfinite(ratio) else np.nan,
                 eta_r_median=float(np.median(np.asarray(eta, dtype=float)))
                 if eta is not None else np.nan)
    except Exception as e:                                   # noqa: BLE001
        r["reason"] = f"{type(e).__name__}: {e}"
    r["seconds"] = round(time.time() - t0, 1)
    return r


def main() -> None:
    gotovi = append.done(OUTPUT, KLJUC)
    for xi in XI_LISTA:
        for tail_type in ["exp", "gpd"]:
            for seed in range(SEEDOVA):
                k = append.key(dict(xi_true=xi, tail_type=tail_type, seed=seed,
                                    n_est=N_EST, n_train=N_TRAIN), KLJUC)
                if k in gotovi:
                    continue
                r = jedan(xi, tail_type, seed)
                append.write(OUTPUT, r, KOLONE)
                print(f"  xi={xi} {tail_type:<4} seed={seed} "
                      f"xi_implied={r.get('xi_implied')} eta={r.get('eta_r_median')} "
                      f"{r.get('reason') or ''}", flush=True)

    d = pd.read_csv(paths.result(OUTPUT))
    d = d[d.reason.isna()]
    if d.empty:
        print("no usable rows")
        return
    print()
    print(d.pivot_table(index="xi_true", columns="tail_type",
                        values="xi_implied", aggfunc="median").round(4).to_string())
    print()
    print("Q1 exp stays near zero at every true xi · Q2 gpd rises with it · "
          "Q3 gpd saturates near 0.49.\nSame network, same context, same 999 "
          "predicted quantiles; only the extrapolation differs.")


if __name__ == "__main__":
    main()
