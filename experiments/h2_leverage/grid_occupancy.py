# -*- coding: utf-8 -*-
"""Where the predictive mass sits on the bar grid, under a dosed leverage row.

WHY THIS EXISTS

`findings/h1/head_tail_family.md` established, from the source of `tabpfn` 8.4.0, three
properties of the output head: its outer tail is half-normal and its scale is not learned;
`icdf`, which `regressor.py` calls for quantiles, is not overridden and therefore never
reads that tail; and the border grid holds 341 borders per unit in the body but three
between 50 and 128 standard deviations, so above about 17 the whole tail is five bars of
22 sd each. Placing mass in the last bar by hand then moved the reported Q(0.999) from
3.29 to 105.78 on a half-per-mille change.

All of that is a statement about what the head *can* do. It says nothing about whether a
fitted TabPFN on real data ever reaches that regime, and that is the difference between an
architectural curiosity and an explanation of the measurements this repository already has.
This script asks the empirical half, on the treatment of `influence.py` in `sd` mode.

WHAT IT IS TRYING TO EXPLAIN

`findings/NALAZI.md` section 2.6: dosed on the sd shift itself, at a true xi = 0.7, TabPFN's
implied xi goes 0.576 / 0.515 / 0.356 / -0.173 / -0.189 across shifts 2 / 4 / 10 / 20 / 50.
GBM is flat to the third decimal at every dose. At a shift of 20 -- which `freMTPL2sev`, the
worst of 99 public datasets, actually carries at 19.9 -- TabPFN's predictive distribution
stops being heavy-tailed and reports as bounded. The finding is currently recorded as a
property of the model, with no mechanism attached to it.

The candidate mechanism is arithmetic. The raw-space grid is
`borders * y_train_std_ + y_train_mean_`, so a row that inflates the standard deviation by
20x stretches the whole grid by 20x while the data stays where it was. The data then
occupies a small number of bars, and the inherited `icdf` interpolates *linearly* inside a
bar. Linear interpolation inside one bar is locally uniform, and a uniform tail is exactly
what a negative xi reports. If Q(0.9) and Q(0.99) end up inside the same bar, the ratio
between them carries no model belief at all -- only the geometry of that bar.

PREDICTIONS, WRITTEN BEFORE THE RUN

P1. `borders_in_data`, the number of borders falling inside the training range, collapses
    roughly in proportion to the sd shift. This one is nearly arithmetic and is here as a
    sanity check on the instrument rather than as a finding.

P2. `same_bar_90_99`, the share of test rows whose Q(0.9) and Q(0.99) land in one bar,
    is near zero in the clean context and rises sharply by shift 20. If it does, section
    2.6's negative xi is bar geometry rather than a belief the model holds, and the two
    parts of the thesis are joined by a measurement instead of by an expression.

P3. `mass_beyond_inner`, the probability mass at or past `borders[-2]`, stays negligible
    at every dose. The half-normal branch would then be latent: real, unreachable here, and
    honestly reported as an architectural property that this treatment does not trigger.
    P2 and P3 are independent; P2 can hold while P3 fails, and the write-up must say which.

If P2 fails, the head finding stays architectural and section 2.6 keeps looking for a
mechanism. That outcome is worth the hour too, and is the reason the predictions are here
rather than written afterwards.

RUNNING

    MODELS=TabPFN-V3 XI=0.7 SD_SHIFTS=1,2,4,10,20,50 SEEDS=5 \
        python -u experiments/h2_leverage/grid_occupancy.py

Only models with a bar grid can be measured: TabPFN exposes it through
`predict(output_type="full")`, which returns the raw-space criterion and the logits.
TabICL and EXAONE have a 999-quantile head and no grid, so they are refused by name rather
than silently producing empty columns.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

XI_LISTA = [float(v) for v in os.environ.get("XI", "0.7").split(",")]
# A shift of 1.0 means the row is added at a value that moves the sd by nothing
# measurable, and stands in for the clean context on the same code path.
SD_SHIFTS = [float(v) for v in os.environ.get("SD_SHIFTS", "1,2,4,10,20,50").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabPFN-V3"))
N_EST = int(os.environ.get("N_EST", "1"))
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
N_TEST = int(os.environ.get("N_TEST", "900"))
OUT = os.environ.get("OUTPUT", "grid_occupancy.csv")

NIVOI = [0.5, 0.9, 0.99, 0.999]
GRID_MODELS = {"TabPFN-V3", "TabPFN-v2.5", "TabPFN-v2.6"}

KOLONE = ["xi", "model", "sd_shift_target", "sd_shift", "seed", "n_est", "n_train",
          "n_bars", "border_last", "y_std_clean", "y_std_with",
          "borders_in_data", "bars_in_data_share",
          "same_bar_90_99", "same_bar_99_999", "mass_beyond_inner",
          "median_bar_width_at_q99_sd", "q999_in_outer_five",
          "xi_implied", "seconds", "reason"]
KLJUC = ["xi", "model", "sd_shift_target", "seed", "n_est", "n_train"]


def _full(name, Xtr, ytr, Xte, seed):
    """Fitted TabPFN's raw-space borders and per-row bar probabilities."""
    from tabpfn import TabPFNRegressor

    kw = dict(n_estimators=N_EST, device="cpu", random_state=seed,
              ignore_pretraining_limits=True)
    if name in models.TABPFN_PATHS:
        kw["model_path"] = models.TABPFN_PATHS[name]
    m = TabPFNRegressor(**kw)
    m.fit(Xtr, ytr)
    # The levels MUST be passed: without them `predict` returns its own default
    # [0.1 .. 0.9], and reading column 1 and 2 out of that silently measures
    # Q(0.2) and Q(0.3). That produced a full, plausible-looking table on the
    # first run of this script, and it disagreed with section 2.6 in sign.
    out = m.predict(Xte, output_type="full", quantiles=list(NIVOI))

    import torch

    borders = out["criterion"].borders.detach().double().numpy()
    probs = torch.softmax(out["logits"].double(), -1).detach().numpy()
    q = np.stack([np.asarray(a, dtype=float) for a in out["quantiles"]], axis=1)
    return borders, probs, q, float(m.y_train_std_)


def _bar_of(borders: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of the bar each value falls in, clipped to the grid."""
    return np.clip(np.searchsorted(borders, values) - 1, 0, len(borders) - 2)


def jedan(xi, ime, cilj, seed):
    t0 = time.time()
    r = dict(xi=xi, model=ime, sd_shift_target=cilj, seed=seed, n_est=N_EST,
             n_train=N_TRAIN, reason=None)
    if ime not in GRID_MODELS:
        r["reason"] = f"{ime} has no bar grid to occupy"
        r["seconds"] = round(time.time() - t0, 1)
        return r

    rng = np.random.default_rng(seed)
    p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = p.X, p.y
    Xte = generator.gpd(N_TEST, rng, xi=xi, clip=True).X

    if cilj <= 1.0:
        Xc, yc = Xtr, ytr                      # clean context, same code path
    else:
        y0 = metrics.y0_for_sd_shift(ytr, cilj)
        if not np.isfinite(y0):
            r["reason"] = f"sd shift {cilj} unreachable on this seed"
            r["seconds"] = round(time.time() - t0, 1)
            return r
        Xc = np.vstack([Xtr, np.zeros((1, Xtr.shape[1]))])
        yc = np.concatenate([ytr, np.array([y0])])

    try:
        borders, probs, q, std_with = _full(ime, Xc, yc, Xte, seed)
    except Exception as e:                                   # noqa: BLE001
        r["reason"] = f"{type(e).__name__}: {e}"
        r["seconds"] = round(time.time() - t0, 1)
        return r

    lo, hi = float(ytr.min()), float(ytr.max())
    inside = int(((borders >= lo) & (borders <= hi)).sum())

    b90 = _bar_of(borders, q[:, NIVOI.index(0.9)])
    b99 = _bar_of(borders, q[:, NIVOI.index(0.99)])
    b999 = _bar_of(borders, q[:, NIVOI.index(0.999)])

    widths = np.diff(borders)
    std_clean = float(np.std(ytr))

    num, den = q[:, NIVOI.index(0.99)], q[:, NIVOI.index(0.9)]
    ok = den > 1e-9
    xi_imp = (metrics.xi_from_ratio(float(np.median(num[ok] / den[ok])))
              if ok.any() else np.nan)

    r.update(
        sd_shift=float(metrics.sd_shift(yc)) if cilj > 1.0 else 1.0,
        n_bars=len(borders) - 1,
        border_last=float(borders[-1]),
        y_std_clean=std_clean,
        y_std_with=std_with,
        borders_in_data=inside,
        bars_in_data_share=inside / (len(borders) - 1),
        # The signature the mechanism predicts: two levels inside one bar means
        # the ratio between them is that bar's geometry, not the model's belief.
        same_bar_90_99=float((b90 == b99).mean()),
        same_bar_99_999=float((b99 == b999).mean()),
        # The half-normal branch lives at or past borders[-2]; icdf never reads it.
        mass_beyond_inner=float(probs[:, -1].mean()),
        median_bar_width_at_q99_sd=float(np.median(widths[b99]) / std_clean),
        q999_in_outer_five=float((b999 >= len(borders) - 6).mean()),
        xi_implied=xi_imp,
    )
    r["seconds"] = round(time.time() - t0, 1)
    return r


def main() -> None:
    gotovi = append.done(OUT, KLJUC)
    for xi in XI_LISTA:
        for ime in MODELI:
            for cilj in SD_SHIFTS:
                for seed in range(SEEDOVA):
                    k = append.key(dict(xi=xi, model=ime, sd_shift_target=cilj,
                                        seed=seed, n_est=N_EST, n_train=N_TRAIN), KLJUC)
                    if k in gotovi:
                        continue
                    r = jedan(xi, ime, cilj, seed)
                    append.write(OUT, r, KOLONE)
                    print(f"  xi={xi} {ime} shift={cilj:<5} seed={seed} "
                          f"same_bar={r.get('same_bar_90_99')} "
                          f"xi_imp={r.get('xi_implied')}", flush=True)

    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna()]
    if d.empty:
        print("no usable rows")
        return

    print()
    print(d.groupby(["model", "sd_shift_target"])[
        ["borders_in_data", "same_bar_90_99", "same_bar_99_999",
         "mass_beyond_inner", "median_bar_width_at_q99_sd", "xi_implied"]
    ].median().round(4).to_string())
    print()
    print("P1 borders_in_data falls with the shift · P2 same_bar_90_99 rises · "
          "P3 mass_beyond_inner stays ~0.\nRead xi_implied beside them: section 2.6 "
          "reports it going negative by shift 20.")


if __name__ == "__main__":
    main()
