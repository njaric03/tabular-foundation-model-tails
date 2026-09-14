# -*- coding: utf-8 -*-
"""Where the predictive mass sits on TabPFN's bar grid under a dosed leverage row.

From the source of `tabpfn` 8.4.0: the raw-space grid is
`borders * y_train_std_ + y_train_mean_`, with 341 borders per unit in the body and three
between 50 and 128 sd, and the inherited `icdf` interpolates linearly inside a bar. A row
that inflates the sd by 20 stretches the grid by 20 while the data stays put, so the data
spans few bars, and a linear, locally uniform tail reads as a negative xi. Dosed on the sd
shift at a true xi = 0.7, TabPFN's implied xi goes negative by a shift of 20 while GBM
stays flat.

Predictions written before the run:
  P1. `borders_in_data` collapses roughly in proportion to the shift (a sanity check).
  P2. `same_bar_90_99`, the share of rows whose Q(0.9) and Q(0.99) share a bar, is near
      zero on a clean context and rises sharply by a shift of 20.
  P3. `mass_beyond_inner`, the mass in the last bar, stays negligible: the half-normal
      tail branch is real but not reached by this treatment.

Only models with a bar grid can be measured; the others are refused by name.

    MODELS=TabPFN-V3 XI=0.7 SD_SHIFTS=1,2,4,10,20,50 SEEDS=5 \
        python -u experiments/h2_leverage/grid_occupancy.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, metrics, models, paths, quiet, runner, tables

quiet.silence()

XI = env.floats("XI", [0.7])
# A target shift of 1 means no added row: the clean context on the same code path.
SD_SHIFTS = env.floats("SD_SHIFTS", [1, 2, 4, 10, 20, 50])
SEEDS = list(range(env.integer("SEEDS", 5)))
MODELS = env.models("TabPFN-V3")
N_EST = env.integer("N_EST", 1)
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = env.integer("N_TEST", 900)
OUTPUT = env.text("OUTPUT", "grid_occupancy.csv")

LEVELS = [0.5, 0.9, 0.99, 0.999]
GRID_MODELS = {"TabPFN-V3", "TabPFN-v2.5", "TabPFN-v2.6"}

COLUMNS = ["xi", "model", "sd_shift_target", "sd_shift", "seed", "n_est", "n_train", "n_test",
           "n_bars", "border_last", "y_std_clean", "y_std_with",
           "borders_in_data", "bars_in_data_share",
           "same_bar_90_99", "same_bar_99_999", "mass_beyond_inner",
           "median_bar_width_at_q99_sd", "q999_in_outer_five",
           "xi_implied", "seconds", "reason"]
KEY = ["xi", "model", "sd_shift_target", "seed", "n_est", "n_train", "n_test"]


def full_prediction(name, X, y, Xte, seed, n_est):
    """Raw-space borders, per-row bar probabilities, quantiles and the fitted target sd."""
    import torch

    m = models.tabpfn_regressor(name, seed, n_est)
    m.fit(X, y)
    # Without `quantiles` the full output carries the default levels 0.1..0.9, and
    # columns 1 and 2 would silently be Q(0.2) and Q(0.3).
    out = m.predict(Xte, output_type="full", quantiles=LEVELS)
    borders = out["criterion"].borders.detach().double().numpy()
    probs = torch.softmax(out["logits"].double(), -1).detach().numpy()
    q = np.stack([np.asarray(a, dtype=float) for a in out["quantiles"]], axis=1)
    return borders, probs, q, float(m.y_train_std_)


def bar_of(borders, values):
    return np.clip(np.searchsorted(borders, values) - 1, 0, len(borders) - 2)


def measure(cell):
    name, target, seed = cell["model"], cell["sd_shift_target"], cell["seed"]
    if name not in GRID_MODELS:
        return dict(reason=f"{name} has no bar grid to occupy")
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=cell["xi"], clip=True)
    Xte = generator.gpd(cell["n_test"], rng, xi=cell["xi"], clip=True).X

    X, y, shift = train.X, train.y, 1.0
    if target > 1.0:
        y0 = metrics.y0_for_sd_shift(train.y, target)
        if not np.isfinite(y0):
            return dict(reason=f"sd shift {target} unreachable on this seed")
        X, y = leverage.add_row(train.X, train.y, y0)
        shift = float(metrics.sd_shift(y))

    borders, probs, q, std_with = full_prediction(name, X, y, Xte, seed, cell["n_est"])
    inside = int(((borders >= train.y.min()) & (borders <= train.y.max())).sum())
    b90, b99, b999 = (bar_of(borders, q[:, i]) for i in (1, 2, 3))
    std_clean = float(np.std(train.y))
    return dict(
        sd_shift=shift, n_bars=len(borders) - 1, border_last=float(borders[-1]),
        y_std_clean=std_clean, y_std_with=std_with,
        borders_in_data=inside, bars_in_data_share=inside / (len(borders) - 1),
        same_bar_90_99=float((b90 == b99).mean()),
        same_bar_99_999=float((b99 == b999).mean()),
        mass_beyond_inner=float(probs[:, -1].mean()),
        median_bar_width_at_q99_sd=float(np.median(np.diff(borders)[b99]) / std_clean),
        q999_in_outer_five=float((b999 >= len(borders) - 6).mean()),
        xi_implied=metrics.implied_xi(q[:, 2], q[:, 1]))


def main():
    cells = runner.grid(xi=XI, model=MODELS, sd_shift_target=SD_SHIFTS, seed=SEEDS,
                        n_est=[N_EST], n_train=[N_TRAIN], n_test=[N_TEST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"same bar {r['same_bar_90_99']:.3f}  "
                                        f"xi {r['xi_implied']:+.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print(d.groupby(["model", "sd_shift_target"])[
        ["borders_in_data", "same_bar_90_99", "same_bar_99_999", "mass_beyond_inner",
         "median_bar_width_at_q99_sd", "xi_implied"]].median().round(4).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
