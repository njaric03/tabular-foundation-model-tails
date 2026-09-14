# -*- coding: utf-8 -*-
"""Transform the context before TabPFN sees it, and test whether resolution comes back.

`grid_occupancy.py` found the mechanism behind the negative implied xi of part two: one
leverage row inflates the sd, the raw-space grid `borders * y_train_std_ + y_train_mean_`
stretches with it, and the borders over the data fall from 2560 to 224 at a shift of 50.
Linear interpolation inside a bar is locally uniform, and a uniform tail reads as a
negative xi. `fit` re-standardises its input by the plain mean and sd, so a repair has to
be non-linear and act before `fit`. The arms are in `common/transforms.py`:

    raw          as shipped
    robust       affine robust standardisation; a control that must equal raw
    clip_C       context capped at median + C * robust_sd
    log          fit on log(y)
    tail_log     logarithmic above median + 50 robust sd, inverted on the way out
    evt_trim     top values the rest of the tail cannot have produced are pulled down
    sd_cap       the cap that brings the context's own sd shift to 1.2
    rank_emp     normal scores, empirical inverse
    rank_gpd     normal scores, empirical inverse spliced to a GPD
    rank_gpd_trim, rank_exp   rank_gpd with its largest exceedance dropped and the shape
                 capped, on normal and on exponential scores

Every cell is also measured on a clean context (sd_shift_target = 1), where any change
is pure cost. The test set is always clean.

Outcomes on TabPFN-V3, xi 0.7 and 0.9, 20 seeds, paired against raw:

  robust        equals raw to 8e-08, so `fit` does undo an affine map.
  clip_C        keeps 2573 to 2668 borders over the data where raw falls to 217, and the
                implied xi positive. clip_200 costs nothing measurable on a clean context
                and cuts pinball at 0.999 by 38 to 48% at a shift of 50; clip_50 costs 14%
                at xi 0.9. C = 200 was picked after the first run.
  evt_trim      fires in 1 of 20 seeds, clean or not: its false-alarm rate, not power.
  log           repairs less reliably than clip_200 and costs nothing on a clean context.
  tail_log      reads the shape right, but its pinball at 0.999 reaches 1073 against 2.0.
  sd_cap        keeps 2546 to 2569 borders and beats raw at 0.999 in 20/20 and 19/20
                seeds, at a cost of +0.8% and +1.4% at 0.99.
  rank_emp      saturates at the largest value: +1341% at 0.999 under a shift of 50.
  rank_gpd      the sharpest arm at 0.99 on every shift (-2.1% to -2.7% against clip_200),
                and unsafe at 0.999 on real tables, see `unit_error_real.py`.
  rank_gpd_trim, rank_exp   the trim moves the median of the error, the exponential
                scale its worst case; neither removes the blow-ups.

The recommendation depends on the level: a rank-transformed context at 0.99, the derived
cap at 0.999.

    XI=0.7 SEEDS=3 SD_SHIFTS=1,20 CLIP_C=50 python -u experiments/h3_repair/clip_context.py
"""
import numpy as np
import pandas as pd

from common import (
    env,
    generator,
    leverage,
    metrics,
    models,
    paths,
    quiet,
    runner,
    tables,
    transforms,
)

quiet.silence()

XI = env.floats("XI", [0.7, 0.9])
SD_SHIFTS = env.floats("SD_SHIFTS", [1, 4, 20, 50])
CLIP_C = env.floats("CLIP_C", [20, 50, 200])
SEEDS = env.seeds(env.integer("SEEDS", 5))
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = env.integer("N_TEST", 900)
N_EST = env.integer("N_EST", 1)
MODEL = env.models("TabPFN-V3", name="MODEL")[0]
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
OUTPUT = env.text("OUTPUT", "clip_context.csv")

COLUMNS = ["model", "xi", "variant", "clip_c", "sd_shift_target", "sd_shift",
           "seed", "n_est", "n_train", "n_test", "n_clipped",
           "borders_in_data", "same_bar_90_99", "xi_implied", "xi_ratio",
           "q50", "q90", "q99", "q999",
           "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY = ["model", "xi", "variant", "clip_c", "sd_shift_target", "seed", "n_est",
       "n_train", "n_test"]

# (variant, c) per cell: the cap for clip and tail_log, tau for sd_cap, else 0.
ARMS = ([("raw", 0.0), ("robust", 0.0)] + [("clip", c) for c in CLIP_C]
        + [("log", 0.0), ("tail_log", transforms.TAIL_LOG_C), ("evt_trim", 0.0),
           ("sd_cap", transforms.TAU), ("rank_emp", 0.0), ("rank_gpd", 0.0),
           ("rank_gpd_trim", 0.0), ("rank_exp", 0.0)])


def fit_predict(X, y, Xte, seed, n_est):
    """Quantiles in the fitted target's units, and the raw-space borders."""
    m = models.tabpfn_regressor(MODEL, seed, n_est)
    m.fit(X, y)
    return (models.tabpfn_quantiles(m, Xte, LEVELS),
            m.raw_space_bardist_.borders.detach().double().numpy())


def measure(cell):
    xi, variant, c, target, seed = (cell[k] for k in ("xi", "variant", "clip_c",
                                                      "sd_shift_target", "seed"))
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
    test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)

    # The treatment of grid_occupancy.py: one row at the centre of x, sized to the shift.
    X, y, shift = train.X, train.y, 1.0
    if target > 1.0:
        y0 = metrics.y0_for_sd_shift(train.y, target)
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {target} unreachable on this seed")
        X, y = leverage.add_row(train.X, train.y, y0)
        shift = float(metrics.sd_shift(y))

    n_clipped = 0
    if variant == "robust":
        med, rsd = float(np.median(y)), transforms.robust_sd(y)
        q, borders = fit_predict(X, (y - med) / rsd, test.X, seed, cell["n_est"])
        q, borders = q * rsd + med, borders * rsd + med
    else:
        y_fit, back, n_clipped = transforms.transform(variant, c, y)
        q, borders = fit_predict(X, y_fit, test.X, seed, cell["n_est"])
        # exp and expm1 overflow on the outer borders; only the ones over the data count.
        with np.errstate(over="ignore", invalid="ignore"):
            q, borders = back(q), back(borders)

    b90, b99 = (np.clip(np.searchsorted(borders, q[:, i]) - 1, 0, len(borders) - 2)
                for i in (1, 2))
    r = dict(sd_shift=shift, n_clipped=n_clipped,
             borders_in_data=int(((borders >= train.y.min()) & (borders <= train.y.max())).sum()),
             same_bar_90_99=float((b90 == b99).mean()),
             xi_implied=metrics.implied_xi(q[:, 2], q[:, 1], q[:, 0]),
             # The raw ratio, biased low under a location component; the gap between the
             # two is how much of the negative xi belongs to the measure.
             xi_ratio=metrics.implied_xi(q[:, 2], q[:, 1]))
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        r[f"q{t}"] = float(np.median(q[:, i]))
        r[f"pb{t}"] = metrics.pinball(test.y, q[:, i], a)
    return r


def main():
    cells = [dict(model=MODEL, xi=xi, variant=v, clip_c=c, sd_shift_target=target, seed=seed,
                  n_est=N_EST, n_train=N_TRAIN, n_test=N_TEST)
             for xi in XI for target in SD_SHIFTS for seed in SEEDS for v, c in ARMS]
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"borders {r['borders_in_data']:>4}  xi "
                                        f"{r['xi_implied']:+.3f}  pb99 {r['pb99']:.4f}  "
                                        f"pb999 {r['pb999']:.4f}  clipped {r['n_clipped']}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    d["arm"] = np.where(d.variant == "clip", "clip_" + d.clip_c.map("{:g}".format), d.variant)
    for col in ("borders_in_data", "xi_implied", "xi_ratio", "pb999"):
        print(f"\n=== {col}, median by shift ===")
        print(d.pivot_table(index=["xi", "sd_shift_target"], columns="arm", values=col,
                            aggfunc="median").round(4).to_string())
    key = ["xi", "sd_shift_target", "seed"]
    j = (d[d.variant == "raw"].set_index(key)[["q99"]]
         .join(d[d.variant == "robust"].set_index(key)[["q99"]], rsuffix="_robust", how="inner"))
    if len(j):
        gap = float((j.q99 - j.q99_robust).abs().max() / j.q99.abs().max())
        print(f"\ncontrol, robust against raw: largest relative gap in Q(0.99) {gap:.2e}"
              f" ({'holds' if gap < 1e-3 else 'FAILS'})")


if __name__ == "__main__":
    main()
