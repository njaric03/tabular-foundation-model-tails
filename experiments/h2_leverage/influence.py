# -*- coding: utf-8 -*-
"""The influence of one extreme row in the context.

The model is fitted twice on the same data, without and with one added row (x0, y0), and
the change of its predicted quantiles is measured:

    local    median relative change on the 10% of test rows nearest to x0
    global   the same over all test rows

A Bayesian model under a heavy-tailed prior would raise the tail, a robust one would
ignore the row, a model clipping outliers in preprocessing would saturate, and one that
standardises the target by a plain sd would move everywhere.

DOSE_MODE=max sets y0 = dose * max(y_train). max(y_train) is itself heavy-tailed, so one
nominal dose is not one treatment across seeds; DOSE_MODE=sd sets y0 so the sd shift
equals the dose, the axis the prevalence survey is measured on (worst of 99 tables: 19.9).

    XI=0.7 SEEDS=3 MODELS=TabICLv2 python -u experiments/h2_leverage/influence.py
    DOSE_MODE=sd DOSES=2,4,10,20 OUTPUT=influence_sd.csv \
      python -u experiments/h2_leverage/influence.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, metrics, models, paths, quiet, runner, tables

quiet.silence()

LEVELS = [0.5, 0.9, 0.99]
TAGS = ["q50", "q90", "q99"]
NEAR_SHARE = 0.10
N_TEST = 900
XI = env.floats("XI", [0.3, 0.7])
DOSES = env.floats("DOSES", [1, 3, 10, 100])
DOSE_MODE = env.text("DOSE_MODE", "max")
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
N_EST = env.integer("N_EST", 1)
POSITION = env.text("POSITION", "centre")
N_TRAIN = env.integer("N_TRAIN", 2000)
OUTPUT = env.text("OUTPUT", "influence.csv")
if DOSE_MODE not in ("max", "sd"):
    raise SystemExit(f"DOSE_MODE={DOSE_MODE!r}: expected max or sd")
if POSITION not in ("centre", "edge", "random"):
    raise SystemExit(f"POSITION={POSITION!r}: expected centre, edge or random")

COLUMNS = ["xi", "model", "dose", "dose_mode", "seed", "position", "n_train", "n_est",
           "sd_shift", "y0_over_max",
           "local_q50", "local_q90", "local_q99", "global_q50", "global_q90", "global_q99",
           "xi_without", "xi_with", "seconds", "reason"]
KEY = ["xi", "model", "dose", "dose_mode", "seed", "position", "n_train", "n_est"]

# The clean fit of the last (xi, model, seed); every dose of that triple reuses it.
_clean: dict = {}


def measure(cell):
    xi, name, seed, dose = cell["xi"], cell["model"], cell["seed"], cell["dose"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
    test = generator.gpd(N_TEST, rng, xi=xi, clip=True)

    def predict(X, y):
        return models.quantiles(name, X, y, test.X, seed=seed, levels=LEVELS, n_est=cell["n_est"])

    if (xi, name, seed) not in _clean:
        _clean.clear()
        _clean[(xi, name, seed)] = predict(train.X, train.y)
    q_clean = _clean[(xi, name, seed)]

    d = train.X.shape[1]
    x0 = {"centre": np.zeros((1, d)), "edge": 2.0 * np.ones((1, d)),
          "random": train.X[[int(np.random.default_rng(seed).integers(len(train.X)))]]}[
        cell["position"]]
    if cell["dose_mode"] == "sd":
        y0 = metrics.y0_for_sd_shift(train.y, dose)
        if not np.isfinite(y0):
            return dict(reason=f"sd shift {dose} unreachable: a duplicate of the "
                               f"maximum already gives more")
    else:
        y0 = dose * train.y.max()
    q_row = predict(*leverage.add_row(train.X, train.y, y0, x0))

    distance = np.linalg.norm(test.X - x0, axis=1)
    near = distance <= np.quantile(distance, NEAR_SHARE)
    r = dict(zip([f"local_{t}" for t in TAGS],
                 metrics.rel_change_by_level(q_clean[near], q_row[near], positive=True)))
    r.update(zip([f"global_{t}" for t in TAGS],
                 metrics.rel_change_by_level(q_clean, q_row, positive=True)))
    return dict(r, xi_without=metrics.implied_xi(q_clean[:, 2], q_clean[:, 1]),
                xi_with=metrics.implied_xi(q_row[:, 2], q_row[:, 1]),
                sd_shift=metrics.sd_shift(np.append(train.y, y0)),
                y0_over_max=float(y0 / train.y.max()))


def main():
    cells = runner.grid(xi=XI, model=MODELS, seed=SEEDS, dose=DOSES, dose_mode=[DOSE_MODE],
                        position=[POSITION], n_train=[N_TRAIN], n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"local q50 {r['local_q50']:+.1%} q99 "
                                        f"{r['local_q99']:+.1%} | global q99 "
                                        f"{r['global_q99']:+.1%} | xi {r['xi_without']:.2f}"
                                        f" -> {r['xi_with']:.2f}")
    # Rows with no implied xi keep their responses: the inversion fails exactly on the
    # heaviest tails, so dropping those rows would bias every table towards light ones.
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for col in ("local_q50", "local_q99", "global_q99"):
        print(f"\n=== {col}, median ===")
        print(d.pivot_table(index="dose", columns="model", values=col,
                            aggfunc="median").round(3).to_string())
    print("\n=== change in implied xi, median ===")
    print(d.assign(dxi=d.xi_with - d.xi_without)
          .pivot_table(index="dose", columns="model", values="dxi", aggfunc="median")
          .round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
