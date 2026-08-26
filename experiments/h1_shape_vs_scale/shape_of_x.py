# -*- coding: utf-8 -*-
"""H1, the core measurement: does the model change the SHAPE of the tail with x?

Every generator used before this one held the tail shape constant across points,
so only the scale varied and the implied xi could be estimated from all rows at
once. In practice that is almost never the case: in claims the tail depends on
the policy type, in rainfall on the region, in prices on the segment. The
question this asks is stronger than the earlier one:

    does the model adapt the tail shape to x at all, or hand everyone the same tail?

GENERATOR
    xi(x) = XI_LO + (XI_HI - XI_LO) * Phi(x[:, 4])
    y | x ~ GPD(xi(x), s(x)),  s(x) = exp(0.6 * x.w / norm(w))

x4 is chosen because its weight in w is zero, so the scale does NOT depend on
it. That separates shape from level: whatever the model captures about the tail
through x4 is genuine shape adaptation, not a by-product of a larger scale.

MEASURE
Test points are split into terciles by x4. Within each tercile the implied xi is
inverted from the ratio Q(0.99)/Q(0.9) and compared with the true mean xi there.

    rises across terciles, tracking the truth -> the model adapts the tail
    flat across terciles                      -> the model gives everyone the same tail
    rises but too little                      -> partial adaptation, measure the slope

The companion measurement is scale_of_x.py, which asks the same question about
the scale. The finding is the DIFFERENCE between the two shares, both measured
by this same procedure on the same data.

    MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv \
      python -u experiments/h1_shape_vs_scale/shape_of_x.py

Results in shape_of_x_20.csv (n_est=4) and shape_of_x_nest1.csv (n_est=1).
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths

warnings.filterwarnings("ignore")

N_TRAIN, N_TEST = 2000, 900
# Ensemble size is a knob: vincentization.md shows aggregation lowers the
# implied xi by 0.17 at xi=0.9 for TabPFN. The library default is 4.
N_EST = int(os.environ.get("N_EST", "4"))
# Seed count is a knob because the oracle control in oracle_xi_of_x.py shows
# wide scatter at n=2000 (p10 +0.14, p90 +0.88): with three seeds the model
# slope distribution cannot be compared with the oracle one.
SEEDS = [7000 + 1000 * i for i in range(int(os.environ.get("SEEDS", "20")))]
OUTPUT = os.environ.get("OUTPUT", "shape_of_x_20.csv")
MODELS = os.environ.get("MODELS", "GBM,TabICLv2,TabPFN-V3").split(",")
QS = [0.5, 0.9, 0.99]
HI, LO = 0.99, 0.9
N_TERCILES = 3

# n_est MUST be a column: the captured shape share depends strongly on it
# (TabPFN-V3 19% at 4 against 42% at 1). While it lived only in the file name,
# numbers from different regimes were mixed together.
COLUMNS = ["model", "seed", "tercile", "n_est", "xi_true", "ratio_model", "xi_implied"]
KEY = ["model", "seed", "tercile", "n_est"]


def terciles(values, k=N_TERCILES):
    """Index 0..k-1 of the tercile each value falls into."""
    edges = np.quantile(values, np.linspace(0, 1, k + 1))
    return np.clip(np.digitize(values, edges[1:-1]), 0, k - 1)


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train = generator.gpd(N_TRAIN, rng, xi=generator.XI_OF_X)
        test = generator.gpd(N_TEST, rng, xi=generator.XI_OF_X)
        # Terciles by x4, which is terciles by the true tail shape.
        tercile = terciles(test.X[:, 4])

        for name in MODELS:
            if all(append.key(dict(model=name, seed=seed, tercile=t + 1, n_est=N_EST),
                              KEY) in done for t in range(N_TERCILES)):
                continue
            q = models.quantiles(name, train.X, train.y, test.X, seed=seed,
                                 levels=QS, n_est=N_EST)
            shown = []
            for t in range(N_TERCILES):
                sel = tercile == t
                num, den = q[sel, QS.index(HI)], q[sel, QS.index(LO)]
                ok = den > 1e-9
                ratio = float(np.median(num[ok] / den[ok])) if ok.any() else np.nan
                row = dict(model=name, seed=seed, tercile=t + 1, n_est=N_EST,
                           xi_true=float(test.xi[sel].mean()),
                           ratio_model=ratio,
                           xi_implied=(metrics.xi_from_ratio(ratio)
                                       if np.isfinite(ratio) else np.nan))
                if append.key(row, KEY) not in done:
                    append.write(OUTPUT, row, COLUMNS)
                shown.append(f"{row['xi_implied']:.2f}")
            print(f"  {name:10s} seed={seed}  by tercile: {' / '.join(shown)}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    report(t0)


def report(t0=None):
    d = pd.read_csv(paths.result(OUTPUT))
    pd.set_option("display.width", 220)
    print("\n=== TRUE xi by tercile ===")
    print(d.groupby("tercile").xi_true.mean().round(3).to_string())
    print("\n=== IMPLIED xi by tercile ===")
    piv = d.pivot_table(index="tercile", columns="model", values="xi_implied",
                        aggfunc="mean")
    print(piv.round(3).to_string())
    print("\n=== slope: change in implied xi from the 1st to the 3rd tercile ===")
    truth = d.groupby("tercile").xi_true.mean()
    true_slope = truth.iloc[-1] - truth.iloc[0]
    print(f"  true slope: {true_slope:.3f}")
    for c in piv.columns:
        slope = piv[c].iloc[-1] - piv[c].iloc[0]
        print(f"  {c:10s} {slope:+.3f}   share tracked: {slope / true_slope:.0%}")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
