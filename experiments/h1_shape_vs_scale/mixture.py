# -*- coding: utf-8 -*-
"""Does a mixture of the members keep the tail Vincentization loses, and why is TabICL
unaffected?

Members are built by hand, one estimator each with its own seed, so every member's
distribution is available. From the same members, per test row:

    single       mean implied xi over the members, no aggregation
    vincentized  quantiles averaged level by level, as the libraries do
    mixture      CDFs averaged, then inverted

and the members' disagreement, the median coefficient of variation of Q(0.99). If the
mixture keeps the tail it is a repair; if TabICL's members disagree less, that explains
why it is unaffected.

    XI=0.9 MEMBERS=4 SEEDS=3 python -u experiments/h1_shape_vs_scale/mixture.py
"""
import numpy as np
import pandas as pd

from common import env, generator, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
XI = env.floats("XI", [0.7, 0.9])
MEMBERS = env.integer("MEMBERS", 4)
SEEDS = env.seeds(env.integer("SEEDS", 3))
MODELS = env.models("TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "mixture.csv")
# A dense grid for the CDF that contains exactly 0.90 and 0.99, so the implied xi is
# read at the levels every other script uses.
LEVELS = np.unique(np.round(np.concatenate([np.linspace(0.02, 0.995, 80), [0.90, 0.99]]), 4))
I_LO = int(np.where(LEVELS == 0.90)[0][0])
I_HI = int(np.where(LEVELS == 0.99)[0][0])

COLUMNS = ["members", "xi", "model", "seed", "xi_single", "xi_vincentized", "xi_mixture",
           "disagreement_cv", "seconds", "reason"]
KEY = ["xi", "model", "seed", "members"]


def mixture_quantiles(Q, levels, targets):
    """Quantiles at `targets` of the equal-weight mixture of members.

    Q has shape (members, rows, levels). Per row, every member's CDF is interpolated on a
    common grid of all members' quantiles, averaged, and inverted.
    """
    M, n, _ = Q.shape
    out = np.empty((n, len(targets)))
    for i in range(n):
        y = np.sort(Q[:, i, :].ravel())
        F = np.zeros_like(y)
        for m in range(M):
            F += np.interp(y, Q[m, i, :], levels, left=levels[0], right=levels[-1])
        out[i] = np.interp(targets, F / M, y)
    return out


def measure(cell):
    rng = np.random.default_rng(cell["seed"])
    train = generator.gpd(N_TRAIN, rng, xi=cell["xi"], clip=True)
    test = generator.gpd(N_TEST, rng, xi=cell["xi"], clip=True)
    Q = np.stack([models.quantiles(cell["model"], train.X, train.y, test.X,
                                   seed=cell["seed"] + 137 * k, levels=list(LEVELS), n_est=1)
                  for k in range(cell["members"])])
    V = Q.mean(axis=0)
    mix = mixture_quantiles(Q, LEVELS, LEVELS[[I_LO, I_HI]])
    q99 = Q[:, :, I_HI]
    return dict(
        xi_single=float(np.nanmean([metrics.implied_xi(m[:, I_HI], m[:, I_LO]) for m in Q])),
        xi_vincentized=metrics.implied_xi(V[:, I_HI], V[:, I_LO]),
        xi_mixture=metrics.implied_xi(mix[:, 1], mix[:, 0]),
        disagreement_cv=float(np.median(q99.std(axis=0, ddof=1)
                                        / np.maximum(np.abs(q99.mean(axis=0)), 1e-9))))


def main():
    cells = runner.grid(xi=XI, model=MODELS, seed=SEEDS, members=[MEMBERS])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"single {r['xi_single']:+.3f} | vincentized "
                                        f"{r['xi_vincentized']:+.3f} | mixture "
                                        f"{r['xi_mixture']:+.3f} | cv {r['disagreement_cv']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== implied xi: one member, Vincentized, mixture; and disagreement ===")
    print(d.pivot_table(index=["xi", "model"], values=["xi_single", "xi_vincentized",
                                                        "xi_mixture", "disagreement_cv"])
          .round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
