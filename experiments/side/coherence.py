# -*- coding: utf-8 -*-
"""Are the predicted quantiles coherent under a reparametrisation of the target?

A monotone map g commutes with quantiles, Q_g(Y)(a | x) = g(Q_Y(a | x)), for any true
distribution. A coherent predictor therefore satisfies

    g^-1(Q_model(a; g(y))) ~ Q_model(a; y)

and any gap is incoherence: the prediction depends on a choice that does not change the
truth. A log transform once raised the implied xi from 0.47 to 0.65 at a true 0.7. On a
multiplicative generator the log linearises the structure by construction, so an additive
generator (GENERATOR=additive) is the control where log is not the natural scale.

    deviation   median |log(Q_transformed / Q_raw)|, averaged over levels

    XI=0.7 TRANSFORMS=log SEEDS=3 python -u experiments/side/coherence.py
"""
import numpy as np
import pandas as pd

from common import env, generator, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
LEVELS = [0.5, 0.9, 0.99]
XI = env.floats("XI", [0.3, 0.7, 0.9])
TRANSFORMS = env.names("TRANSFORMS", "log,cbrt,sqrt")
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("TabICLv2,TabPFN-V3,GBM")
N_EST = env.integer("N_EST", 1)
GENERATOR = env.text("GENERATOR", "multiplicative")
OUTPUT = env.text("OUTPUT", "coherence.csv")

COLUMNS = ["n_est", "generator", "xi", "model", "transform", "seed", "xi_raw", "xi_transformed",
           "deviation", "deviation_q50", "deviation_q90", "deviation_q99", "seconds", "reason"]
KEY = ["xi", "model", "transform", "seed", "n_est", "generator"]

MAPS = {
    "log": (np.log, np.exp),
    "cbrt": (np.cbrt, lambda z: z ** 3),
    "sqrt": (np.sqrt, lambda z: np.maximum(z, 0.0) ** 2),
}


def draw(n, xi, rng, kind):
    """The generator's design; X is drawn before U, as in `generator.gpd`."""
    c = generator.covariates(n, rng, xi=xi)
    U = np.clip(rng.random(n), 1e-9, 1 - 1e-9)
    noise = ((1 - U) ** (-xi) - 1) / xi
    if kind == "additive":
        return c.X, 5.0 + 2.0 * c.lin + noise        # positive, and log(y) is not additive
    return c.X, np.exp(0.6 * c.lin) * noise


def measure(cell):
    g, g_inv = MAPS[cell["transform"]]
    rng = np.random.default_rng(cell["seed"])
    Xtr, ytr = draw(N_TRAIN, cell["xi"], rng, cell["generator"])
    Xte, _ = draw(N_TEST, cell["xi"], rng, cell["generator"])
    ytr = np.maximum(ytr, 1e-9)

    def predict(y):
        return models.quantiles(cell["model"], Xtr, y, Xte, seed=cell["seed"], levels=LEVELS,
                                n_est=cell["n_est"])

    q_raw = predict(ytr)
    q_back = g_inv(predict(g(ytr)))
    ok = (q_raw > 1e-9) & (q_back > 1e-9)
    deviation = [float(np.median(np.abs(np.log(q_back[ok[:, i], i] / q_raw[ok[:, i], i]))))
                 if ok[:, i].any() else np.nan for i in range(len(LEVELS))]
    return dict(xi_raw=metrics.implied_xi(q_raw[:, 2], q_raw[:, 1]),
                xi_transformed=metrics.implied_xi(q_back[:, 2], q_back[:, 1]),
                deviation=float(np.nanmean(deviation)), deviation_q50=deviation[0],
                deviation_q90=deviation[1], deviation_q99=deviation[2])


def main():
    cells = runner.grid(xi=XI, model=MODELS, transform=TRANSFORMS, seed=SEEDS, n_est=[N_EST],
                        generator=[GENERATOR])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"xi raw {r['xi_raw']:+.3f} -> transformed "
                                        f"{r['xi_transformed']:+.3f} | "
                                        f"deviation {r['deviation']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    print("\n=== implied xi, raw fit against transformed and mapped back, median ===")
    print(d.pivot_table(index=["transform", "xi"], columns="model",
                        values=["xi_raw", "xi_transformed"], aggfunc="median").round(3).to_string())
    print("\n=== incoherence, median |log ratio| ===")
    print(d.pivot_table(index=["transform", "xi"], columns="model", values="deviation",
                        aggfunc="median").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
