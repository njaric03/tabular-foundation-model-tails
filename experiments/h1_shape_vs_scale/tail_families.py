# -*- coding: utf-8 -*-
"""Does the dissociation survive other tail families?

The design and the measure of `shape_of_x.py` over four families with the same tail
index xi(x) = 0.15 + 0.75 * Phi(x4) and the same scale s(x):

    gpd       s * ((1 - U)^(-xi) - 1) / xi
    frechet   s * (-ln U)^(-xi)
    burr      s * ((1 - U)^(-xi c) - 1)^(1/c),  c = 2
    studentt  s * |T_nu|,  nu = 1 / xi

The reference is not the family's parameter but the same ratio inversion applied to the
family's true quantiles, so the GPD assumption of the measure drops out of the comparison.

    FAMILIES=frechet,burr SEEDS=3 python -u experiments/h1_shape_vs_scale/tail_families.py
"""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from common import env, generator, gradients, metrics, models, paths, quiet, runner, tables

quiet.silence()

N_TRAIN, N_TEST = 2000, 900
BURR_C = 2.0
N_EST = env.integer("N_EST", 4)
FAMILIES = env.names("FAMILIES", "gpd,frechet,burr,studentt")
SEEDS = env.seeds(env.integer("SEEDS", 5))
MODELS = env.models("GBM,TabICLv2,TabPFN-V3")
OUTPUT = env.text("OUTPUT", "tail_families.csv")

COLUMNS = ["n_est", "family", "model", "seed", "shape_share", "scale_share",
           "xi_mod_t1", "xi_mod_t3", "xi_ref_t1", "xi_ref_t3", "seconds", "reason"]
KEY = ["family", "model", "seed", "n_est"]


def quantile(family, a, xi, s):
    """The true conditional quantile at level `a`, row by row."""
    a = np.asarray(a, dtype=float)
    if family == "gpd":
        core = ((1 - a) ** (-xi) - 1) / xi
    elif family == "frechet":
        core = (-np.log(a)) ** (-xi)
    elif family == "burr":
        core = ((1 - a) ** (-xi * BURR_C) - 1) ** (1.0 / BURR_C)
    elif family == "studentt":
        core = student_t.ppf(0.5 + a / 2.0, df=1.0 / xi)   # |T_nu|
    else:
        raise ValueError(family)
    return s * core


def sample(family, U, xi, s):
    # Written apart from `quantile`, in the association the published runs used.
    if family == "gpd":
        return s * ((1 - U) ** (-xi) - 1) / xi
    if family == "frechet":
        return s * (-np.log(U)) ** (-xi)
    if family == "burr":
        return s * ((1 - U) ** (-xi * BURR_C) - 1) ** (1.0 / BURR_C)
    if family == "studentt":
        return s * np.abs(student_t.ppf(U, df=1.0 / xi))
    raise ValueError(family)


def draw(family, n, rng):
    """The generator's design with another tail family on top; X is drawn before U."""
    c = generator.covariates(n, rng, xi=generator.XI_OF_X)
    U = np.clip(rng.random(n), 1e-9, 1 - 1e-9)
    return c.X, sample(family, U, c.xi, c.s), c.s, c.xi


def measure(cell):
    family, seed = cell["family"], cell["seed"]
    rng = np.random.default_rng(seed)
    Xtr, ytr, _, _ = draw(family, N_TRAIN, rng)
    Xte, _, s_te, xi_te = draw(family, N_TEST, rng)
    q = models.quantiles(cell["model"], Xtr, ytr, Xte, seed=seed,
                         levels=gradients.LEVELS, n_est=cell["n_est"])
    ref_hi, ref_lo = quantile(family, 0.99, xi_te, s_te), quantile(family, 0.9, xi_te, s_te)
    r = gradients.generator_shares(
        q, Xte, lambda rows: metrics.xi_from_ratio(float(np.median(ref_hi[rows] / ref_lo[rows]))),
        quantile(family, 0.5, xi_te, s_te))
    return dict(shape_share=r["shape_share"], scale_share=r["scale_share"],
                xi_mod_t1=r["xi_t1"], xi_mod_t3=r["xi_t3"],
                xi_ref_t1=r["xi_true_t1"], xi_ref_t3=r["xi_true_t3"])


def main():
    cells = runner.grid(family=FAMILIES, model=MODELS, seed=SEEDS, n_est=[N_EST])
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"shape {r['shape_share']:6.0%}  "
                                        f"scale {r['scale_share']:6.0%}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    for col in ("shape_share", "scale_share"):
        print(f"\n=== {col} by family ===")
        print(d.pivot_table(index="family", columns="model", values=col,
                            aggfunc="mean").round(3).to_string())
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


if __name__ == "__main__":
    main()
