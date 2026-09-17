# -*- coding: utf-8 -*-
"""The pinball of two forecasters no model enters, on the test rows of the leverage tests.

Pinball is in units of the target, so it cannot be pooled across xi, families or tables,
and a model's own clean pinball is no reference when its clean prediction is far off (the
fourth outcome of `prior_pretraining_eval.py`). Two references instead:

  oracle    on the generator, the true conditional quantile of every test row: s(x) times
            the family's quantile at the level. The test rows are those that
            `prior_pretraining_eval.py` and `clip_context.py` draw for (family, xi, seed,
            n_train, n_test), both as context first and test second; the leverage row
            enters the context only, so one oracle serves every shift and every repair.
  context   on the claim tables of `prior_pretraining_real.py`, the empirical quantile of
            the clean context, one value for every test row: what a forecaster that
            ignores the features and trusts the context would say.

The cells are read from the evaluation files, so every cell measured there has its
reference. A model's excess pinball is its pinball minus the oracle's, and its ratio to
the oracle is at least 1 in expectation; on the claims the ratio to the context quantile
is below 1 when the model beats it.

READ AFTER THE OUTCOMES, 17.9.2026: PINBALL OVER A REFERENCE NO MODEL ENTERS
--------------------------------------------------------------------------
Not predicted. Median over cells of a model's pinball over the oracle's (generator) or over
the context quantile's (claims); 1 is the reference. On 900 or 1000 test rows 0.999 sees
about one exceedance, so that level reads mostly overshoot.

nanoTabPFN arms at 200 context rows, pre-training seed 92 / 93, at 0.99:

                   clean          shift 4        shift 20       shift 50
    GPD      A0B0  12.0 / 11.8    30.9 / 26.6    54.4 / 48.5    84.1 / 70.8
             A1B0  1.71 / 1.73    1.13 / 1.15    1.15 / 1.12    1.11 / 1.14
             ATB0  1.35 / 1.35    1.11 / 1.12    1.15 / 1.42    1.28 / 2.18
             ACB0  4.74 / 3.20    4.09 / 1.66    1.11 / 1.17    1.24 / 1.19
    families A0B0  11.7 / 11.7    28.9 / 27.5    58.8 / 53.8    85.5 / 72.5
             A1B0  4.19 / 4.20    1.20 / 1.16    1.12 / 1.10    1.08 / 1.13
             ATB0  4.02 / 4.01    1.09 / 1.10    1.20 / 1.43    1.28 / 2.28
             ACB0  4.57 / 3.19    3.15 / 1.53    1.10 / 1.22    1.22 / 1.18
    claims   A0B0  19.8 / 21.7    50.6 / 39.6    69.7 / 157     55.6 / 105
             A1B0  17.3 / 17.7    2.04 / 1.14    1.06 / 0.99    1.01 / 0.99
             ATB0  13.7 / 15.1    14.4 / 14.9    1.12 / 1.54    1.02 / 1.36
             ACB0  18.7 / 17.3    2.15 / 1.16    1.00 / 1.00    1.03 / 0.99

At 0.999 and a shift of 50: A0B0 188 to 205 on the generator, 298 and 320 on the claims;
A1B0 1.22 to 1.55 and 1.18 / 0.84; ACB0 1.53 to 2.01 and 0.94 / 0.81; ATB0 75 to 131 and
13.7 / 106. At a shift of 4 on the generator: ATB0 3.9 to 5.8, A1B0 3.8 to 7.7, ACB0 9.0
to 13.5, A0B0 17.1 to 17.8.

TabPFN-V3 at 2000 context rows (`clip_context.csv`, 20 seeds, xi 0.7 and 0.9), clean and
at shifts 4 / 20 / 50:

    raw        0.99  1.05   1.09 / 1.19 / 1.23     0.999  1.17   1.27 / 1.70 / 2.03
    sd_cap     0.99  1.05   1.06 at every shift     0.999  1.12   1.03 at every shift
    clip_200   0.99  1.04   1.04 at every shift     0.999  1.14   1.08 at every shift
    rank_gpd   0.99  1.02   1.01 at every shift     0.999  1.18   1.25 / 1.31 / 1.36
    rank_emp   0.99  1.02   1.04 at every shift     0.999  1.15   2.20 / 8.14 / 19.5

What the reference adds.

1. Against the oracle the small model's failure is larger than its own clean pinball
   showed: A0B0 is 12 times the oracle on a clean context and 71 to 86 times at a shift of
   50. Under a shift of 20 or 50 the modified priors sit at 1.1 to 1.4 (ATB0-s93 at 50,
   2.2 and 2.3), in most cells closer to the oracle than on their own clean contexts
   (1.35 to 4.74).
2. At 0.999 the ablation's order reverses at large shifts. The tail alone is 75 to 131
   times the oracle at 50, in A0B0's range, while both arms that carry the contamination
   stay within about 2. At a shift of 4 the tail alone and both together are close (3.8 to
   7.7) and the contamination alone is worst (9.0 to 13.5). Which treatment matters depends
   on the level and on the shift.
3. On the claims no arm beats the quantile of its clean context, a forecaster that ignores
   every feature: 14 to 22 times worse at 0.99 on a clean context, and at 0.5 and 0.9 within
   3% of it. Under a shift of 20 or 50, A1B0 and ACB0 match it (0.99 to 1.06). At 200 rows
   the small models add almost nothing to the marginal of these tables.
4. The production model is a different case. TabPFN-V3 is within 5% of the oracle at 0.99
   on a clean context, and one leverage row at a shift of 50 moves it to 1.23, and at 0.999
   from 1.17 to 2.03. In the same cells its implied xi falls from 0.61 and 0.75 to -0.30
   and -0.29, and the borders over the data from about 2550 to about 200. The mechanism
   measures show a change of tail family; the pinball shows what that costs a forecast, 17%
   more at 0.99 and 74% more at 0.999. Both belong in part two, and they are not
   interchangeable. The caps hold it at 1.03 to 1.08 at both levels under every shift. The
   rank transforms and log are the sharpest at 0.99 (1.01 to 1.04), and the empirical rank
   reaches 19.5 times the oracle at 0.999 at a shift of 50: the level dependence of part
   three, in these units.

The oracle is exact on the generator. The context quantile on the claims is a baseline and
not an oracle, and 1000 held-out claims give it noise of its own. Medians over cells, not
tests; the units are those of the sections above.

    venv-tfmp/Scripts/python.exe -u experiments/h3_repair/pinball_reference.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, generator, leverage, metrics, paths, quiet, runner

quiet.silence()

LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
MAX_ATTEMPTS = env.integer("MAX_ATTEMPTS", 4000)
OUTPUT1 = env.text("OUTPUT1", "pinball_oracle.csv")
OUTPUT2 = env.text("OUTPUT2", "pinball_reference_real.csv")
# The synthetic evaluations whose test rows the oracle scores.
SOURCES = ("prior_pretraining_eval*.csv", "clip_context.csv")

COLUMNS1 = ["family", "xi", "seed", "n_train", "n_test",
            "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY1 = ["family", "xi", "seed", "n_train", "n_test"]
COLUMNS2 = ["dataset", "repeat", "n_train", "n_test", "q99",
            "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY2 = ["dataset", "repeat", "n_train", "n_test"]


def synthetic_cells():
    folder = paths.result("prior_pretraining_eval.csv").parent
    frames = []
    for f in sorted(p for pattern in SOURCES for p in folder.glob(pattern)):
        d = pd.read_csv(f)
        if "family" not in d:
            d["family"] = "gpd"             # written before the knob existed
        frames.append(d[KEY1])
    cells = pd.concat(frames).drop_duplicates().sort_values(KEY1)
    return [dict(family=r.family, xi=float(r.xi), seed=int(r.seed), n_train=int(r.n_train),
                 n_test=int(r.n_test)) for r in cells.itertuples()]


def oracle(cell):
    rng = np.random.default_rng(cell["seed"])
    family, xi = cell["family"], cell["xi"]
    # Drawn as both evaluations draw them: the context first, then the test.
    if family == "gpd":
        generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
        test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)
    else:
        generator.family(family, cell["n_train"], rng, xi=xi)
        test = generator.family(family, cell["n_test"], rng, xi=xi)

    def quantile(a):
        if family == "gpd":
            return test.s * metrics.gpd_quantile(a, test.xi)
        return generator.family_quantile(family, a, test.xi, test.s)

    return {f"pb{t}": metrics.pinball(test.y, quantile(a), a) for a, t in zip(LEVELS, TAGS)}


def context_reference(cell):
    _, y, _, y_test, _ = cell["_context"]
    q = np.quantile(y, LEVELS)
    r = {f"pb{t}": metrics.pinball(y_test, np.full(len(y_test), q[i]), a)
         for i, (a, t) in enumerate(zip(LEVELS, TAGS))}
    r["q99"] = float(q[2])
    return r


def main():
    seconds = runner.run(OUTPUT1, COLUMNS1, KEY1, synthetic_cells(), oracle,
                         show=lambda r: f"oracle pb99 {r['pb99']:.4g}")
    real = paths.result("prior_pretraining_real.csv")
    if real.exists():
        d = pd.read_csv(real)
        for (name, n_train, n_test), g in d.groupby(["dataset", "n_train", "n_test"]):
            X, y = datasets.load(name)
            ctx = leverage.clean_subsamples(X, y, int(n_train), int(n_test),
                                            int(g.repeat.max()) + 1, MAX_ATTEMPTS)
            cells = [dict(dataset=name, repeat=j, n_train=int(n_train), n_test=int(n_test),
                          _context=ctx[j]) for j in sorted(g.repeat.unique())]
            seconds += runner.run(OUTPUT2, COLUMNS2, KEY2, cells, context_reference,
                                  show=lambda r: f"context pb99 {r['pb99']:.4g}")
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT1}, {OUTPUT2}")


if __name__ == "__main__":
    main()
