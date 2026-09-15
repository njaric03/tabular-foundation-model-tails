# -*- coding: utf-8 -*-
"""Is the upper-tail failure in the pre-training prior or in the architecture?

WHY
---
Part two found that one leverage row inflates the context sd, the bar grid stretches with
it, and the implied tail index of TabPFN turns negative; on real insurance data TabPFN
v2.5 and v2.6 lose about half of Q(0.99). EXAONE standardises the target the same way and
loses nothing, and TabICLv2 has no grid and still loses 15%. So standardisation alone costs
part of the tail, the grid multiplies it, and something else, plausibly the prior, lets
one model avoid it. The production models cannot be re-trained. A small model of the same
architecture can: TFM-Playground's nanoTabPFN, pre-trained here under a 2 x 2 of prior and
target encoding with everything else fixed (`prior_pretraining_train.py`). It shows the
direction of an effect, not its size in the production models.

    A0  TabICL's SCM prior as generated. Its Reg2Cls clamps every target at four inner sd
        before standardising: 3% of tables still pass four plain sd, but a context sd shift
        of 4 or more occurs in 2 of 14000 tables and 20 or more in none.
    A1  the same tables, half with a GPD tail whose shape follows one feature, 30% with one
        context row rescaled to an sd shift in [4, 50] (`common/prior_arms.py`).
    B0  context target standardised by mean and sd, as shipped.
    B1  by median and IQR / 1.349, in the edges, the loss and the adapter alike.
        Both encodings cap the encoded target at +-1e4, which fp16 needs: the cap touches
        0.7% of A0B1 contexts, 2.6% of A1B1 contexts and one table of 56000 under A0B0.

WHAT IS MEASURED
----------------
The treatment of `clip_context.py`: a GPD generator of constant xi, a context of N_TRAIN
rows, and at a target sd shift above 1 one added row at the centre of x sized by
`metrics.y0_for_sd_shift`. The test rows are always clean. Per cell: the implied xi
(location-invariant and raw ratio), pinball loss at 0.5, 0.9, 0.99 and 0.999, the median
predicted quantiles, and the borders of the arm's grid, mapped back through its encoding,
that lie over the range of the clean context. Contexts are 768 rows, inside the 512 to 896
the models were pre-trained on.

The free unit is the pre-training seed crossed with the generator seed (rule 6). Arms are
compared only on cells where all arms exist.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Q1, the gate. Under A0B0, a shift of 20 or more lowers the median predicted Q(0.99) against
the clean context of the same seed in at least 8 of 10 generator seeds per pre-training
seed, at both xi, and moves the implied xi at a shift of 50 down by at least 0.2 or below
zero. If A0B0 does not reproduce the failure, the small model is not a valid proxy: that is
the result, and Q2 to Q5 are not interpreted.

Q2. A1B0 loses less of Q(0.99) at shifts 20 and 50 than A0B0, by at least half of A0B0's
loss, paired over cells; its pinball at 0.99 on a clean context is at most 5% worse than
A0B0's.

Q3. A1B0 captures a larger share of the conditional tail shape than A0B0 in
`shape_of_x.py` with the same seeds.

Q4. A0B1 keeps the borders over the data within 10% of its clean-context count at every
shift, where A0B0 loses more than half at a shift of 50; A1B1 has the lowest pinball at
0.99 under shifts 20 and 50 of the four arms.

Q5. No arm reaches the shape share of the GBM control in `shape_of_x.py`.

    MODELS=nanoTabPFN-A0B0-s1 SEEDS=2 SD_SHIFTS=1,50 \\
      venv-tfmp/Scripts/python.exe -u experiments/h3_repair/prior_pretraining_eval.py
"""
import numpy as np
import pandas as pd

from common import env, generator, leverage, metrics, models, paths, quiet, runner, tables
from common.adapters import nanotabpfn

quiet.silence()

XI = env.floats("XI", [0.7, 0.9])
SD_SHIFTS = env.floats("SD_SHIFTS", [1, 4, 20, 50])
SEEDS = env.seeds(env.integer("SEEDS", 10))
N_TRAIN = env.integer("N_TRAIN", 768)
N_TEST = env.integer("N_TEST", 900)
MODELS = env.models(",".join(f"nanoTabPFN-{a}-s{s}" for s in (1, 2)
                             for a in ("A0B0", "A0B1", "A1B0", "A1B1")))
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
OUTPUT = env.text("OUTPUT", "prior_pretraining_eval.csv")

COLUMNS = ["model", "arm", "pretrain_seed", "xi", "sd_shift_target", "sd_shift", "seed",
           "n_train", "n_test", "borders_in_data", "xi_implied", "xi_ratio",
           "q50", "q90", "q99", "q999", "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY = ["model", "xi", "sd_shift_target", "seed", "n_train", "n_test"]


def measure(cell):
    xi, target, seed = cell["xi"], cell["sd_shift_target"], cell["seed"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
    test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)

    X, y, shift = train.X, train.y, 1.0
    if target > 1.0:
        y0 = metrics.y0_for_sd_shift(train.y, target)
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {target} unreachable on this seed")
        X, y = leverage.add_row(train.X, train.y, y0)
        shift = float(metrics.sd_shift(y))

    q, borders = nanotabpfn.predict(nanotabpfn.checkpoint_path(cell["model"]), X, y,
                                    test.X, LEVELS)
    r = dict(sd_shift=shift,
             borders_in_data=int(((borders >= train.y.min())
                                  & (borders <= train.y.max())).sum()),
             xi_implied=metrics.implied_xi(q[:, 2], q[:, 1], q[:, 0]),
             xi_ratio=metrics.implied_xi(q[:, 2], q[:, 1]))
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        r[f"q{t}"] = float(np.median(q[:, i]))
        r[f"pb{t}"] = metrics.pinball(test.y, q[:, i], a)
    return r


def main():
    for m in MODELS:
        if m not in models.NANOTABPFN:
            raise SystemExit(f"{m} is not a nanoTabPFN arm")
    cells = [dict(model=m, arm=m.split("-")[1], pretrain_seed=int(m.rsplit("-s", 1)[1]),
                  xi=xi, sd_shift_target=target, seed=seed, n_train=N_TRAIN, n_test=N_TEST)
             for m in MODELS for xi in XI for target in SD_SHIFTS for seed in SEEDS]
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"borders {r['borders_in_data']:>4}  xi "
                                        f"{r['xi_implied']:+.3f}  q99 {r['q99']:.3f}  "
                                        f"pb99 {r['pb99']:.4f}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    unit = ["model", "xi", "seed"]
    clean = d[d.sd_shift_target == 1].set_index(unit)[["q99", "borders_in_data"]]
    j = d.join(clean, on=unit, rsuffix="_clean")
    j["q99_loss"] = 1.0 - j.q99 / j.q99_clean
    for col in ("borders_in_data", "xi_implied", "q99_loss", "pb99", "pb999"):
        print(f"\n=== {col}, median by arm ===")
        print(j.pivot_table(index=["xi", "sd_shift_target"], columns="arm", values=col,
                            aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    main()
