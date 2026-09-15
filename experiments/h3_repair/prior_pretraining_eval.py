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

OUTCOME, 15.9.2026: THE GATE FAILS ON BOTH PRE-TRAINING SEEDS
-------------------------------------------------------------
Only arm A0B0 was evaluated, on pre-training seeds 1 and 2 (7000 steps of 8 tables each,
1000 buckets), 10 generator seeds, xi 0.7 and 0.9, 768 context rows. The rows are in
`results/h3_repair/prior_pretraining_eval.csv`. Seed 1 was read on the CPU at
OMP_NUM_THREADS=4 and seed 2 on the GPU; the effects below are factors of 12 and more,
where the thread count moved TabPFN's pinball at 0.99 by at most 4%.

Q1  FAILED. A shift of 20 or 50 lowers the median predicted Q(0.99) against the clean
    context in 0 of 10 generator seeds, for both pre-training seeds and both xi. It raises
    it instead, 12 to 16 times at a shift of 20 and 34 to 47 times at 50. The implied xi
    at a shift of 50 rises, from +0.76 to +1.56 on the clean context to +2.22 to +2.26,
    and falls by 0.2 or below zero in 0 of 10 seeds. The grid does stretch as in TabPFN:
    the borders over the data fall from 654 on a clean context to 170 at a shift of 50
    (xi 0.7; 612 to 200 at xi 0.9), the same for both seeds, since the edges and the
    encoding are the same. The small model reads the stretched grid as a heavier tail
    where TabPFN reads it as a lighter one. By the rule above it is not a valid proxy for
    this failure, and Q2 to Q5 are not interpreted.

    It is no proxy for part one either. In `shape_of_x.py` and `scale_of_x.py` at 768
    training rows, 20 seeds, one member, A0B0 reproduces -0.002 and -0.010 of the true
    shape slope and -0.002 and -0.001 of the true change in the conditional median: it
    ignores x. The GBM control at the same size keeps 0.31 of the shape and 0.85 of the
    scale. After 56000 tables the small model has learned a marginal distribution, not a
    conditional one.

Q2, Q3, Q4, Q5  NOT TESTED, because the gate failed. A1B0-s1 was stopped at step 250 of
    7000. A0B1-s1 had diverged before the gate was read (median loss 1.49 at step 3000,
    12.47 at 7000, only the decoder grown) with the encoded target capped at 1e4, so B1
    would have needed another fix before any comparison.

What this does not say: that the failure lies in the architecture rather than the prior.
A model this weak cannot separate the two. What it says: the direction of the grid effect
depends on what the network has learned to put into the stretched bars, so the question
needs a model that at least tracks the conditional scale, which this budget on an RTX 2060
did not reach.

    medians over xi 0.7 and 0.9 and 10 generator seeds
    arm       Q(0.99) at shift >= 20  implied xi at 50  clean pinball 0.99  shape share
    A0B0-s1   +3163%                  2.255             3.615               -0.002
    A0B0-s2   +3421%                  2.225             4.444               -0.010
    A0B1-s1   diverged                -                 -                   -
    A1B0-s1   stopped at step 250     -                 -                   -
    A1B1      not trained             -                 -                   -

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
