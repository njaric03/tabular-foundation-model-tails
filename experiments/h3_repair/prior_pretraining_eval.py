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

SECOND OUTCOME, 17.9.2026: A CURRICULUM THAT LEARNS, AND THE PRIOR DECIDES THE COLLAPSE
-------------------------------------------------------------------------------------
Every later run from scratch at more than 3 features or more than 50 rows stayed at the
loss of the context marginal (`data/tfmp_artifacts.json`). A model that learned at 50 rows
and 3 features was carried on instead: A0B0-s42 to 8 features (A0B0-s91, linear-task R2
+0.911 at 35 rows), then to 256 rows for 3755 steps of 16 tables in each arm
(`run/prior_pretraining_curriculum.sh`). Every arm starts from s91, so the arms differ only
in the prior and the encoding of that last stage. Seed 92 reads two 256-row shards, seed 93
the other two. Evaluation at N_TRAIN=200, inside the 128 to 223 rows of pre-training.
Learning check (linear task, 200 rows): A0B0 +0.950 and +0.948, A1B0 +0.939 and +0.935;
A0B1-s92 and A1B1-s92 end at NaN and learn nothing. Rows in
`results/h3_repair/prior_pretraining_eval.csv` (s92) and `prior_pretraining_eval_curriculum_1.csv`
and `_2.csv`; shares in `shape_of_x_nanotabpfn_n200_curriculum.csv` and its scale twin.

Q1  HALF. A0B0-s92 holds: Q(0.99) below the clean value in 10, 10, 10 and 9 of 10 seeds at
    shifts 20 and 50 and both xi, falling to 2 to 4% of it, the implied xi at 50 down by
    2.50 to -0.06 in 20 of 20. A0B0-s93 holds at a shift of 50 (10 and 10 of 10) and for
    the implied xi (down by 2.09 in 20 of 20) but not at 20 (6 and 7 of 10). The borders
    over the data fall from 64 to 9, as in TabPFN.

Q2  HOLDS on both seeds. A0B0 loses a median 97% and 94% of Q(0.99) at shifts 20 and 50;
    A1B0 loses none and rises by 71% and 38%, at most half of A0B0's loss in 31 and 30 of 40
    cells. Median Q(0.99) over both xi, clean / 20 / 50: A0B0-s92 1991 / 105 / 29, A1B0-s92
    25 / 40 / 43, against true values near 35 (xi 0.7) and 69 (xi 0.9). Pinball at 0.99
    under shifts 20 and 50 is
    lower for A1B0 in 40 of 40 cells on each seed (sign test p = 9e-13); on the clean
    context it is 5.6 times lower, not 5% higher.

Q3  HOLDS on both seeds. Shape share at 200 rows, 20 seeds: A1B0 +1.093 and +1.071, A0B0
    +0.100 and -0.080. Scale share: A1B0 0.410 and 0.384, A0B0 0.324 and 0.356.

Q4  NOT TESTED. Both B1 arms reached NaN when the encoding changed at the last stage.

Q5  WRONG as written, and empty at this size. GBM at 200 rows has a shape share of -0.428
    (scale 0.782): 200 rows do not give a tree a usable Q(0.99), so no arm could fail to
    reach it. A1B0 exceeds it at +1.09.

What this says. In a model of TabPFN's architecture that learns, trained identically up
to its last stage, one leverage row collapses the upper tail under the standard prior and
does not under a prior with heavy tails and contaminated contexts, while the grid stretches
exactly the same in both (64 to 9 borders over the data). Stretching the grid is not
enough for the collapse; what the network has learned to put in the stretched bars
decides it. That supports the reading of `clip_context_real.py`, that EXAONE and TabICLv2
differ by what they learned and not by their pipeline, and it matches the direction of the
TabICLv2 fine-tuning (`tabicl_prior_finetune.py`), where the same prior helped a little.

What this does not say. A1 adds two things at once, a GPD tail whose shape follows a
feature and a contaminated context row, and the generator of this experiment is a GPD
with a covariate-dependent shape: part or all of A1B0's gain may be that its prior now
resembles the test family, which also explains its clean calibration and shape share. An
arm with the tail alone and one with the contamination alone would separate the two. The
arms share their first two stages, the contexts are 200 rows against the thesis's 768 to
2000, the model is small, and A0B0's clean tail is far off (Q(0.99) near 1900 against a
true 35 to 69), so the size of the effect does not transfer to the production models.

THIRD OUTCOME, 17.9.2026: THE TAIL ALONE STOPS THE COLLAPSE, THE CONTAMINATION ALONE ONCE
----------------------------------------------------------------------------------------
Arms ATB0 (A1's heavy tail alone) and ACB0 (its contaminated context alone), streams 3 and
4 of `run/prior_pretraining_curriculum.sh`, as A1B0 in every other respect and with A1's
own draws on the same tables: at step 200 a tail in 0.4913 of seed 92's tables and a
contamination in 0.2981, the shares of A1B0-s92. The predictions R0 to R4 were committed
in e7b763e, before any step was read. Rows in `prior_pretraining_eval_curriculum_3.csv`
and `_4.csv`; shares appended to the curriculum shape and scale files. "Meets Q2" is Q2's
criterion as it was read for A1B0, a reading that reproduces A1B0's 31 and 30 of 40.

               meets Q2   median Q(0.99)         clean pinball    shape share
               (of 40)    clean / 20 / 50        / A0B0's
    A0B0-s92       -      1991 /  105 /  29       1                +0.100
    A1B0-s92      31        25 /   40 /  43       0.19             +1.093
    ATB0-s92      35        20 /   77 / 121       0.16             +1.419
    ACB0-s92      14       189 /   37 /  23       0.50             +0.673
    A0B0-s93       -      1860 /  202 /  79       1                -0.080
    A1B0-s93      30        25 /   36 /  30       0.22             +1.071
    ATB0-s93      33        20 /  147 / 279       0.18             +0.702
    ACB0-s93      24        45 /   58 /  33       0.39             +0.562

R0  HOLDS. Linear-task R2 at 200 rows: ATB0 +0.947 and +0.944, ACB0 +0.950 and +0.944.

R1  WRONG. ACB0 meets Q2 on seed 93 (24 of 40) but not on 92 (14 of 40), where its Q(0.99)
    still falls by a median 77% and 85% at shifts 20 and 50. Its pinball at 0.99 under
    those shifts is lower than A0B0's in 40 of 40 cells on both seeds.

R2  WRONG. ATB0 meets Q2 on both seeds (35 and 33 of 40), although its prior showed no
    context row at a shift of 20 in the 2000 tables measured before the run.

R3  HALF. ATB0's shape share is above 0.5 (+1.419 and +0.702), but so is ACB0's (+0.673 and
    +0.562). The scale shares do not separate the arms (0.307 to 0.411; A0B0 0.324, 0.356).

R4  HALF. ATB0's clean pinball is 0.16 and 0.18 of A0B0's, but ACB0's is at most half too,
    0.495 and 0.386 (ratio of the medians over 20 cells).

The reading fixed in advance, seed by seed: on 92 the tail meets Q2 and the contamination
does not, "the resemblance carries it"; on 93 both do, "either treatment suffices". Only
the tail meets it on both seeds, so the claim narrows. In this experiment a prior with a
GPD tail stops the collapse, and that tail cannot be told apart from the GPD family of the
test. That contaminated contexts in the prior protect against a leverage row is not shown.

Not predicted, read after the outcome. Q2's criterion counts a rise as no loss, and the
tail alone meets it by overshooting: its median Q(0.99) climbs to 121 and 279 at a shift of
50 against a true 35 to 69, and its implied xi at 50 rises (+0.09 and +0.66), where A1B0's
moves by -0.12 and -0.33. Shown heavy tails but never a contaminated row, the network
reads the leverage row as a heavier tail. Under shifts 20 and 50 the pinball at 0.99 of
A1B0 is lower than ATB0's in 29 and 34 of 40 cells (median ratio of ATB0 to A1B0 1.06 and
1.49) and lower than ACB0's in 33 and 32 (1.03 and 1.05). Neither treatment alone does as
well under leverage as the two together, while on the clean context the tail alone is a
little better than both (lower than A1B0 in 13 and 17 of 20). The borders over the data
fall from 64 to 9 in all four arms on both seeds: the grid stretches the same under four
priors, and what the network puts in the stretched bars differs.

FOURTH OUTCOME, 17.9.2026: OTHER TAIL FAMILIES, AND WHAT Q2 MEASURES
-------------------------------------------------------------------
With FAMILIES=frechet,burr,studentt (`prior_pretraining_eval_families.csv`), the same
design with a tail of the same index and another body: F0 and F2 hold, F1 is wrong (A1B0
meets Q2 on 2 of 6 family-seed pairs, ATB0 on 3). A0B0 puts the clean Q(0.99) about 40
times too high on the families and on the GPD alike, so its fall under leverage is partly a
fall towards the truth, and Q2's loss against the clean prediction is not a measure of
collapse. By pinball at 0.99, one leverage row at a shift of 20 or 50 makes A0B0 4.4 to 7.4
times worse than on its clean context on the GPD and the families, and none of A1B0, ATB0
and ACB0 worse except ATB0-s93 at 50 on the GPD (1.54). The full outcome, with five real
claim tables, is in `prior_pretraining_real.py` and PREREGISTRATION.md.

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
# The tail family of the generator, one of `generator.FAMILIES`: same tail index and scale,
# another body. gpd draws through `generator.gpd`, as every row did before this knob.
FAMILIES = env.names("FAMILIES", "gpd")
MODELS = env.models(",".join(f"nanoTabPFN-{a}-s{s}" for s in (1, 2)
                             for a in ("A0B0", "A0B1", "A1B0", "A1B1")))
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
OUTPUT = env.text("OUTPUT", "prior_pretraining_eval.csv")

COLUMNS = ["model", "arm", "pretrain_seed", "family", "xi", "sd_shift_target", "sd_shift",
           "seed", "n_train", "n_test", "borders_in_data", "xi_implied", "xi_ratio",
           "q50", "q90", "q99", "q999", "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY = ["model", "family", "xi", "sd_shift_target", "seed", "n_train", "n_test"]


def measure(cell):
    xi, target, seed = cell["xi"], cell["sd_shift_target"], cell["seed"]
    rng = np.random.default_rng(seed)
    if cell["family"] == "gpd":
        train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
        test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)
    else:
        train = generator.family(cell["family"], cell["n_train"], rng, xi=xi)
        test = generator.family(cell["family"], cell["n_test"], rng, xi=xi)

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
    for f in FAMILIES:
        if f not in generator.FAMILIES:
            raise SystemExit(f"{f} is not one of {generator.FAMILIES}")
    cells = [dict(model=m, arm=m.split("-")[1], pretrain_seed=int(m.rsplit("-s", 1)[1]),
                  family=f, xi=xi, sd_shift_target=target, seed=seed, n_train=N_TRAIN,
                  n_test=N_TEST)
             for f in FAMILIES for m in MODELS for xi in XI for target in SD_SHIFTS
             for seed in SEEDS]
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
    if "family" not in d:
        d["family"] = "gpd"                 # written before the knob, when gpd was the only one
    unit = ["model", "family", "xi", "seed"]
    clean = d[d.sd_shift_target == 1].set_index(unit)[["q99", "borders_in_data"]]
    j = d.join(clean, on=unit, rsuffix="_clean")
    j["q99_loss"] = 1.0 - j.q99 / j.q99_clean
    for col in ("borders_in_data", "xi_implied", "q99_loss", "pb99", "pb999"):
        print(f"\n=== {col}, median by arm ===")
        print(j.pivot_table(index=["family", "xi", "sd_shift_target"], columns="arm",
                            values=col, aggfunc="median").round(4).to_string())


if __name__ == "__main__":
    main()
