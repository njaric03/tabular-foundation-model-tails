# -*- coding: utf-8 -*-
"""The pre-training arms on real claims, under the leverage row of the synthetic test.

On the GPD generator of `prior_pretraining_eval.py` the heavy tail of the prior alone
(ATB0) kept Q(0.99) under one leverage row, but that prior is a GPD too. Here the context
is real and only the perturbation is synthetic: neither the tail nor the body was chosen.

Per dataset, REPEATS subsamples of N_TRAIN + N_TEST positive rows are drawn with
`leverage.subsamples` from seed 31337 in the lowest sd-shift bin (below 1.2), so the
context carries no leverage row of its own. At each shift above 1 one row is added at the
context's feature mean with the target `metrics.y0_for_sd_shift` gives, the construction of
`prior_pretraining_eval.py`, whose generator has its feature mean at zero; shift 1 is the
clean context. Pinball is on the N_TEST held-out rows of the same subsample, which the row
never touches. nanoTabPFN normalises every feature by the context, so the added row sits
at zero in what the network sees.

The datasets are the claim tables with at most 16 features, the most any arm saw in
pre-training: freMTPL2sev and the four of `data/external_selected.txt` that qualify.

The predictions written before the run, F0 to F2 for the tail families of
`prior_pretraining_eval.py` and R0 to R3 here, are in PREREGISTRATION.md.

OUTCOME, 17.9.2026: THE CRITERION FAILS BEFORE THE PRIORS DO
-----------------------------------------------------------
The predictions are those committed in 77f3176. Rows in
`results/h3_repair/prior_pretraining_eval_families.csv` (1920) and
`prior_pretraining_real.csv` (3200); no cell failed. "Meets Q2" counts, of 40 cells, those
where the loss of Q(0.99) at shifts 20 and 50 is at most half of A0B0's; it takes more
than 20.

                          seed   A0B0 falls    meets Q2:  A1B0   ATB0   ACB0
    families  burr         92        39                    19     12     22
              frechet      92        39                    25     20     22
              studentt     92        35                    18     22     12
              burr         93        32                    18     18     24
              frechet      93        32                    19     22     24
              studentt     93        29                    22     25     16
    claims    AutoClaims   92        40                     0      0      0
              ausprivauto  92        39                     0     10      0
              beMTPL97     92        38                     4      6      0
              freMTPL2sev  92        40                     0      0      0
              norauto      92        40                     0      0      0
              AutoClaims   93        20                     0      0      0
              ausprivauto  93        20                     0      5      0
              beMTPL97     93        25                     4      8      9
              freMTPL2sev  93        21                     0      0      0
              norauto      93        20                     0      0      0

F0  HOLDS on every family and seed (29 to 39 of 40).
F1  WRONG. A1B0 meets Q2 on 2 of the 6 family-seed pairs and ATB0 on 3; neither on burr.
    ACB0, not predicted, meets it on 4.
F2  HOLDS. Pinball at 0.99 under shifts 20 and 50 is lower than A0B0's in 40 of 40 cells for
    each of the three arms, on every family and seed.

R0  HOLDS on all five tables for seed 92, and on seed 93 only on beMTPL97 and freMTPL2sev.
    On the other three, A0B0's Q(0.99) falls at a shift of 50 only (20 of 40); at 20 it
    rises.
R1  HOLDS. ATB0 meets Q2 on none of the passing tables (0 of 5, 0 of 2).
R2  WRONG. A1B0 meets it on none either (0 of 5, 0 of 2), and nor does ACB0.
R3  HALF. ATB0's pinball under leverage is lower than A1B0's in fewer than 20 of 40 cells on
    2 of 5 tables for seed 92 (12 to 27 of 40, median ratio 0.98 to 1.06) and on 2 of 2
    for seed 93 (8 and 17).

By the reading fixed in advance, F1 failing leaves the ablation's narrowing as written.

Not predicted, read after the outcome, and it changes what Q1, Q2, F1 and R1 to R3 can
mean. Their loss is measured against each model's own clean Q(0.99), and that is far off.
Median over cells of the predicted Q(0.99) over the true one (GPD and families: the true
conditional quantile; claims: the empirical 0.99 quantile of the 1000 held-out rows), and
the median pinball at 0.99 over the same model's clean pinball, seed 92 / 93:

                    Q(0.99) / true                            pinball / clean pinball
                    clean         shift 20      shift 50      shift 20      shift 50
    GPD       A0B0  42.4 / 40.0   2.30 / 5.51   0.71 / 1.68   4.36 / 5.95   7.02 / 6.18
              A1B0  0.58 / 0.54   0.79 / 0.78   0.87 / 0.62   0.66 / 0.62   0.67 / 0.63
              ATB0  0.51 / 0.52   1.70 / 3.23   2.38 / 6.01   0.83 / 0.97   0.84 / 1.54
              ACB0  4.65 / 0.95   0.75 / 1.14   0.46 / 0.66   0.23 / 0.52   0.25 / 0.41
    families  A0B0  41.3 / 38.9   2.42 / 12.1   0.76 / 2.15   4.79 / 6.09   7.36 / 6.71
              A1B0  0.94 / 2.86   0.84 / 0.85   0.90 / 0.72   0.29 / 0.31   0.28 / 0.31
              ATB0  7.57 / 5.35   1.73 / 3.28   2.48 / 6.14   0.29 / 0.33   0.34 / 0.66
              ACB0  2.92 / 0.99   0.81 / 1.14   0.54 / 0.69   0.24 / 0.47   0.25 / 0.36
    claims    A0B0  34.1 / 36.8   2.21 / 235    0.59 / 0.72   3.50 / 7.38   2.85 / 4.88
              A1B0  32.2 / 32.6   1.17 / 1.15   1.13 / 1.00   0.06 / 0.05   0.06 / 0.05
              ATB0  31.1 / 30.1   1.46 / 2.26   0.95 / 1.60   0.07 / 0.09   0.08 / 0.08
              ACB0  32.4 / 30.8   0.80 / 0.90   0.67 / 0.84   0.06 / 0.06   0.06 / 0.06

A0B0 puts the clean Q(0.99) about 40 times too high on every test set, so its "collapse"
under leverage is a fall towards the truth: 0.59 to 2.15 of it at a shift of 50, 2.2 to 12
at 20, and 235 once. On the claims every arm is 30 times too high, and Q2 scores each fall
towards the truth as a failure. Relative Q(0.99) does not measure a collapse when the clean
prediction is off by more than the collapse. Pinball does, and by pinball the picture is
the same on all three test sets and both seeds. One leverage row at a shift of 20 or 50
makes A0B0's pinball at 0.99 2.9 to 7.4 times worse than on its clean context. Under each
of the three modified priors it gets no worse, 0.05 to 0.97 of the clean pinball, in 35 of
the 36 arm-seed-shift-test-set cells of the table; the exception is ATB0-s93 at 50 on the
GPD (1.54). Under leverage, A1B0's Q(0.99) lands at 0.62 to 1.17 of the truth, ACB0's at
0.46 to 1.14, ATB0's at 0.95 to 6.14: the tail alone overshoots on every test set.

What carries over from the GPD, then, is the leverage response: a prior with heavy tails, a
contaminated context row or both keeps the pinball at 0.99 from degrading under one
leverage row, on tails of another body and on five claim tables nobody built to resemble
the priors. What does not carry over is the clean calibration. A1B0's clean Q(0.99) is 0.54
to 0.58 of the truth on the GPD, 0.94 and 2.86 on the families and 32 on the claims, as bad
as A0B0's there; its clean pinball is 0.19 and 0.22 of A0B0's on the GPD, 0.30 to 0.42 on
the families and 0.65 to 1.20 on the claims. The resemblance the ablation feared explains
the clean calibration, and on this evidence not the response to leverage.

Limits. Five tables are five units per seed, three families three; the 40 cells of one
table are not independent. The pinball ratios are medians over cells without a test, and
the empirical Q(0.99) of 1000 held-out claims is itself noisy. On the claims every arm is
calibrated so badly that a leverage row improves it, so "no worse under leverage" is a
weaker statement there than on the GPD. The models are small, the contexts are 200 rows,
and the clean miscalibration of every arm off the GPD says the 256-row stage did not teach
a usable tail for real claims.

    DATASETS=AutoClaims MODELS=nanoTabPFN-A0B0-s92 REPEATS=2 SD_SHIFTS=1,50 \\
      venv-tfmp/Scripts/python.exe -u experiments/h3_repair/prior_pretraining_real.py
"""
import numpy as np
import pandas as pd

from common import datasets, env, leverage, metrics, models, paths, quiet, runner, tables
from common.adapters import nanotabpfn

quiet.silence()

DATASETS = env.names("DATASETS", "freMTPL2sev,beMTPL97,ausprivauto0405,norauto,AutoClaims")
MODELS = env.models(",".join(f"nanoTabPFN-{a}-s{s}" for s in (92, 93)
                             for a in ("A0B0", "A1B0", "ATB0", "ACB0")))
SD_SHIFTS = env.floats("SD_SHIFTS", [1, 4, 20, 50])
REPEATS = env.integer("REPEATS", 20)
N_TRAIN = env.integer("N_TRAIN", 200)
N_TEST = env.integer("N_TEST", 1000)
MAX_ATTEMPTS = env.integer("MAX_ATTEMPTS", 4000)
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
OUTPUT = env.text("OUTPUT", "prior_pretraining_real.csv")

COLUMNS = ["dataset", "model", "arm", "pretrain_seed", "repeat", "sd_shift_target",
           "sd_shift", "natural_shift", "n_train", "n_test", "borders_in_data",
           "xi_implied", "xi_ratio", "q50", "q90", "q99", "q999",
           "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY = ["dataset", "model", "repeat", "sd_shift_target", "n_train", "n_test"]


def contexts(name):
    """[(X, y, Xte, yte, natural shift), ...] of one dataset, the same on every run."""
    X, y = datasets.load(name)
    return leverage.clean_subsamples(X, y, N_TRAIN, N_TEST, REPEATS, MAX_ATTEMPTS)


def measure(cell):
    X, y, Xte, yte, _ = cell["_context"]
    target, shift = cell["sd_shift_target"], 1.0
    Xc, yc = X, y
    if target > 1.0:
        y0 = metrics.y0_for_sd_shift(y, target)
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {target} unreachable on this subsample")
        Xc, yc = leverage.add_row(X, y, y0, x0=X.mean(axis=0))
        shift = float(metrics.sd_shift(yc))
    q, borders = nanotabpfn.predict(nanotabpfn.checkpoint_path(cell["model"]), Xc, yc, Xte,
                                    LEVELS)
    r = dict(sd_shift=shift,
             borders_in_data=int(((borders >= y.min()) & (borders <= y.max())).sum()),
             xi_implied=metrics.implied_xi(q[:, 2], q[:, 1], q[:, 0]),
             xi_ratio=metrics.implied_xi(q[:, 2], q[:, 1]))
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        r[f"q{t}"] = float(np.median(q[:, i]))
        r[f"pb{t}"] = metrics.pinball(yte, q[:, i], a)
    return r


def main():
    for m in MODELS:
        if m not in models.NANOTABPFN:
            raise SystemExit(f"{m} is not a nanoTabPFN arm")
    seconds = 0.0
    for name in DATASETS:
        ctx = contexts(name)
        cells = [dict(dataset=name, model=m, arm=m.split("-")[1],
                      pretrain_seed=int(m.rsplit("-s", 1)[1]), repeat=j,
                      sd_shift_target=target, natural_shift=ctx[j][4], n_train=N_TRAIN,
                      n_test=N_TEST, _context=ctx[j])
                 for m in MODELS for j in range(REPEATS) for target in SD_SHIFTS]
        seconds += runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                              show=lambda r: f"q99 {r['q99']:.4g}  pb99 {r['pb99']:.4g}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    unit = ["dataset", "model", "repeat"]
    clean = d[d.sd_shift_target == 1].set_index(unit)["q99"]
    j = d.join(clean.rename("q99_clean"), on=unit)
    j["q99_loss"] = 1.0 - j.q99 / j.q99_clean
    for col in ("q99_loss", "pb99"):
        print(f"\n=== {col}, median by arm ===")
        print(j.pivot_table(index=["dataset", "sd_shift_target"],
                            columns=["pretrain_seed", "arm"], values=col,
                            aggfunc="median").round(3).to_string())


if __name__ == "__main__":
    main()
