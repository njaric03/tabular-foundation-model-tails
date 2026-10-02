# -*- coding: utf-8 -*-
"""What the frozen cap costs a legitimate high-risk segment, where the truth is known.

A cap cannot tell an error from a genuine extreme. On real tables the two cannot be
separated either, so the cost of capping genuine extremes is measured here, on the GPD
generator with a constant xi, plus a segment: a binary feature g, 1 with probability
SHARE, that multiplies the conditional scale by M. The model sees g, so the segment is
learnable and its larger values are signal, not noise. M = 1 is the control: g is then a
feature with no effect.

    clean       the context as drawn
    shift 20    one added row at the centre of x (g = 0) sized to an sd shift of 20, the
                treatment of `clip_context.py`

Arms raw and clip_200 (median + 200 robust sd of the context), the cap frozen after the
runs of 12.9.2026. The loss is the expected pinball under the true conditional law,
computed in closed form for every test row, so it carries no Monte Carlo noise from the
test targets:

    E rho_tau(Y - q) = E(Y - q)+ - (1 - tau)(E Y - q),
    E(Y - q)+ = m (1 + xi q / sigma)^(1 - 1/xi) for q >= 0,  m = sigma / (1 - xi)

The pinball on the drawn test targets is kept beside it. Both are reported for the whole
test set and for the rows in and out of the segment, with the clipped context rows
counted by segment. Seeds are 7500, 8500, ..., which no earlier run of the generator used.

PREDICTIONS WRITTEN BEFORE THE RUN, 1.10.2026
---------------------------------------------
Expected pinball, paired over seeds, foundation models (TabPFN-V3, TabICLv2) unless said.

S1, the control. At M = 1 on a clean context clip_200 changes the expected pinball at 0.99
over the whole test set by less than 2% on the median for each foundation model.

S2, the benefit replicates. At M = 1 under the shift of 20 clip_200 lowers the expected
pinball at 0.99 in at least 18 of 20 seeds for each foundation model.

S3, the segment pays. At M = 20 on a clean context:
  (a) at least half of the clipped context rows are segment rows, on the median;
  (b) the segment's expected pinball at 0.999 is higher under clip_200 than under raw in at
      least 14 of 20 seeds for each foundation model;
  (c) for each foundation model the median relative change of the segment's expected
      pinball at 0.99 is larger than that of the whole test set.

S4, the benefit survives the segment. At M = 20 under the shift of 20 clip_200 lowers the
expected pinball at 0.99 over the whole test set in at least 15 of 20 seeds for TabPFN-V3.

GBM is reported without a prediction. M = 5 lies between and carries none either.

OUTCOME, 1.10.2026
------------------
S1 holds (-0.17% and -0.31%). S2 half: TabPFN-V3 20 of 20, TabICLv2 12 of 20 and +7.4% at
0.999. S3 holds: 3 of 4 clipped rows are segment rows, the segment's 0.999 loss rises in 18
and 20 of 20 seeds (+5.3%, +15.9%), and at 0.99 the segment pays more than the whole set.
S4 holds, 20 of 20. Whether the cap pays depends on what the error does to the raw model:
it hurts TabPFN-V3 (+10% at 0.99), which the cap repairs at a small cost to the segment;
it lifts TabICLv2's too-low clean Q(0.999), and the cap takes the lift away (+54% at 0.999
at M = 20 under the error). Full outcome in PREREGISTRATION.md.

    MODELS=GBM SEEDS=2 python -u experiments/h3_repair/legit_segment.py
"""
import numpy as np
import pandas as pd

from common import (
    env,
    generator,
    leverage,
    metrics,
    models,
    paths,
    quiet,
    runner,
    tables,
    transforms,
)

quiet.silence()

MODELS = env.models("TabPFN-V3,TabICLv2,GBM")
XI = env.floats("XI", [0.7])
SD_SHIFTS = env.floats("SD_SHIFTS", [1, 20])
CLIP_C = env.floats("CLIP_C", [200])
SEEDS = [s + 500 for s in env.seeds(env.integer("SEEDS", 20))]
N_TRAIN = env.integer("N_TRAIN", 2000)
N_TEST = env.integer("N_TEST", 5000)
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "legit_segment.csv")
# Not knobs: the design of the segment. Both are columns.
SCALES = (1.0, 5.0, 20.0)
SHARE = 0.05
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
PARTS = ("all", "seg", "out")
ARMS = [("raw", 0.0)] + [("clip", c) for c in CLIP_C]

COLUMNS = (["model", "xi", "seg_scale", "seg_share", "sd_shift_target", "variant", "clip_c",
            "seed", "n_est", "n_train", "n_test", "sd_shift", "natural_shift", "n_seg_context",
            "n_seg_test", "n_clipped", "n_clipped_seg"]
           + [f"epb{t}_{p}" for p in PARTS for t in TAGS]
           + [f"pb{t}_{p}" for p in PARTS for t in TAGS]
           + [f"qratio{t}_{p}" for p in ("seg", "out") for t in ("99", "999")]
           + ["seconds", "reason"])
KEY = ["model", "xi", "seg_scale", "seg_share", "sd_shift_target", "variant", "clip_c",
       "seed", "n_est", "n_train", "n_test"]


def expected_pinball(q, sigma, xi, tau):
    """E rho_tau(Y - q) per row for Y ~ GPD(0, sigma, xi), 0 < xi < 1."""
    q, sigma = np.asarray(q, float), np.asarray(sigma, float)
    m = sigma / (1.0 - xi)
    above = m * (1.0 + xi * np.maximum(q, 0.0) / sigma) ** (1.0 - 1.0 / xi)
    plus = np.where(q >= 0, above, m - q)
    return plus - (1.0 - tau) * (m - q)


def design(cell):
    """Context, test and the true scales, the same base draws for every M of a seed."""
    xi, seed, M = cell["xi"], cell["seed"], cell["seg_scale"]
    rng = np.random.default_rng(seed)
    train = generator.gpd(cell["n_train"], rng, xi=xi, clip=True)
    test = generator.gpd(cell["n_test"], rng, xi=xi, clip=True)
    seg = np.random.default_rng([seed, 1])
    g_tr = (seg.random(cell["n_train"]) < cell["seg_share"]).astype(float)
    g_te = (seg.random(cell["n_test"]) < cell["seg_share"]).astype(float)
    X = np.column_stack([train.X, g_tr])
    Xte = np.column_stack([test.X, g_te])
    y = train.y * np.where(g_tr > 0, M, 1.0)
    yte = test.y * np.where(g_te > 0, M, 1.0)
    sigma_te = test.s * np.where(g_te > 0, M, 1.0)
    return X, y, g_tr, Xte, yte, g_te, sigma_te


def measure(cell):
    X, y, g, Xte, yte, g_te, sigma = design(cell)
    natural = float(metrics.sd_shift(y))
    if cell["sd_shift_target"] > 1.0:
        y0 = metrics.y0_for_sd_shift(y, cell["sd_shift_target"])
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {cell['sd_shift_target']} unreachable on this seed")
        X, y = leverage.add_row(X, y, y0)
        g = np.append(g, 0.0)
    y_fit, back, n = transforms.transform(cell["variant"], cell["clip_c"], y)
    q = back(models.quantiles(cell["model"], X, y_fit, Xte, seed=cell["seed"],
                              levels=LEVELS, n_est=cell["n_est"]))
    xi = cell["xi"]
    parts = {"all": np.ones(len(yte), bool), "seg": g_te > 0, "out": g_te == 0}
    r = dict(sd_shift=float(metrics.sd_shift(y)), natural_shift=natural,
             n_seg_context=int(g.sum()), n_seg_test=int(parts["seg"].sum()),
             n_clipped=n, n_clipped_seg=int(((y_fit < y) & (g > 0)).sum()))
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        e = expected_pinball(q[:, i], sigma, xi, a)
        for p, rows in parts.items():
            r[f"epb{t}_{p}"] = float(e[rows].mean())
            r[f"pb{t}_{p}"] = metrics.pinball(yte[rows], q[rows, i], a)
    for i, (a, t) in ((2, (0.99, "99")), (3, (0.999, "999"))):
        truth = sigma * metrics.gpd_quantile(a, xi)
        for p in ("seg", "out"):
            r[f"qratio{t}_{p}"] = float(np.median(q[parts[p], i] / truth[parts[p]]))
    return r


def main():
    cells = [dict(model=m, xi=xi, seg_scale=M, seg_share=SHARE, sd_shift_target=s,
                  variant=v, clip_c=c, seed=seed, n_est=N_EST, n_train=N_TRAIN,
                  n_test=N_TEST)
             for m in MODELS for xi in XI for M in SCALES for s in SD_SHIFTS
             for seed in SEEDS for v, c in ARMS]
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"epb99 {r['epb99_all']:.4f}  seg {r['epb99_seg']:.4f}"
                                        f"  epb999 seg {r['epb999_seg']:.3f}"
                                        f"  clipped {r['n_clipped']} ({r['n_clipped_seg']} seg)")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    key = ["model", "xi", "seg_scale", "sd_shift_target", "seed"]
    raw = d[d.variant == "raw"].set_index(key)
    cap = d[d.variant == "clip"].set_index(key)
    cols = ["epb99_all", "epb99_seg", "epb999_all", "epb999_seg", "epb99_out"]
    rel = (cap[cols] / raw[cols] - 1).dropna()
    by = ["model", "seg_scale", "sd_shift_target"]
    print("\n=== clip_200 against raw, expected pinball, median over seeds ===")
    print(rel.groupby(level=by).median().round(4).to_string())
    print("\n=== seeds where clip_200 is lower than raw ===")
    print((rel < 0).groupby(level=by).sum().to_string())
    clipped = cap[["n_clipped", "n_clipped_seg"]].groupby(level=by).median()
    print("\n=== clipped context rows, median ===")
    print(clipped.to_string())
    ratio = d.groupby(["model", "seg_scale", "sd_shift_target", "variant"])[
        ["qratio99_seg", "qratio999_seg", "qratio99_out", "qratio999_out"]].median()
    print("\n=== predicted over true quantile, median ===")
    print(ratio.round(3).to_string())


if __name__ == "__main__":
    main()
