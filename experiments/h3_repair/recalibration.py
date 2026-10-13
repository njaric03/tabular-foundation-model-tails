# -*- coding: utf-8 -*-
"""Can a plain recalibration on clean held-out outcomes replace or complete the cap?

`legit_segment.py` found that clip_200 repairs TabPFN-V3 under a leverage row, costs
TabICLv2, and that its cost falls on a legitimate high-risk segment. A natural objection
is that both models are biased on a clean context already, and that a recalibration of
their quantiles would remove the bias whether or not the context was capped. This run
measures what one global correction does, given a clean calibration sample the size of
the context. That sample is new information, so the result says what can be repaired when
such a sample exists, not that the bias is removed for free.

Design of `legit_segment.py` (GPD at xi = 0.7, a visible segment of 5% rows with its scale
times M), with M in {1, 20}, clean and under an added row at an sd shift of 20, arms raw
and clip_200. For every fitted context the model also predicts N_CALIB clean calibration
rows drawn from the same law, and one factor per level is computed from them:

    multiplicative (primary)    c * q, c the q-weighted tau-quantile of y / q
    additive (secondary)        q + c, c the tau-quantile of y - q

Each is the exact pinball minimiser of its family on the calibration rows. Both are
applied to the test predictions of the same fit, so the six variants (raw, raw + mul,
raw + add, cap, cap + mul, cap + add) share every fit. The loss is the expected pinball
under the true law in closed form, beside the oracle's, for the whole test set and for the
rows in and out of the segment. Seeds are 7700, 8700, ..., unused before.

The level that counts is 0.99. At 0.999 the 2000 calibration rows hold about two
exceedances, so that level is reported without a prediction. With 2000 calibration rows a
correct model still carries about 2% of calibration noise at 0.99 (checked on the
generator with known biases), so differences under 2% are not read as effects.

PREDICTIONS WRITTEN BEFORE THE RUN, 2.10.2026
---------------------------------------------
Expected pinball at 0.99, multiplicative calibration unless said, paired over 20 seeds.

K1, calibration helps a biased model. For TabPFN-V3 and TabICLv2, raw + mul is lower than
raw over the whole test set in at least 15 of 20 seeds in each of the four (M, shift) cells.

K2, the cap still matters for TabPFN-V3 after calibration. At M = 1 under the shift of 20,
cap + mul is lower than raw + mul in at least 14 of 20 seeds, and the median relative gap
between them is smaller than the gap between cap and raw without calibration.

K3, a global correction leaves TabICLv2's segment behind. At M = 20 on a clean context,
after raw + mul the segment's excess over the oracle is larger than the excess outside the
segment in at least 15 of 20 seeds for TabICLv2.

K4, calibration does not remove the cap's cost to the segment. At M = 20 on a clean
context, cap + mul gives TabICLv2's segment a higher expected pinball than raw + mul in at
least 14 of 20 seeds.

K5, the additive form fails the segment by construction. At M = 20 on a clean context the
segment's excess over the oracle after raw + add is larger than after raw + mul in at least
15 of 20 seeds for each foundation model.

GBM is reported without a prediction, as is TabPFN-V3 in K3 and K4.

OUTCOME, 13.10.2026
-------------------
K1 fails (TabPFN-V3 17, 19, 9, 14 of 20; TabICLv2 12, 12, 12, 9). K2 holds (20 of 20, gap
-3.2% against -6.7%). K3 holds (16). K4 holds (17). K5 fails (14 and 10). Calibration takes
half of the cap's benefit to TabPFN-V3 and none of its cost to the segment. Full outcome in
PREREGISTRATION.md.

    MODELS=GBM SEEDS=2 python -u experiments/h3_repair/recalibration.py
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
SEEDS = [s + 700 for s in env.seeds(env.integer("SEEDS", 20))]
N_TRAIN = env.integer("N_TRAIN", 2000)
N_CALIB = env.integer("N_CALIB", 2000)
N_TEST = env.integer("N_TEST", 5000)
N_EST = env.integer("N_EST", 1)
OUTPUT = env.text("OUTPUT", "recalibration.csv")
# Not knobs: the design of the segment, as in legit_segment.py. Both are columns.
SCALES = (1.0, 20.0)
SHARE = 0.05
LEVELS = [0.5, 0.9, 0.99, 0.999]
CAL_LEVELS = ((2, 0.99, "99"), (3, 0.999, "999"))
PARTS = ("all", "seg", "out")
CALS = ("none", "mul", "add")
ARMS = [("raw", 0.0)] + [("clip", c) for c in CLIP_C]

COLUMNS = (["model", "xi", "seg_scale", "seg_share", "sd_shift_target", "variant", "clip_c",
            "seed", "n_est", "n_train", "n_calib", "n_test", "sd_shift", "n_clipped",
            "n_clipped_seg"]
           + [f"c_{c}{t}" for _, _, t in CAL_LEVELS for c in ("mul", "add")]
           + [f"epb{t}_{p}_{c}" for _, _, t in CAL_LEVELS for p in PARTS for c in CALS]
           + [f"epb{t}_{p}_oracle" for _, _, t in CAL_LEVELS for p in PARTS]
           + [f"exc{t}_{p}_{c}" for _, _, t in CAL_LEVELS for p in PARTS for c in CALS]
           + ["seconds", "reason"])
KEY = ["model", "xi", "seg_scale", "seg_share", "sd_shift_target", "variant", "clip_c",
       "seed", "n_est", "n_train", "n_calib", "n_test"]


def design(cell):
    """Context, calibration and test rows with their true scales; the same base draws for
    every M of a seed."""
    xi, seed, M = cell["xi"], cell["seed"], cell["seg_scale"]
    rng = np.random.default_rng(seed)
    seg = np.random.default_rng([seed, 1])
    out = []
    for n in (cell["n_train"], cell["n_calib"], cell["n_test"]):
        p = generator.gpd(n, rng, xi=xi, clip=True)
        g = (seg.random(n) < cell["seg_share"]).astype(float)
        mult = np.where(g > 0, M, 1.0)
        out.append((np.column_stack([p.X, g]), p.y * mult, g, p.s * mult))
    return out


def measure(cell):
    (X, y, g, _), (Xc, yc, _, _), (Xt, yt, gt, st) = design(cell)
    if cell["sd_shift_target"] > 1.0:
        y0 = metrics.y0_for_sd_shift(y, cell["sd_shift_target"])
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {cell['sd_shift_target']} unreachable on this seed")
        X, y = leverage.add_row(X, y, y0)
        g = np.append(g, 0.0)
    y_fit, back, n = transforms.transform(cell["variant"], cell["clip_c"], y)
    q = back(models.quantiles(cell["model"], X, y_fit, np.vstack([Xc, Xt]),
                              seed=cell["seed"], levels=LEVELS, n_est=cell["n_est"]))
    qc, qt = q[:len(yc)], q[len(yc):]
    xi = cell["xi"]
    parts = {"all": np.ones(len(yt), bool), "seg": gt > 0, "out": gt == 0}
    r = dict(sd_shift=float(metrics.sd_shift(y)), n_clipped=n,
             n_clipped_seg=int(((y_fit < y) & (g > 0)).sum()))
    for i, tau, t in CAL_LEVELS:
        c_mul = metrics.factor_for_pinball(yc, qc[:, i], tau)
        c_add = metrics.shift_for_pinball(yc, qc[:, i], tau)
        r[f"c_mul{t}"], r[f"c_add{t}"] = c_mul, c_add
        variants = {"none": qt[:, i], "mul": c_mul * qt[:, i], "add": qt[:, i] + c_add}
        oracle = metrics.gpd_expected_pinball(st * metrics.gpd_quantile(tau, xi), st, xi, tau)
        for p, rows in parts.items():
            r[f"epb{t}_{p}_oracle"] = float(oracle[rows].mean())
            for c, qq in variants.items():
                e = metrics.gpd_expected_pinball(qq, st, xi, tau)
                r[f"epb{t}_{p}_{c}"] = float(e[rows].mean())
                r[f"exc{t}_{p}_{c}"] = float((yt[rows] > qq[rows]).mean())
    return r


def main():
    cells = [dict(model=m, xi=xi, seg_scale=M, seg_share=SHARE, sd_shift_target=s,
                  variant=v, clip_c=c, seed=seed, n_est=N_EST, n_train=N_TRAIN,
                  n_calib=N_CALIB, n_test=N_TEST)
             for m in MODELS for xi in XI for M in SCALES for s in SD_SHIFTS
             for seed in SEEDS for v, c in ARMS]
    seconds = runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
                         show=lambda r: f"epb99 none {r['epb99_all_none']:.4f}  mul "
                                        f"{r['epb99_all_mul']:.4f}  add {r['epb99_all_add']:.4f}"
                                        f"  oracle {r['epb99_all_oracle']:.4f}  c_mul "
                                        f"{r['c_mul99']:.3f}")
    report()
    print(f"\ntotal {seconds:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    by = ["model", "seg_scale", "sd_shift_target"]
    key = by + ["seed"]
    raw = d[d.variant == "raw"].set_index(key)
    cap = d[d.variant == "clip"].set_index(key)

    def excess(f, p, c):
        return f[f"epb99_{p}_{c}"] / f[f"epb99_{p}_oracle"] - 1

    print("\n=== expected pinball at 0.99 over the oracle, median over seeds (%) ===")
    rows = {}
    for name, f in (("raw", raw), ("cap", cap)):
        for c in CALS:
            for p in PARTS:
                rows[f"{name}+{c} {p}"] = excess(f, p, c)
    t = pd.DataFrame(rows).groupby(level=by).median() * 100
    print(t.round(2).to_string())

    def lower(a, b):
        return (a < b).groupby(level=by).sum()

    k1 = lower(raw["epb99_all_mul"], raw["epb99_all_none"])
    k2 = lower(cap["epb99_all_mul"], raw["epb99_all_mul"])
    gap_cal = (cap["epb99_all_mul"] / raw["epb99_all_mul"] - 1).groupby(level=by).median()
    gap_raw = (cap["epb99_all_none"] / raw["epb99_all_none"] - 1).groupby(level=by).median()
    k3 = (excess(raw, "seg", "mul") > excess(raw, "out", "mul")).groupby(level=by).sum()
    k4 = lower(raw["epb99_seg_mul"], cap["epb99_seg_mul"])
    k5 = (excess(raw, "seg", "add") > excess(raw, "seg", "mul")).groupby(level=by).sum()
    print("\nK1 raw+mul < raw, seeds (>= 15 in every cell, FMs):")
    print(k1.to_string())
    print("\nK2 cap+mul < raw+mul, seeds, and median gaps with and without calibration:")
    print(pd.DataFrame({"seeds": k2, "gap_mul_%": gap_cal * 100,
                        "gap_none_%": gap_raw * 100}).round(2).to_string())
    print("\nK3 segment excess > outside excess after raw+mul, seeds:")
    print(k3.to_string())
    print("\nK4 cap+mul segment > raw+mul segment, seeds:")
    print(k4.to_string())
    print("\nK5 segment excess raw+add > raw+mul, seeds:")
    print(k5.to_string())
    print("\n=== exceedance rate at 0.99 on the drawn test rows, median ===")
    exc = d.groupby(by + ["variant"])[[f"exc99_{p}_{c}" for p in ("all", "seg")
                                        for c in CALS]].median()
    print(exc.round(4).to_string())


if __name__ == "__main__":
    main()
