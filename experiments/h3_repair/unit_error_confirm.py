# -*- coding: utf-8 -*-
"""The frozen cap on new subsamples, with the corrupted row drawn at random as well.

`unit_error_real.py` corrupts a row chosen among those whose error reaches an sd shift of
2: a conditional stress test. How often an error lands on such a row is a separate
question, and a cap that helps under the stress test can still cost more on clean
contexts than it saves under an error that rarely matters. This run measures both regimes
on subsamples that no earlier result has seen, with the cap fixed in advance.

Per table, REPEATS subsamples of N_FIT + N_TEST positive rows from DRAW_SEED, and from a
second stream two context rows: one among the visible rows, as before, and one uniform
over the whole context. Three contexts per subsample:

    clean     as drawn
    visible   the visible row times 100
    random    the uniform row times 100; a row whose error stays small is kept, because
              that is what a random error mostly does

Two arms, raw and clip_200 (median + 200 robust sd of the context), the cap chosen after
the runs of 12.9.2026 and frozen here. The test rows are the clean held-out rows of the
same permutation. When the cap touches no row its fit would repeat the raw one, so the raw
prediction is reused and `reused` says so.

Datasets are the five insurance tables. MEPS and BlogFeedback stay out: they are transfer
checks, and MEPS carried 6933 zero expenditures as positive values until 1.10.2026.

PREDICTIONS WRITTEN BEFORE THE RUN, 1.10.2026
---------------------------------------------
A unit is a (dataset, model) pair, 15 in all; a unit's value is its median over repeats.

C1, the stress test replicates. Under the visible error clip_200's pinball at 0.99 is
below raw's in at least 10 of 15 units (16 of 21 on the earlier draws).

C2, the clean cost stays small. On the clean context clip_200's pinball against raw is
within 2% at 0.99 in every unit, and at most 10% above it at 0.999 in every unit.

C3, a random error matters less. For each of the ten foundation-model units the rise of
raw's pinball at 0.99 over the clean context is smaller under the random row than under
the visible row.

Reported without a prediction: the two regimes per unit from mean losses within the unit,
D0 = mean pb99(clip) - mean pb99(raw) on clean contexts and D1 the same under the random
error, with the break-even error rate p* = D0 / (D0 - D1) where D0 > 0 > D1.

OUTCOME, 1.10.2026
------------------
C1 fails, 7 of 15: the cap lowers TabPFN-V3's pinball at 0.99 under the visible error in 4
of 5 units and TabICLv2's in 1 of 5 (+1.0% to +5.5% elsewhere). The earlier 16 of 21 owed
its margin to BlogFeedback and MEPS. C2 holds: the cap fires on no clean context except on
freMTPL2sev, where it lowers the 0.99 loss slightly. C3 fails, 6 of 10: a random x100 row
hurts TabICLv2 about as much as a visible one. D0 is never positive, so p* is undefined.
Full outcome in PREREGISTRATION.md.

    DATASETS=AutoClaims MODELS=GBM REPEATS=2 python -u experiments/h3_repair/unit_error_confirm.py
"""
import time

import numpy as np
import pandas as pd

from common import (
    append,
    datasets,
    env,
    leverage,
    metrics,
    models,
    paths,
    quiet,
    tables,
    transforms,
)

quiet.silence()

DATASETS = env.names("DATASETS", "freMTPL2sev,beMTPL97,ausprivauto0405,norauto,AutoClaims")
MODELS = env.models("TabPFN-V3,TabICLv2,GBM")
REPEATS = env.integer("REPEATS", 20)
N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
N_EST = env.integer("N_EST", 1)
CLIP_C = env.floats("CLIP_C", [200])
OUTPUT = env.text("OUTPUT", "unit_error_confirm.csv")
# Not a knob: a draw seed no earlier run used. The earlier runs drew from 31337.
DRAW_SEED = 20261001
FACTOR = 100.0
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ("50", "90", "99", "999")
ARMS = [("raw", 0.0)] + [("clip", c) for c in CLIP_C]
CONTEXTS = ("clean", "visible", "random")

COLUMNS = (["dataset", "model", "repeat", "context", "arm", "clip_c", "n_est", "n_fit",
            "n_test", "draw_seed", "row", "sd_shift", "natural_shift", "share_visible",
            "corrupt_rank", "n_touched", "reused"]
           + [f"d_q{t}" for t in TAGS] + [f"pb{t}" for t in TAGS]
           + ["n_above99", "seconds", "reason"])
KEY = ["dataset", "model", "repeat", "context", "arm", "clip_c", "n_est", "n_fit", "n_test",
       "draw_seed"]


def draws(y):
    """(indices, visible row or -1, random row, share visible) for every repeat."""
    rng = np.random.default_rng(DRAW_SEED)
    pick = np.random.default_rng(DRAW_SEED + 1)
    out = []
    for _ in range(REPEATS):
        idx = rng.permutation(len(y))[:N_FIT + N_TEST]
        shift = leverage.shift_if_corrupted(y[idx[:N_FIT]], FACTOR)
        visible = np.flatnonzero(shift >= leverage.VISIBLE_SHIFT)
        v = int(pick.choice(visible)) if len(visible) else -1
        out.append((idx, v, int(pick.integers(N_FIT)), float(len(visible) / N_FIT)))
    return out


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    for name in DATASETS:
        X, y = datasets.load(name, positive_only=True)
        for r, (idx, vis, rnd, share) in enumerate(draws(y)):
            Xf, y_clean, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
            natural = float(metrics.sd_shift(y_clean))
            contexts = {"clean": (y_clean, -1)}
            for label, row in (("visible", vis), ("random", rnd)):
                if row >= 0:
                    yc = y_clean.copy()
                    yc[row] *= FACTOR
                    contexts[label] = (yc, row)
            for model in MODELS:
                keys = [dict(dataset=name, model=model, repeat=r, context=cx, arm=a, clip_c=c,
                             n_est=N_EST, n_fit=N_FIT, n_test=N_TEST, draw_seed=DRAW_SEED)
                        for cx in CONTEXTS for a, c in ARMS]
                todo = [k for k in keys if append.key(k, KEY) not in done]
                if not todo:
                    continue
                seed = 7000 + r

                def predict(yc):
                    return models.quantiles(model, Xf, yc, Xte, seed=seed, levels=LEVELS,
                                            n_est=N_EST)

                t1, raw = time.time(), {}
                try:
                    q_ref = predict(y_clean)
                    err = None
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:110]
                # Raw before the caps, so a cap that touches nothing can reuse it.
                for k in sorted(todo, key=lambda k: k["arm"] != "raw"):
                    t2 = time.time()
                    row = dict(k, natural_shift=natural, share_visible=share)
                    if k["context"] not in contexts:
                        row["reason"] = "no visible row"
                        append.write(OUTPUT, row, COLUMNS)
                        continue
                    yc, bad = contexts[k["context"]]
                    row.update(row=bad, sd_shift=float(metrics.sd_shift(yc)),
                               corrupt_rank=int((yc > yc[bad]).sum()) + 1 if bad >= 0 else "")
                    if err:
                        row["reason"] = err
                    else:
                        try:
                            y_fit, back, n = transforms.transform(k["arm"], k["clip_c"], yc)
                            reused = False
                            if k["arm"] == "raw" and k["context"] == "clean":
                                qa = q_ref
                            elif k["arm"] != "raw" and n == 0 and k["context"] in raw:
                                qa, reused = raw[k["context"]], True
                            else:
                                qa = back(predict(y_fit))
                            if k["arm"] == "raw":
                                raw[k["context"]] = qa
                            row.update(n_touched=n, reused=reused, reason="")
                            row.update(zip([f"d_q{t}" for t in TAGS],
                                           metrics.rel_change_by_level(q_ref, qa)))
                            for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
                                row[f"pb{t}"] = metrics.pinball(yte, qa[:, i], a)
                            row["n_above99"] = int((yte > qa[:, 2]).sum())
                        except Exception as e:
                            row["reason"] = f"{type(e).__name__}: {e}"[:110]
                    row["seconds"] = round(time.time() - t2, 1)
                    append.write(OUTPUT, row, COLUMNS)
                print(f"  {name[:16]:16s} {model:10s} #{r} natural {natural:5.2f} "
                      f"visible {share:.1%}  [{time.time() - t1:.0f}s, {len(todo)} cells]",
                      flush=True)
    report()
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    d["arm_c"] = np.where(d.arm == "clip", "clip_" + d.clip_c.map("{:g}".format), d.arm)
    unit = ["dataset", "model"]
    w = d.pivot_table(index=unit + ["repeat"], columns=["context", "arm_c"],
                      values=["pb99", "pb999"])
    cap = [c for c in d.arm_c.unique() if c != "raw"][0]

    def rel(level, cx, a, b, cx_b=None):
        return w[(level, cx, a)] / w[(level, cx_b or cx, b)] - 1

    med = pd.DataFrame({
        "C1 vis cap/raw 99": rel("pb99", "visible", cap, "raw"),
        "C2 clean cap/raw 99": rel("pb99", "clean", cap, "raw"),
        "C2 clean cap/raw 999": rel("pb999", "clean", cap, "raw"),
        "C3 raw vis/clean 99": rel("pb99", "visible", "raw", "raw", "clean"),
        "C3 raw rnd/clean 99": rel("pb99", "random", "raw", "raw", "clean"),
        "rnd cap/raw 99": rel("pb99", "random", cap, "raw"),
    }).groupby(level=unit).median()
    print("\n=== medians over repeats, per unit ===")
    print(med.round(4).to_string())
    m = w.groupby(level=unit).mean()
    two = pd.DataFrame({"D0": m[("pb99", "clean", cap)] - m[("pb99", "clean", "raw")],
                        "D1": m[("pb99", "random", cap)] - m[("pb99", "random", "raw")]})
    two["p_star"] = np.where((two.D0 > 0) & (two.D1 < 0), two.D0 / (two.D0 - two.D1), np.nan)
    print("\n=== two regimes from mean pinball at 0.99 within each unit ===")
    print(two.to_string())
    fm = med.index.get_level_values("model") != "GBM"
    print(f"\nC1 {int((med['C1 vis cap/raw 99'] < 0).sum())} of {len(med)} units (>= 10)")
    print(f"C2 0.99 {int((med['C2 clean cap/raw 99'].abs() <= 0.02).sum())} of {len(med)}, "
          f"0.999 {int((med['C2 clean cap/raw 999'] <= 0.10).sum())} of {len(med)} (all)")
    c3 = med[fm]
    print(f"C3 {int((c3['C3 raw rnd/clean 99'] < c3['C3 raw vis/clean 99']).sum())} of "
          f"{len(c3)} foundation-model units (all)")


if __name__ == "__main__":
    main()
