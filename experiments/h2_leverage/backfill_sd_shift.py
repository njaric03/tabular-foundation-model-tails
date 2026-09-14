# -*- coding: utf-8 -*-
"""Write the measured size of the treatment into rows already measured, without a model.

A dose was `y0 = dose * max(y_train)`, and max(y_train) is itself heavy-tailed: at
xi = 0.7 one nominal dose of 100 shifted the sd by 50.3 to 97.1 across five seeds. Until
the shift is on the row, the scatter of the model cannot be told from the scatter of the
treatment, and the generator cannot be drawn on the axis of the prevalence survey.

Every row is reconstructed from (xi, seed, n_train), since the generator is
deterministic, and gets:

    sd_shift      metrics.sd_shift(y_train + [y0])
    y0_over_max   y0 / max(y_train), so the old dose still reads
    dose_mode     "max", the rule every existing row was measured with

No measured value changes, and a row that already has `sd_shift` is skipped.

    python -u experiments/h2_leverage/backfill_sd_shift.py --check   # print only
"""
import sys

import numpy as np
import pandas as pd

from common import append, generator, metrics, paths, quiet

quiet.silence()

# (file, clip as in the script that wrote it, whether it has dose_mode, missing columns).
# influence.csv predates `position`, `n_train` and `n_est` as columns. Every default of
# influence.py matches its content (models, xi 0.3 and 0.7, doses 1/3/10/100, five seeds
# from 7000), so it was run with no environment, and the defaults are provable.
FILES = [
    ("h2_leverage/influence.csv", True, True, dict(position="centre", n_train=2000, n_est=1)),
    ("h2_leverage/influence_exaone.csv", True, True, {}),
    ("h2_leverage/mean_exaone.csv", False, False, {}),
]


def sd_shift_for(xi, seed, n_train, dose, clip):
    y = generator.gpd(int(n_train), np.random.default_rng(int(seed)), xi=float(xi), clip=clip).y
    return metrics.sd_shift(np.append(y, float(dose) * float(y.max())))


def main(check_only=False):
    for name, clip, with_mode, defaults in FILES:
        path = paths.result(name)
        if not path.exists():
            print(f"{name}: missing, skipped")
            continue
        d = pd.read_csv(path)
        for col, value in defaults.items():
            if col not in d.columns:
                d[col] = value
                print(f"{name}: column {col} filled with the script default {value!r}")
        for col in ["sd_shift", "y0_over_max"] + (["dose_mode"] if with_mode else []):
            if col not in d.columns:
                d[col] = pd.NA

        todo = d.sd_shift.isna() & d[["xi", "seed", "n_train", "dose"]].notna().all(axis=1)
        for i in d.index[todo]:
            r = d.loc[i]
            d.at[i, "sd_shift"] = sd_shift_for(r.xi, r.seed, r.n_train, r.dose, clip)
            d.at[i, "y0_over_max"] = float(r.dose)
        if with_mode:
            d.loc[d.dose_mode.isna(), "dose_mode"] = "max"

        print(f"{name}: {int(todo.sum())} rows got sd_shift")
        print(d.dropna(subset=["sd_shift"]).groupby(["xi", "dose"]).sd_shift
              .agg(["min", "median", "max"]).round(2).to_string())
        if not check_only:
            append.replace(name, d)
    if check_only:
        print("\n--check: nothing written")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
