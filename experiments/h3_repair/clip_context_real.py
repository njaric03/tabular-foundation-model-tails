# -*- coding: utf-8 -*-
"""The context repairs of `clip_context.py` on real subsamples that carry leverage.

Subsamples are drawn exactly as `prevalence_models.py` draws them (`leverage.subsamples`,
seed 31337), so the raw arm must reproduce `prevalence_models_wide.csv` on the same cells:
the control that the instrument is the same. For each subsample and model, one fit per
context:

    without_max   the largest row removed; the reference of the leverage sweep
    raw, clip_C, log, tail_log, evt_trim, sd_cap, rank_*   the arms of common/transforms.py

`d_q*` is the median relative change against `without_max`. The held-out rows are the
arbiter of cost: pinball at each level on N_TEST clean rows.

Outcomes, 12.9.2026, on freMTPL2sev:

  The raw arm reproduces the wide sweep for GBM exactly and for TabICLv2 to 1e-5; for
  TabPFN-V3 it differs by up to 0.03 per cell, from a rebuilt environment.

  Raw d_q99 at 2-4 sd / at 4 sd and above: TabPFN-v2.5 -29.7% / -51.6%, TabPFN-v2.6
  -54.6% / -48.5%, TabICLv2 -5.2% / -15.4%, TabPFN-V3 -6.8% / -7.8%, GBM +13.3% / -1.7%,
  EXAONE +2.8% / +0.9%. EXAONE standardises the target like the others and loses
  nothing; TabICLv2 has no grid and still loses 15%. So non-robust standardisation costs
  part of the tail, the fixed grid multiplies that loss, and a quantile head can avoid it.

  Pinball at 0.99 against raw, subsamples at 2 sd and above: clip_200 better in 9-10 of 10
  for every model (TabPFN-v2.6 -17.9%, EXAONE -1.8%); sd_cap close behind; rank_gpd 6-10 of
  10; evt_trim touches no context anywhere.

    DATASETS=OnlineNewsPopularity MODELS=TabICLv2 \
      python -u experiments/h3_repair/clip_context_real.py
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

DATASETS = env.names("DATASETS", "freMTPL2sev")
MODELS = env.models("TabPFN-V3,TabICLv2,GBM")
CLIP_C = env.floats("CLIP_C", [50, 200])
# The sizes, bins, quota and seed of the leverage sweep, so the subsamples are the same.
N_FIT = env.integer("N_FIT", 2000)
N_TEST = env.integer("N_TEST", 1000)
N_EST = env.integer("N_EST", 1)
N_PER_BIN = env.integer("N_PER_BIN", 5)
MAX_ATTEMPTS = env.integer("MAX_ATTEMPTS", 4000)
LEVELS = [0.5, 0.9, 0.99, 0.999]
TAGS = ["50", "90", "99", "999"]
OUTPUT = env.text("OUTPUT", "clip_context_real.csv")

COLUMNS = ["dataset", "model", "bin", "repeat", "arm", "clip_c", "n_est", "n_fit",
           "n_test", "sd_shift", "leverage", "n_clipped",
           "d_q50", "d_q90", "d_q99", "d_q999",
           "pb50", "pb90", "pb99", "pb999", "n_above99", "seconds", "reason"]
KEY = ["dataset", "model", "bin", "repeat", "arm", "clip_c", "n_est", "n_fit", "n_test"]

ARMS = ([("without_max", 0.0), ("raw", 0.0)] + [("clip", c) for c in CLIP_C]
        + [("log", 0.0), ("tail_log", transforms.TAIL_LOG_C), ("evt_trim", 0.0),
           ("sd_cap", transforms.TAU), ("rank_emp", 0.0), ("rank_gpd", 0.0),
           ("rank_gpd_trim", 0.0), ("rank_exp", 0.0)])
# VARIANTS limits the arms for a run that needs only some of them, as clip_200 is named;
# the rows of an arm do not depend on which other arms run, and the reference
# (without_max) is fitted either way.
_ARM_NAMES = [a if a != "clip" else f"clip_{c:g}" for a, c in ARMS]
VARIANTS = env.names("VARIANTS", _ARM_NAMES)
ARMS = [arm for arm, n in zip(ARMS, _ARM_NAMES) if n in VARIANTS]


def arm_row(q, q_ref, yte):
    r = dict(zip([f"d_q{t}" for t in TAGS], metrics.rel_change_by_level(q_ref, q)))
    for i, (a, t) in enumerate(zip(LEVELS, TAGS)):
        r[f"pb{t}"] = metrics.pinball(yte, q[:, i], a)
    r["n_above99"] = int((yte > q[:, 2]).sum())
    return r


def main():
    t0 = time.time()
    done = append.done(OUTPUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    for name in DATASETS:
        X, y = datasets.load(name)
        if models.CATEGORICAL:
            # The columns prepare() coded from categories, handed to the model as such.
            models.CATEGORICAL_INDICES = datasets.LAST.get("categorical", [])
            print(f"  [categorical] {len(models.CATEGORICAL_INDICES)} of {X.shape[1]} "
                  f"columns passed as categories", flush=True)
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        found = leverage.subsamples(y, np.random.default_rng(31337), N_FIT, N_TEST,
                                    N_PER_BIN, MAX_ATTEMPTS)
        for k, subsamples in found.items():
            for j, (idx, shift, lev) in enumerate(subsamples):
                Xf, yf, Xte, yte = X[idx[:N_FIT]], y[idx[:N_FIT]], X[idx[N_FIT:]], y[idx[N_FIT:]]
                keep = np.ones(len(yf), bool)
                keep[int(np.argmax(yf))] = False
                for model in MODELS:
                    keys = [dict(dataset=name, model=model, bin=leverage.bin_name(k), repeat=j,
                                 arm=a, clip_c=c, n_est=N_EST, n_fit=N_FIT, n_test=N_TEST)
                            for a, c in ARMS]
                    todo = [kk for kk in keys if append.key(kk, KEY) not in done]
                    if not todo:
                        continue
                    t1 = time.time()
                    seed = 7000 + j                    # as in prevalence_models.py

                    def predict(Xc, yc):
                        return models.quantiles(model, Xc, yc, Xte, seed=seed, levels=LEVELS,
                                                n_est=N_EST)

                    try:
                        # Refitted on resume too; every model here is deterministic within
                        # one environment, so the earlier rows share this reference.
                        q_ref = predict(Xf[keep], yf[keep])
                        fits = {("without_max", 0.0): (q_ref, 0)}
                        for kk in todo:
                            a, c = kk["arm"], kk["clip_c"]
                            if a != "without_max":
                                y_fit, back, n = transforms.transform(a, c, yf)
                                with np.errstate(over="ignore", invalid="ignore"):
                                    fits[(a, c)] = (back(predict(Xf, y_fit)), n)
                        err = None
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"[:110]
                    secs = round((time.time() - t1) / (len(todo) + 1), 1)
                    for kk in todo:
                        row = dict(kk, sd_shift=shift, leverage=lev, seconds=secs)
                        if err:
                            row["reason"] = err
                        else:
                            qa, n_clipped = fits[(kk["arm"], kk["clip_c"])]
                            row.update(arm_row(qa, q_ref, yte), n_clipped=n_clipped, reason="")
                        append.write(OUTPUT, row, COLUMNS)
                    msg = err or "  ".join(
                        f"{a if a != 'clip' else f'clip_{c:g}'} "
                        f"{arm_row(v[0], q_ref, yte)['d_q99']:+6.1%}"
                        for (a, c), v in fits.items() if a != "without_max")
                    print(f"  {name[:14]:14s} {model:10s} {leverage.bin_name(k):>8} #{j} "
                          f"sd={shift:5.2f}  d_q99: {msg}  [{secs}s/arm]", flush=True)
    report()
    print(f"\ntotal {time.time() - t0:.0f}s -> {OUTPUT}")


def report():
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT)))
    if d.empty:
        return
    d["arm_c"] = np.where(d.arm == "clip", "clip_" + d.clip_c.map("{:g}".format), d.arm)
    for col in ("d_q99", "pb99"):
        print(f"\n=== {col}, median by bin ===")
        print(d.pivot_table(index=["dataset", "model", "bin"], columns="arm_c", values=col,
                            aggfunc="median").round(4).to_string())
    wide = paths.result("prevalence_models_wide.csv")
    if wide.exists():
        key = ["dataset", "model", "bin", "repeat"]
        ref = pd.read_csv(wide).dropna(subset=["d_q99"])
        j = (d[d.arm == "raw"].set_index(key)[["d_q99"]]
             .join(ref.set_index(key)[["d_q99"]], rsuffix="_wide", how="inner"))
        if len(j):
            gap = float((j.d_q99 - j.d_q99_wide).abs().max())
            print(f"\ncontrol, raw against prevalence_models_wide on {len(j)} cells: largest "
                  f"|d_q99 gap| {gap:.2e} ({'holds' if gap < 1e-3 else 'FAILS'})")


if __name__ == "__main__":
    main()
