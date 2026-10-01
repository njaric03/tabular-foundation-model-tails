# -*- coding: utf-8 -*-
"""The border grid of LimiX-2's regression head, read from the checkpoint. No GPU.

LimiX-2 decodes its bar head with half-normal outer buckets, the design of TabPFN's
`FullSupportBarDistribution`, and maps the borders to target units through the plain mean
and sd of the context (`common/adapters/limix.py`). How finely the grid resolves a value
therefore depends on how many sds from the mean it lies, and one leverage row moves every
value towards the centre in sd units. This counts the borders per band, the counterpart of
the TabPFN grid in `findings/h1/head_tail_family.md`.

    venv-limix/Scripts/python.exe -u experiments/side/head_grid_limix.py
"""
from __future__ import annotations

import numpy as np

from common import append, env, quiet

quiet.silence()

OUTPUT = env.text("OUTPUT", "head_grid_limix2.csv")
BANDS = [(-1, 1), (1, 3), (3, 10), (10, 20), (20, 30), (30, 100)]
COLUMNS = ["model", "revision", "n_buckets", "lo", "hi", "borders", "per_sd",
           "last_inner", "outer_width", "reason"]
KEY = ["model", "revision", "lo", "hi"]


def main() -> None:
    from common.adapters import limix
    _, ckpt = limix._checkpoint()
    b = ckpt["state_dict"]["_reg_borders"].double().numpy()
    done = append.done(OUTPUT, KEY)
    for lo, hi in BANDS:
        row = dict(model="LimiX-2", revision=limix.REVISION[:7], lo=lo, hi=hi)
        if append.key(row, KEY) in done:
            continue
        n = int(((b >= lo) & (b < hi)).sum())
        row.update(n_buckets=len(b) - 1, borders=n, per_sd=n / (hi - lo),
                   last_inner=float(b[-2]), outer_width=float(b[-1] - b[-2]), reason="")
        append.write(OUTPUT, row, COLUMNS)
        print(f"  [{lo:>3}, {hi:>3})  {n:5d} borders, {n / (hi - lo):6.1f} per sd", flush=True)
    print(f"  {len(b) - 1} buckets; last inner border {b[-2]:.2f}, outer bucket {b[-1] - b[-2]:.2f}"
          f" wide; first borders {np.round(b[:2], 2)}")


if __name__ == "__main__":
    main()
