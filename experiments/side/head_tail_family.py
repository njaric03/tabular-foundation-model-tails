# -*- coding: utf-8 -*-
"""The tail TabPFN's output head defines, against the tail it reports. No GPU.

If the head cannot express a heavy tail, no prior can teach it one. From the source of
`tabpfn` 8.4.0:

* `FullSupportBarDistribution` puts a half-normal beyond the outermost border, whose only
  parameter is that bucket's width: the tail is Gaussian, xi = 0, and nothing the network
  emits reaches its shape.
* It overrides `forward`, `mean`, `pi` and `ei`, but not `icdf`, which is what
  `regressor.py` calls for quantiles. The inherited `icdf` interpolates linearly inside a
  bar and stops at `borders[-1]`: the loss trains on a half-normal tail, the predictions
  report a bounded bar.
* `_rebuild_raw_space_bardist` builds `borders * y_train_std_ + y_train_mean_`, so the
  largest value the model can report is a pre-training constant times a non-robust sd.

Measured on the border grid of the shipped checkpoint, with a discretised N(0, 1) body
and `tail_mass` placed in the last bar: both quantile functions on the same logits.
`rep` is `icdf`, what a caller gets; `den` inverts the density the loss defines.

    python -u experiments/side/head_tail_family.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import env, metrics, models, paths, quiet, runner, tables

quiet.silence()

OUTPUT = env.text("OUTPUT", "head_tail_family.csv")
# Mass in the outermost bar: small values are the realistic regime, large ones the
# heaviest statement the head can make at all.
MASS = env.floats("MASS", [0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.2])
LEVELS = [0.5, 0.9, 0.99, 0.999]

COLUMNS = ["source", "n_buckets", "tail_mass", "border_last", "width_last",
           "sigma_tail",
           "q50_rep", "q90_rep", "q99_rep", "q999_rep", "ratio_rep", "xi_rep",
           "q50_den", "q90_den", "q99_den", "q999_den", "ratio_den", "xi_den",
           "q999_gap", "q99_in_last_bar", "seconds", "reason"]
KEY = ["source", "n_buckets", "tail_mass"]


def checkpoint_borders() -> torch.Tensor:
    """The z-space border grid of the shipped checkpoint, a constant of the pre-trained
    model; fitting on noise is enough to read it."""
    rng = np.random.default_rng(0)
    m = models.tabpfn_regressor("TabPFN-V3", seed=0, n_est=1)
    m.fit(rng.normal(size=(200, 4)), rng.normal(size=200))
    return m.znorm_space_bardist_.borders.detach().double()


def logits_with_tail(borders: torch.Tensor, tail_mass: float) -> torch.Tensor:
    """A discretised N(0, 1) body on the grid with `tail_mass` added to the last bar."""
    from scipy.stats import norm

    body = np.diff(norm.cdf(borders.numpy()))
    body = body / body.sum() * (1.0 - tail_mass)
    body[-1] += tail_mass
    return torch.log(torch.tensor(np.clip(body, 1e-300, None), dtype=torch.float64)).unsqueeze(0)


def density_quantiles(dist, logits) -> np.ndarray:
    """Quantiles of the density the loss defines, half-normal tail included. Inside the grid
    they equal `icdf`; past `borders[-2]` the last bar's mass follows the half-normal."""
    from scipy.stats import halfnorm

    probs = torch.softmax(logits, -1).squeeze(0).numpy()
    before_last, p_last = float(probs[:-1].sum()), float(probs[-1])
    sigma = float(dist.halfnormal_with_p_weight_before(dist.bucket_widths[-1]).scale)
    return np.array([float(dist.icdf(logits, a).squeeze()) if a <= before_last
                     else float(dist.borders[-2] + halfnorm.ppf((a - before_last) / p_last,
                                                                scale=sigma))
                     for a in LEVELS])


def main() -> None:
    from tabpfn.architectures.shared.bar_distribution import FullSupportBarDistribution

    # The grid needs the Prior Labs weights. Without them every row records why, instead
    # of a substitute grid that would look like a result.
    try:
        borders, missing = checkpoint_borders(), None
    except Exception as e:
        borders, missing = None, runner.reason(e)

    def measure(cell):
        if missing:
            raise RuntimeError(missing)
        dist = FullSupportBarDistribution(borders.clone())
        logits = logits_with_tail(borders, cell["tail_mass"])
        rep = np.array([float(dist.icdf(logits, a).squeeze()) for a in LEVELS])
        den = density_quantiles(dist, logits)
        ratio_rep = (rep[2] - rep[0]) / (rep[1] - rep[0])
        ratio_den = (den[2] - den[0]) / (den[1] - den[0])
        r = dict(border_last=float(borders[-1]), width_last=float(borders[-1] - borders[-2]),
                 sigma_tail=float(dist.halfnormal_with_p_weight_before(dist.bucket_widths[-1]).scale),
                 ratio_rep=ratio_rep, xi_rep=metrics.xi_from_ratio(ratio_rep),
                 ratio_den=ratio_den, xi_den=metrics.xi_from_ratio(ratio_den),
                 q999_gap=float(den[3] - rep[3]), q99_in_last_bar=bool(rep[2] > float(borders[-2])))
        for v, t in zip(rep, ("50", "90", "99", "999")):
            r[f"q{t}_rep"] = v
        for v, t in zip(den, ("50", "90", "99", "999")):
            r[f"q{t}_den"] = v
        return r

    n_buckets = len(borders) - 1 if borders is not None else -1
    cells = runner.grid(source=["TabPFN-V3"], n_buckets=[n_buckets], tail_mass=MASS)
    runner.run(OUTPUT, COLUMNS, KEY, cells, measure,
               show=lambda r: f"q999 reported {r['q999_rep']:.3f}  density {r['q999_den']:.3f}")
    d = tables.ok_rows(pd.read_csv(paths.result(OUTPUT))).sort_values("tail_mass")
    print(d[["tail_mass", "q99_rep", "xi_rep", "q999_rep", "q999_den", "q999_gap",
             "q99_in_last_bar"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
