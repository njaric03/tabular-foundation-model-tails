# -*- coding: utf-8 -*-
"""What tail the output head defines, against what it reports. Source, then numbers.

WHY THIS EXISTS

`experiments/h1_shape_vs_scale/` measures how much of a true tail gradient a
trained model reproduces, and the answer is 40 to 50 percent. That number invites
one explanation -- the synthetic prior is too light -- and the third direction in
`thesis/notes/model_survey.md` section 7 was built on it. But a prior can only be
blamed for what the output layer is able to say. If the head cannot express a
heavy tail, no prior can teach it one, and the ceiling is architectural rather
than a matter of pre-training data. This script settles that without a GPU and
without pre-training anything.

WHAT THE SOURCE SAYS, `tabpfn` 8.4.0

`tabpfn/architectures/shared/bar_distribution.py`:

    class FullSupportBarDistribution(BarDistribution):
        \"\"\"Bar distribution with half-normal tails, giving support over all of R.\"\"\"

    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=0.5):
        s = range_max / HalfNormal(1.0).icdf(p)
        return HalfNormal(s)

Beyond the outermost border the density is half-normal, so the tail is Gaussian,
xi = 0, lighter than the exponential that `thesis/literature/bibliography.md`
section 8.1 established for TabICL. Its only parameter is `range_max`, the width
of the outermost bucket, fixed when the bucket limits are chosen. The logits move
mass between buckets; nothing the network emits reaches the shape of the tail.

Then the part that matters more. `FullSupportBarDistribution` overrides
`forward`, `mean`, `mean_of_square`, `pi` and `ei`, and each of them builds the
two `side_normals`. It does **not** override `icdf`, and `icdf` is what the
public quantile path calls, `tabpfn/regressor.py` line 2151:

    return [criterion.icdf(logits, q).cpu().detach().numpy() for q in quantiles]

The inherited `icdf` interpolates linearly inside whichever bucket the level
falls in and stops at `borders[-1]`; its own comment reads
`# this might not do the right for outliers`. The tail the loss trained on is
half-normal over all of R. The tail the model reports is a bounded bar. They are
not the same object, and the difference is measured below.

WHERE THE BORDERS COME FROM, and why this is also the second part of the thesis

`regressor.py` `_rebuild_raw_space_bardist`:

    borders = self.znorm_space_bardist_.borders.detach()
    self.raw_space_bardist_ = FullSupportBarDistribution(
        borders * self.y_train_std_ + self.y_train_mean_)

The grid is a pre-training constant in z-space; the data supplies only the affine
map, and it supplies it through the plain mean and standard deviation. So the
largest value the model can report is `borders[-1] * y_train_std_ + y_train_mean_`.
That single expression carries both weaknesses this thesis measures: the ceiling
is fixed relative to the scale (part one), and the scale is not robust (part two).

WHAT IS MEASURED

Both quantile functions on the same logits, in the unit used everywhere else:
the location-invariant ratio (Q99 - Q50) / (Q90 - Q50) and the xi it implies.
`rep` is `icdf`, what a caller of `predict(output_type="quantiles")` receives.
`den` inverts the density the loss defines, bars plus the half-normal past the
last border. `q999_gap` is the piece of the predictive tail that exists in
training and is dropped at prediction time.

The border grid is read from a fitted regressor, so the numbers are the real
checkpoint's and not an assumption. That needs the Prior Labs weights like every
other TabPFN measurement here; without them the run records `reason` and stops,
rather than substituting a grid of its own and looking successful.

    python -u experiments/side/head_tail_family.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from common import append, metrics, paths, quiet

quiet.silence()

OUTPUT = os.environ.get("OUTPUT", "head_tail_family.csv")
# Mass in the outermost bar. The small values are the realistic regime; the
# large ones are the most tail-heavy statement the head can make at all, so the
# ceiling they produce is a ceiling for any prior whatsoever.
MASS = [float(x) for x in os.environ.get(
    "MASS", "0.0,0.0001,0.0005,0.001,0.002,0.005,0.008,0.01,0.05,0.2").split(",")]

KOLONE = ["source", "n_buckets", "tail_mass", "border_last", "width_last",
          "sigma_tail",
          "q50_rep", "q90_rep", "q99_rep", "q999_rep", "ratio_rep", "xi_rep",
          "q50_den", "q90_den", "q99_den", "q999_den", "ratio_den", "xi_den",
          "q999_gap", "q99_in_last_bar", "seconds", "reason"]
KEY = ["source", "n_buckets", "tail_mass"]

LEVELS = [0.5, 0.9, 0.99, 0.999]


def checkpoint_borders() -> torch.Tensor:
    """The z-space border grid of the shipped checkpoint.

    Fitting on four columns of noise is enough: the grid is a constant of the
    pre-trained model and does not depend on the table, which is the whole point
    of the finding this script records.
    """
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(0)
    m = TabPFNRegressor(n_estimators=1, device="cpu")
    m.fit(rng.normal(size=(200, 4)), rng.normal(size=200))
    return m.znorm_space_bardist_.borders.detach().double()


def _logits(borders: torch.Tensor, tail_mass: float) -> torch.Tensor:
    """A standard normal body on the grid, with `tail_mass` in the last bar.

    The body has to be something a trained model could plausibly emit, because
    the implied xi is read off Q50, Q90 and Q99 and a uniform body puts all three
    in the wrong place. A discretised N(0, 1) in the z-space the grid lives in is
    the honest default: it is what the head outputs for a light-tailed target,
    and `tail_mass` is then exactly the amount of tail belief added on top.
    """
    from scipy.stats import norm

    b = borders.numpy()
    body = np.diff(norm.cdf(b))
    body = body / body.sum() * (1.0 - tail_mass)
    body[-1] += tail_mass
    probs = torch.tensor(np.clip(body, 1e-300, None), dtype=torch.float64)
    return torch.log(probs).unsqueeze(0)


def reported_quantiles(dist, logits) -> np.ndarray:
    """What `regressor.py` returns: the inherited `icdf`."""
    return np.array([float(dist.icdf(logits, a).squeeze()) for a in LEVELS])


def density_quantiles(dist, logits) -> np.ndarray:
    """Quantiles of the density the loss defines, half-normal tail included.

    Inside the grid the two agree by construction, so the only branch that
    matters is past `borders[-2]`, where the mass of the last bar is spread over
    a half-normal instead of over a bar of finite width.
    """
    from scipy.stats import halfnorm

    probs = torch.softmax(logits, -1).squeeze(0).numpy()
    borders = dist.borders.numpy()
    cum_before_last = float(probs[:-1].sum())
    p_last = float(probs[-1])
    sigma = float(dist.halfnormal_with_p_weight_before(dist.bucket_widths[-1]).scale)

    out = []
    for a in LEVELS:
        if a <= cum_before_last:
            out.append(float(dist.icdf(logits, a).squeeze()))
        else:
            inner = (a - cum_before_last) / p_last
            out.append(float(borders[-2] + halfnorm.ppf(inner, scale=sigma)))
    return np.array(out)


def main() -> None:
    from tabpfn.architectures.shared.bar_distribution import (
        FullSupportBarDistribution,
    )

    gotovi = append.done(OUTPUT, KEY)
    try:
        borders = checkpoint_borders()
        source, err = "TabPFN-V3", None
    except Exception as e:                           # noqa: BLE001
        borders, source = None, "TabPFN-V3"
        err = f"{type(e).__name__}: {e}"

    n_buckets = int(len(borders) - 1) if borders is not None else -1

    for mass in MASS:
        if append.key(dict(source=source, n_buckets=n_buckets,
                           tail_mass=mass), KEY) in gotovi:
            continue
        t0 = time.time()
        r = dict(source=source, n_buckets=n_buckets, tail_mass=mass, reason=err)
        if borders is not None:
            try:
                dist = FullSupportBarDistribution(borders.clone())
                logits = _logits(borders, mass)
                sigma = float(dist.halfnormal_with_p_weight_before(
                    dist.bucket_widths[-1]).scale)
                rep = reported_quantiles(dist, logits)
                den = density_quantiles(dist, logits)
                ratio_rep = (rep[2] - rep[0]) / (rep[1] - rep[0])
                ratio_den = (den[2] - den[0]) / (den[1] - den[0])
                r.update(border_last=float(borders[-1]),
                         width_last=float(borders[-1] - borders[-2]),
                         sigma_tail=sigma,
                         q50_rep=rep[0], q90_rep=rep[1], q99_rep=rep[2],
                         q999_rep=rep[3], ratio_rep=ratio_rep,
                         xi_rep=metrics.xi_from_ratio(ratio_rep),
                         q50_den=den[0], q90_den=den[1], q99_den=den[2],
                         q999_den=den[3], ratio_den=ratio_den,
                         xi_den=metrics.xi_from_ratio(ratio_den),
                         q999_gap=float(den[3] - rep[3]),
                         q99_in_last_bar=bool(rep[2] > float(borders[-2])))
            except Exception as e:                   # noqa: BLE001
                r["reason"] = f"{type(e).__name__}: {e}"
        r["seconds"] = round(time.time() - t0, 1)
        append.write(OUTPUT, r, KOLONE)
        print(f"  mass={mass:<6} q999 rep={r.get('q999_rep')} "
              f"den={r.get('q999_den')}", flush=True)

    d = pd.read_csv(paths.result(OUTPUT)).sort_values("tail_mass")
    ok = d[d.reason.isna()]
    print()
    print(ok[["tail_mass", "q99_rep", "xi_rep", "q999_rep", "q999_den",
              "q999_gap", "q99_in_last_bar"]].round(4).to_string(index=False))
    if len(ok):
        b = float(ok.border_last.iloc[0])
        print()
        print(f"The reported quantile cannot leave [-{b:g}, {b:g}] standard "
              f"deviations of the training target:\n`icdf` is not overridden, so "
              f"it stops at the last border. Above it the loss still defines a\n"
              f"half-normal of scale {float(ok.sigma_tail.iloc[0]):.4g}, and "
              f"nothing in the prediction path reads it.")


if __name__ == "__main__":
    main()
