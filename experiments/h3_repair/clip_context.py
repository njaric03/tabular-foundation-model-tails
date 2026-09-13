# -*- coding: utf-8 -*-
"""Clip the context before TabPFN sees it, and test whether resolution comes back.

WHY
---
`grid_occupancy.py` named the mechanism behind the negative implied xi in part
two, and it is resolution, not the ceiling. The raw-space grid is
`borders * y_train_std_ + y_train_mean_`; one leverage row inflates the sd, the
grid stretches with it, and the number of borders over the actual data range
falls from 2560 at a clean context to 224 at a shift of 50. The inherited `icdf`
interpolates linearly inside a bar, linear is locally uniform, and a uniform
tail reads as negative xi. The model has not lost its belief; there are no bars
left between Q(0.9) and Q(0.99) to express it in.

`q999_in_outer_five` in that same file is 0.0000 at every shift above 4, so the
half-normal tail and the `icdf` mismatch are real but never reached in practice.
A patch to `icdf` was written and dropped for that reason: it moved nothing.

WHY `robust_scale.py` DID NOT TEST THIS
---------------------------------------
It replaces `y_train_std_` after the fit and rebuilds the grid. But the context
the network conditions on is normalised inside `fit`, at `regressor.py:1190`:

    mean, std = np.mean(y), np.std(y)
    self.y_train_std_ = std.item() + 1e-20
    y = (y - self.y_train_mean_) / self.y_train_std_

so after the swap the network still sees a context scaled by the contaminated
sd, while its output is decoded with a scale 0.053 times smaller. That is
decoding with a different scale than was used to encode, and it alone would
produce the +925% and the worse pinball at every level, median included. The
recorded conclusion, that the vulnerability and the safety mechanism are the
same thing, rests on it.

WHAT IS MEASURED
----------------
The fix has to act before `fit`, because `fit` re-standardises whatever it is
given by its plain mean and sd. That also means a robust scale on its own can
do nothing: `(y - median) / robust_sd` is affine, and the internal
standardisation undoes any affine map exactly. So it is included here as a
control that must come out identical to `raw`, down to float noise. If it does
not, the reading of `regressor.py` above is wrong.

The only operation that is not undone is a non-linear one. `clip` caps the
context at `median + C * robust_sd`, in the target's own units, before the fit.
The leverage row is pulled back to the edge of the body instead of stretching
the grid over it, and the grid stays calibrated on the data. Nothing else
changes: the network, the weights, the test inputs, the output head.

    raw          the shipped behaviour
    robust       affine robust standardisation; control, must equal raw
    clip_C       context capped at median + C * robust_sd, for each C

C is a knob and a column, and it is the price: on a heavy tail a small C also
cuts values that are genuine. So every cell is measured on a clean context too
(`sd_shift_target = 1`), where any damage is pure cost.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  P1. `robust` equals `raw` in every column. It is here to fail loudly if the
      reading of `fit` is wrong.
  P2. Under a shift of 20 or 50, `clip_C` keeps `borders_in_data` near its clean
      value, where `raw` falls towards 224.
  P3. Where P2 holds, `xi_implied` stays positive under `clip_C` instead of
      crossing zero. If `borders_in_data` recovers and xi still goes negative,
      resolution is not the mechanism, and that is worth as much to know.
  P4. On a clean context, `clip_C` costs pinball at 0.999, more for small C and
      heavy tails, and close to nothing at 0.5.

OUTCOME OF THE FIRST RUN
------------------------
TabPFN-V3, xi in {0.7, 0.9}, five seeds, 200 cells.

  P1 HELD. robust equals raw to 6.75e-08 in Q(0.99). The reading of `fit` is
     right, which also means `robust_scale.py` never tested a robust scale.
  P2 HELD. Under raw the borders over the data fall 2559 -> 543 -> 217 at shifts
     1 / 20 / 50. Every clip arm keeps them between 2573 and 2668 at every shift.
  P3 HELD. At a shift of 20 raw reads xi = -0.23 at a true 0.7 and -0.30 at 0.9.
     clip_50 reads 0.697 and 0.884, clip_200 0.674 and 0.841, and neither moves
     with the shift. The raw-ratio column agrees in sign (-0.16 and -0.15), so the
     negative xi is not an artefact of the measure.
  P4 HELD, and it picks C. On a clean context clip_20 costs pinball at 0.999
     heavily (0.247 -> 0.382 at xi 0.7, 0.781 -> 1.718 at 0.9) and clip_50 some;
     clip_200 is -4% at 0.7 and +9% at 0.9. Under a shift of 20, clip_200 cuts
     pinball at 0.999 by 59% at xi 0.7 and by 36% at 0.9.

C = 200 was picked from a grid of three after the run, not predicted, and that is
a selection to state. On a clean context at xi 0.9, clip_50 also reads the shape
closer to the truth than raw (0.876 against 0.801): heavy-tailed data carries its
own leverage, which touches part one and has not been followed up. The run is
synthetic, one model, one leverage row at the centre of x, and has not been tried
on real data.

SECOND RUN: A RULE INSTEAD OF A PICKED C
----------------------------------------
Clipping cannot tell a unit error from a genuine extreme, and C is the knob that
decides for it. Three arms replace the knob with something that has a reason,
and the seeds go from 5 to 20 so the first run's numbers settle as well.

    log        fit on log(y), exponentiate the quantiles. A monotone map, so
               quantiles come back exactly; but it reshapes the whole body.
    tail_log   identity up to median + C * robust_sd, logarithmic above it, and
               inverted on the way out. The body is untouched, the leverage row
               is compressed rather than cut, and a quantile above the cap is
               still reachable. C = 50, the value at which hard clipping costs.
    evt_trim   no C. The top five values are tested against the tail of the
               rest: under a Pareto tail the scaled log spacings
               i * (log X(i) - log X(i+1)) are independent exponentials with mean
               xi (Renyi), so a spacing far above the mean of the others says the
               value above it is not from that tail. xi is estimated from the
               spacings below the candidates, the threshold is Bonferroni at 0.01
               over five, and flagged values are pulled down to the largest one
               not flagged. A simplified form of the trimmed-Hill outlier test of
               Bhattacharya, Kallitsis and Stoev (2019). By construction it
               flags a clean heavy-tailed context about 1% of the time.

Predictions, written before the second run:
  P5. evt_trim flags the injected row at shifts 20 and 50 in most seeds, and
      touches a clean context in at most 5% of seeds.
  P6. Where evt_trim flags, its xi_implied and pb999 are within noise of
      clip_200; on the clean context its cost is close to zero because it
      touches nothing.
  P7. log keeps xi_implied positive under leverage, but on the clean context
      costs pinball at 0.999 more than clip_200, because it reshapes the body
      (`repair.py` saw +2 to +30% at 0.999 without any leverage).
  P8. tail_log at C = 50 costs less than clip_50 on the clean context at 0.999,
      because it is inverted on the way out.

OUTCOME OF THE SECOND RUN
-------------------------
TabPFN-V3, xi in {0.7, 0.9}, 20 seeds (the first five are the first run's),
1280 cells. Pinball at 0.999 is compared with raw seed by seed; "better" counts
seeds, p is a sign test over the 20.

  P1 HELD again: robust equals raw to 8.0e-08.
  P5 FAILED. evt_trim touches 1 of 20 seeds at every shift, clean or not (2 of
     20 at xi 0.7 and a shift of 50). That is its false-alarm rate, not power:
     a single value 30 times the second largest is within what a tail of 0.7
     produces often enough at n = 2000, so the test cannot call it an error. An
     honest form of the dilemma: the context's own order statistics cannot tell
     an error from an extreme at this size.
  P6 VACUOUS: where evt_trim does not fire it is raw.
  P7 FAILED. log does not cost on the clean context (-0.7% at 0.7, -11.8% at
     0.9, neither significant) and keeps xi_implied near 0.66 and 0.83 at every
     shift. Under leverage it repairs, less reliably than clip_200: at a shift of
     20, 13/20 (p = 0.26) at 0.7 and 15/20 at 0.9, against 16/20 and 18/20.
  P8 FAILED, badly. tail_log reads the shape right (0.64, 0.80) but its pinball
     at 0.999 is 1.5 to 1073 against raw's 0.3 to 2.0. Inverting the compression
     re-expands whatever mass the model puts near the compressed top, and
     exponentially. Compressing a tail and giving it back does not work.

And the first run's numbers settle. clip_200 on a clean context is +0.0% (7/20)
at 0.7 and -1.0% (11/20) at 0.9: no measurable cost. The "+9% at 0.9" of the
first run was five seeds. Under leverage it is better in 15/20 to 20/20 seeds at
every shift above 1, by 9% at a shift of 4 and 38 to 48% at 50. clip_50 does
cost: +14% at 0.9 on the clean context, better in only 2 of 20 seeds
(p = 0.0004). So the trade-off is real for a tight cap and not measurable at 200.

THIRD RUN: A CAP THAT IS DERIVED, NOT PICKED
-------------------------------------------
`sd_cap` drops C and caps at whatever brings the context's own sd shift to 1.2.
Predictions, written before that run:
  P9.  On a context already below 1.2 it clips nothing and costs exactly zero,
       where clip_200 still clips one to five values on some real tables.
  P10. Where the leverage is, it lands on a higher cap than C = 200 (284 robust
       sd against 200 on the worst freMTPL2sev subsample) and clips the same two
       values, so it should recover resolution and implied xi within noise of
       clip_200 while distorting less.
  P11. On the generator at shifts 20 and 50 its implied xi is positive and within
       noise of clip_200's 0.63 and 0.80.

FOURTH RUN: RANKS IN, EXTREME VALUE THEORY OUT
---------------------------------------------
Every arm so far either damages the context (clip, sd_cap, evt_trim) or reshapes
it and hands the damage to the output (log, tail_log). `rank_emp` and `rank_gpd`
do neither. The context target becomes its own normal scores, so the mechanism
is bypassed rather than repaired: a rank cannot inflate an sd. The cost is that
the inverse map saturates at the largest observed value, and `rank_gpd` pays it
with a generalised Pareto fitted to the context's own exceedances above its 90th
percentile, splicing the empirical inverse below and the Pareto above.

Predictions, written before the run:
  R1. rank_emp is worse than raw on pinball at 0.999 on a clean context, because
      every upper quantile saturates at the largest observed y.
  R2. rank_gpd removes that: its pinball at 0.999 on a clean context is within
      noise of clip_200's, or better.
  R3. Under shifts of 20 and 50 both rank arms keep borders_in_data near the
      clean value and the implied xi positive, and they do so by construction,
      not by tuning. If they do not, the reading of the mechanism is wrong.
  R4. rank_gpd imposes the marginal tail shape, so its implied xi is pulled
      towards the average of xi(x) rather than the local one. On the generator
      that is a bias to see; on real data it should lift Q(0.99) towards the
      reference.
  R5. Under a unit error its own GPD fit is contaminated, because the corrupted
      row is the top exceedance, so its gain at 0.99 is smaller than the cap's
      and may be negative at 0.999.

OUTCOME OF THE THIRD AND FOURTH RUNS, 12.9.2026
-----------------------------------------------
20 seeds each, paired seed by seed against raw unless stated otherwise.

  P9 HELD as a construction claim and needs one caveat as an empirical one. A
     context below a shift of 1.2 is returned untouched, but a clean heavy-tailed
     generator context is often not below 1.2: at xi 0.7 and 0.9 sd_cap still
     clips one or two values with no injected row, which is the same "heavy tails
     carry their own leverage" the first run saw from the other side. On
     freMTPL2sev's clean bin it clips 0 where clip_200 clips 3.
  P10 HELD. It lands on a higher cap (284 robust sd against 200 on the worst
     freMTPL2sev subsample) and clips 2 values where clip_200 clips 5 at xi 0.9,
     and it still restores resolution: borders 2546 to 2569 at every shift,
     against raw's 193 to 554.
  P11 HELD. Implied xi 0.626 to 0.628 at a true 0.7 and 0.730 to 0.749 at 0.9,
     flat across shifts. At 0.999 it beats raw in 20/20 seeds at xi 0.7
     (-48.4%) and 19/20 at 0.9 (-47.7%), which is better than clip_200's -38.5%
     at 0.9, and on a clean context it costs nothing (13/20, -0.8% and -2.0%).
     The price is at 0.99, where it is slightly worse than clip_200 (5/20 and
     3/20 seeds better, p = 0.041 and 0.0026, +0.8% and +1.4%).

  R1 HELD, harder than predicted. rank_emp's pinball at 0.999 at a shift of 50 is
     +1341% at xi 0.7 and +762% at 0.9, better in 0 of 20 seeds: with no tail
     extension every upper quantile saturates at the largest observed value, and
     under leverage that value is the contaminated one.
  R2 NOT HELD. rank_gpd at 0.999 on a clean context is within noise of raw
     (+5.1% at 0.7, -5.8% at 0.9) but stays worse than clip_200 (0.350 against
     0.298 at 0.7). The extension removes the saturation; it does not make the
     0.999 level competitive with a cap.
  R3 HELD, and by construction. Both rank arms keep borders between 3558 and
     3633 at every shift and the implied xi positive throughout (0.612 to 0.659
     at a true 0.7, 0.792 to 0.849 at 0.9). A rank cannot inflate an sd.
  R4 WRONG, in the direction worth having. The marginal tail shape does not drag
     the implied xi towards an average: at a true 0.9 rank_gpd reads 0.849, the
     closest of every arm measured, against clip_200's 0.804.
  R5 HELD, and worse than predicted, on real tables rather than here. See
     `unit_error_real.py`: 13.8% of cells are worse than raw by more than 100% at
     0.999, the worst by 86000x, all of them TabPFN-V3 on the tables whose
     marginal shape is near 1. (Recomputed from `unit_error_real.csv`, rank_gpd
     against raw on pb999 paired by dataset-model-repeat-factor: 13.810% of 420
     cells, worst +8619507%. An earlier draft of this block read "the worst by
     51889%", which no slice of the file reproduces.)

The finding that was not predicted at all: **rank_gpd is the sharpest arm at
0.99**, better than clip_200 in 15 to 20 of 20 seeds at every shift and both tail
indices (p = 0.0000 to 0.041, median -2.1% to -2.7%), including on clean contexts
where there is nothing to repair. That is not a repair; it is a better
parametrisation of the target, and it is what the fifth run tries to make safe.

FIFTH RUN: WHICH SCALE THE RANKS GO ONTO
---------------------------------------
The first rank run answered R1 to R4 and then failed on real tables: at 0.999,
13.8% of cells were worse than raw by more than 100%, the worst by 86000x, all of
them TabPFN-V3 on the tables whose marginal shape is near 1 (freMTPL2sev 0.84,
beMTPL97 0.83, MEPS 0.41, BlogFeedback 0.39). The cause is the composition, not
the fit: a normal forward map has a tail of exp(-z^2 / 2), so a Pareto inverse
turns an error in z into exp(xi * z^2 / 2). That is the same failure as tail_log,
one scale further out.

    rank_gpd_trim  normal scores, but the largest exceedance is dropped before the
                   GPD is fitted and the shape is capped at 0.95. Tests whether the
                   blow-ups are the fit.
    rank_exp       exponential scores, z = -log(1 - p), plus the same guard. The
                   amplification becomes exp(xi * z), and the largest score is
                   log(n + 1) = 7.6 rather than 3.3. Tests whether the blow-ups are
                   the scale.

Predictions:
  R6. rank_exp keeps rank_gpd's advantage at 0.99 and its worst case at 0.999 is
      smaller by orders of magnitude.
  R7. rank_gpd_trim reduces the blow-ups but does not remove them, because the
      sensitivity is in the composition rather than in the fitted shape. If it
      does remove them, the diagnosis above is wrong.
  R8. On the generator both stay within noise of rank_gpd at 0.99.

OUTCOME OF THE FIFTH RUN, 12.9.2026
-----------------------------------
20 seeds on the generator, and the same two arms on seven real tables through
`unit_error_real.py`. Both predictions about the cause are half right, and they
split cleanly along a line I did not anticipate.

  R6 HALF. On the generator the exponential scale does not improve pinball at
     0.999 against rank_gpd (36 of 80 seeds at xi 0.7, p = 0.43). On the seven
     real tables under a unit error it does, and where it matters: the share of
     cells more than 100% worse than raw falls from 16.2% to 5.2%, and the worst
     cell from +86195% to +84%.
  R7 HALF, and against the diagnosis. Dropping the largest exceedance and capping
     the shape at 0.95 does help, significantly: 55 of 80 seeds at 0.999 on the
     generator (p = 0.001, -15.3%) and 127 of 210 cells under the error
     (p = 0.003, -6.3%). So the sensitivity is not only in the composition; the
     fitted shape carries part of it. It does not remove the blow-ups either:
     11.9% of cells stay above +100%, worst +10191%.
  R8 HELD. Both stay within noise of rank_gpd at 0.99 (+0.08% to +0.22%).

The split to report: **the trim fixes the centre of the error distribution, the
scale fixes its tail**. Neither fixes both, and the cap arms have no blow-ups at
all (0.0% and 0.5% of cells, worst +1%).

WHICH PANEL THE ABOVE IS MEASURED ON. R6 to R8 stand on two complete files:
`clip_context.csv` and `unit_error_real.csv` both carry all 160 and all 420 cells
for every arm, rank_gpd_trim and rank_exp included, so every paired figure quoted
here compares arms on the same cells. `clip_context_real.csv` is NOT complete:
those two arms have 80 of the 120 cells the other nine arms have, because the run
is still going. Nothing above is quoted from it, and nothing should be until it
finishes -- a median over arms measured on different cell sets is the error
`tail_splice.py` records under the second run, where a 3% gap between thresholds
turned out to be 0.06% once the comparison was paired.

What survives as the recommendation is level-dependent, and that is the finding:
at 0.99 a rank-transformed context is the sharpest thing measured here, on the
generator at every shift and on real tables under an error; at 0.999 the derived
cap is the only safe arm. A repair chosen without naming the level is not a
repair.

HOW TO READ IT
--------------
  borders_in_data up, xi_implied positive     the fix works for the named reason
  borders_in_data up, xi_implied negative     resolution was not the mechanism
  pb999 on the clean context                  what clipping costs when there is
                                              nothing to protect against

Results in `clip_context.csv`.

RUNNING
-------
    python -u experiments/h3_repair/clip_context.py
    XI=0.7 SEEDS=3 SD_SHIFTS=1,20 CLIP_C=50 python -u clip_context.py
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm, rankdata

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

XI_LIST = [float(v) for v in os.environ.get("XI", "0.7,0.9").split(",")]
SD_SHIFTS = [float(v) for v in os.environ.get("SD_SHIFTS", "1,4,20,50").split(",")]
CLIP_C = [float(v) for v in os.environ.get("CLIP_C", "20,50,200").split(",")]
SEEDS_N = int(os.environ.get("SEEDS", "5"))
N_TRAIN = int(os.environ.get("N_TRAIN", "2000"))
N_TEST = int(os.environ.get("N_TEST", "900"))
N_EST = int(os.environ.get("N_EST", "1"))
MODEL = models.parse_list(os.environ.get("MODEL", "TabPFN-V3"))[0]
LEVELS = [0.5, 0.9, 0.99, 0.999]
OUT = os.environ.get("OUTPUT", "clip_context.csv")

COLUMNS = ["model", "xi", "variant", "clip_c", "sd_shift_target", "sd_shift",
           "seed", "n_est", "n_train", "n_test", "n_clipped",
           "borders_in_data", "same_bar_90_99", "xi_implied", "xi_ratio",
           "q50", "q90", "q99", "q999",
           "pb50", "pb90", "pb99", "pb999", "seconds", "reason"]
KEY = ["model", "xi", "variant", "clip_c", "sd_shift_target", "seed", "n_est",
       "n_train", "n_test"]


def robust_sd(y):
    """IQR / 1.349, the same estimator `robust_scale.py` uses."""
    q75, q25 = np.percentile(y, [75, 25])
    return float(max((q75 - q25) / 1.349, 1e-12))


def pinball(y, q, a):
    d = y - q
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


# The cap of tail_log, in robust sd above the median. Fixed before the second
# run at the value where hard clipping was seen to cost; a column, not a grid.
TAIL_LOG_C = 50.0

# The derived cap. Instead of a multiple of the robust sd, the cap is whatever
# brings the context's own sd shift down to TAU. A multiple does not transfer
# between tables: on beMTPL97 the largest value sits 79 robust sd above the
# median, so C = 200 can never bind there, while on BlogFeedback it sits at 213
# and C = 200 clips a value that is genuine. The sd shift is the quantity part
# two measured, it is scale free, and TAU = 1.2 is the top of the lowest bin of
# that design, fixed there before any repair was measured.
TAU = 1.2


def sd_cap(y, tau=TAU, iters=80):
    """The largest cap whose capped context has an sd shift of at most tau.

    A context already below tau comes back untouched, so the rule costs exactly
    nothing, by construction rather than by measurement, where there is nothing
    to repair. Otherwise bisection: a cap at the median is always feasible, since
    capping ties the top values together and the shift falls to 1, and a cap at
    the maximum is by definition not.
    """
    y = np.asarray(y, dtype=float)
    if metrics.sd_shift(y) <= tau:
        return float(y.max()), 0
    lo, hi = float(np.median(y)), float(y.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if metrics.sd_shift(np.minimum(y, mid)) <= tau:
            lo = mid
        else:
            hi = mid
    return lo, int((y > lo).sum())


def evt_trim(y, k_share=0.05, k_min=50, j_max=5, alpha=0.01):
    """Pull down the top values that the tail of the rest cannot have produced.

    Returns the new target and how many values were changed. See the docstring
    of this module for the test; `spacings[i - 1]` is i * (log X(i) - log
    X(i+1)) with X(1) the largest.
    """
    y = np.asarray(y, dtype=float)
    pos = np.sort(y[y > 0])[::-1]
    k = max(k_min, int(k_share * len(pos)))
    if len(pos) < k + 2:
        return y.copy(), 0
    logs = np.log(pos[:k + 1])
    spacings = np.arange(1, k + 1) * (logs[:-1] - logs[1:])
    xi_hat = float(spacings[j_max:].mean())
    if xi_hat <= 0:
        return y.copy(), 0
    threshold = -np.log(alpha / j_max)
    flagged = [i for i in range(1, j_max + 1) if spacings[i - 1] / xi_hat > threshold]
    if not flagged:
        return y.copy(), 0
    cap = pos[max(flagged)]              # the largest value not flagged
    return np.minimum(y, cap), int((y > cap).sum())


# The rank arms' own knobs, declared here rather than spelled out at four call
# sites. They are NOT in KEY: the variant name is, and each name pins its own
# (trim, xi_cap, scores), so the arms cannot collide with each other. What is not
# distinguished is the same arm measured at a different RANK_U_SHARE or
# RANK_K_MIN, so changing either needs a new OUTPUT rather than a resume over
# `clip_context.csv` -- rule 2's silent-skip, which the key alone cannot catch
# here. RANK_U_SHARE = 0.9 is the same free choice `tail_splice.py` turned into a
# measured grid once it decided a fixed 0.9 was as arbitrary as C = 200; it is
# still fixed here and has not been measured as a sensitivity.
RANK_U_SHARE = 0.9
RANK_K_MIN = 30
RANK_TRIM = 1                                  # rank_gpd_trim, rank_exp
RANK_XI_CAP = 0.95                             # rank_gpd_trim, rank_exp


def rank_gpd(y, u_share=RANK_U_SHARE, k_min=RANK_K_MIN, with_tail=True,
             scores="normal", trim=0, xi_cap=None):
    """See below; `scores` picks the scale the ranks are mapped onto."""
    return _rank_map(y, u_share, k_min, with_tail, scores, trim, xi_cap)


def _rank_map(y, u_share=RANK_U_SHARE, k_min=RANK_K_MIN, with_tail=True,
              scores="normal", trim=0, xi_cap=None):
    """Normal scores into the context, a spliced empirical-GPD map back out.

    The context target is replaced by its own normal scores. The largest row can
    then not stretch anything, because ranks are bounded: the sd shift of the
    transformed context is 1 by construction, whatever the original value was.
    That is where a rank transform normally stops being useful, since its inverse
    is the empirical quantile function and cannot return a value above the largest
    one observed, so every upper quantile saturates there. Above the u_share
    quantile the inverse therefore switches to a generalised Pareto fitted to the
    context's own exceedances, which makes the map unbounded. Below u the map is
    the empirical inverse and carries quantiles exactly; above it the shape of the
    marginal tail is imposed, since the GPD is fitted on the context's own margin
    and not per x, so the top u_share of the range comes back smoothed rather than
    exactly (up to 66% relative at the far end on a Pareto sample). That is the
    price of the extension, not an accident of it.

    The two branches are spliced at the probability where the EMPIRICAL inverse
    returns u, not at `1 - len(exc)/n`. Those are not the same number: with the
    exceedance count as the rate, the empirical branch ends one order statistic
    ABOVE u while the Pareto branch starts AT u, so the map stepped DOWN at the
    seam -- by 1.5x the local order-statistic gap on a Pareto(xi=0.9) context.
    A non-monotone inverse is not merely inelegant here: `one()` feeds
    `back(borders)` to `np.searchsorted`, which is undefined on an unsorted array,
    so `same_bar_90_99` was being read off a garbage bin index for every rank arm.

    `with_tail=False` is the same map without that extension, which isolates how
    much of the result is the ranks and how much is the tail.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    p = rankdata(y, method="average") / (n + 1)     # ties share a rank, and a score

    # Which scale the ranks are mapped onto decides how an error in the model's
    # own quantile is amplified on the way back. With normal scores the upper tail
    # of the forward map decays like exp(-z^2 / 2), so composing it with a Pareto
    # inverse amplifies an error in z like exp(xi * z^2 / 2): a prediction at
    # z = 4.2 instead of 3.3 comes back three orders of magnitude too high, which
    # is what the first run of this arm did on MEPS and BlogFeedback (13.8% of
    # cells worse than +100% at 0.999, the worst by 86000x). Exponential scores
    # decay like exp(-z), so the amplification is exp(xi * z), and the largest
    # score is log(n + 1) = 7.6 rather than 3.3. The leverage is still gone either
    # way, because both scales are bounded by the rank.
    if scores == "normal":
        z, to_p = norm.ppf(p), lambda q: norm.cdf(q)
    elif scores == "exponential":
        z, to_p = -np.log1p(-p), lambda q: 1.0 - np.exp(-np.maximum(q, 0.0))
    else:
        raise ValueError(scores)

    order = np.argsort(p, kind="mergesort")
    xp, fp = p[order], y[order]                    # the empirical inverse, as points
    u = float(np.quantile(y, u_share))
    exc = np.sort(y[y > u]) - u
    # The tail probability the Pareto branch is given, taken as the point where the
    # empirical branch itself reaches u rather than as len(exc) / n. Both estimate
    # P(Y > u); only this one makes the two branches meet, because it is defined by
    # the map that hands over. See the docstring on the seam.
    zeta = max(1.0 - float(np.interp(u, fp, xp)), 1e-12)
    xi = scale = np.nan
    if with_tail and len(exc) >= k_min:
        try:
            # `trim` drops that many of the largest exceedances before fitting, so a
            # contaminated top value does not set the shape it is then judged by;
            # `xi_cap` bounds the shape, because above 1 the extrapolated quantile
            # grows faster than any sample can support.
            fit_on = exc[:-trim] if trim else exc
            xi, _, scale = genpareto.fit(fit_on, floc=0.0)
            if xi_cap is not None:
                xi = min(float(xi), float(xi_cap))
        except Exception:
            xi = np.nan

    def back(q):
        pr = np.clip(to_p(np.asarray(q, dtype=float)), 1e-12, 1 - 1e-12)
        out = np.interp(pr, xp, fp)
        if np.isfinite(xi):
            hi = pr > 1 - zeta
            if hi.any():
                r = (1 - pr[hi]) / zeta
                out[hi] = (u + scale * np.log(1 / r) if abs(xi) < 1e-8
                           else u + scale / xi * (r ** (-xi) - 1))
        return out

    # Every value is rewritten, not none of them. `transform` promises "how many
    # context values the arm changed", and reporting 0 made a rank arm's n_clipped
    # indistinguishable from raw's -- in the column the first three runs used to
    # argue that sd_cap costs nothing where there is nothing to repair.
    return z, back, n


def transform(arm, c, y):
    """The context target an arm fits on, the map back for its quantiles, and how
    many context values the arm changed."""
    y = np.asarray(y, dtype=float)
    med, rsd = float(np.median(y)), robust_sd(y)
    same = lambda q: q
    if arm == "raw":
        return y, same, 0
    if arm == "clip":
        cap = med + c * rsd
        return np.minimum(y, cap), same, int((y > cap).sum())
    if arm == "log":
        if (y <= 0).any():
            raise ValueError("log needs a positive target")
        return np.log(y), np.exp, 0
    if arm == "tail_log":
        cap, d = med + c * rsd, c * rsd
        over = y > cap
        z = y.copy()
        z[over] = cap + d * np.log1p((y[over] - cap) / d)
        back = lambda q: np.where(q > cap, cap + d * np.expm1((q - cap) / d), q)
        return z, back, int(over.sum())
    if arm == "evt_trim":
        z, n = evt_trim(y)
        return z, same, n
    if arm == "sd_cap":
        # `c is None`, not `c or TAU`: tau = 0.0 is a real request (it caps at the
        # median) and `or` silently turned it into TAU, so the arm ran at a
        # different setting than the clip_c column recorded.
        cap, n = sd_cap(y, TAU if c is None else c)
        return np.minimum(y, cap), same, n
    if arm == "rank_emp":
        return rank_gpd(y, with_tail=False)
    if arm == "rank_gpd":
        return rank_gpd(y)
    if arm == "rank_gpd_trim":
        return rank_gpd(y, trim=RANK_TRIM, xi_cap=RANK_XI_CAP)
    if arm == "rank_exp":
        return rank_gpd(y, scores="exponential", trim=RANK_TRIM, xi_cap=RANK_XI_CAP)
    raise ValueError(arm)


def fit_predict(Xc, yc, Xte, seed):
    """Quantiles in the target's own units, plus the fitted raw-space borders."""
    from tabpfn import TabPFNRegressor
    kw = dict(n_estimators=N_EST, device="cpu", random_state=seed,
              ignore_pretraining_limits=True)
    if MODEL in models.TABPFN_PATHS:
        kw["model_path"] = models.TABPFN_PATHS[MODEL]
    m = TabPFNRegressor(**kw)
    m.fit(Xc, yc)
    q = np.stack([np.asarray(a, dtype=float) for a in
                  m.predict(Xte, output_type="quantiles", quantiles=LEVELS)], axis=1)
    borders = m.raw_space_bardist_.borders.detach().double().numpy()
    return q, borders


def one(xi, variant, c, target, seed):
    rng = np.random.default_rng(seed)
    p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = p.X, p.y
    pt = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, yte = pt.X, pt.y

    # The same treatment as grid_occupancy: one row at the centre of x, sized to
    # hit the sd shift exactly rather than as a multiple of a random maximum.
    if target <= 1.0:
        Xc, yc, shift = Xtr, ytr, 1.0
    else:
        y0 = metrics.y0_for_sd_shift(ytr, target)
        if not np.isfinite(y0):
            raise RuntimeError(f"sd shift {target} unreachable on this seed")
        Xc = np.vstack([Xtr, np.zeros((1, Xtr.shape[1]))])
        yc = np.concatenate([ytr, [y0]])
        shift = float(metrics.sd_shift(yc))

    med, rsd = float(np.median(yc)), robust_sd(yc)
    n_clipped = 0
    if variant == "raw":
        q, borders = fit_predict(Xc, yc, Xte, seed)
    elif variant == "robust":
        q, borders = fit_predict(Xc, (yc - med) / rsd, Xte, seed)
        q, borders = q * rsd + med, borders * rsd + med
    elif variant in ("clip", "log", "tail_log", "evt_trim", "sd_cap",
                     "rank_emp", "rank_gpd", "rank_gpd_trim", "rank_exp"):
        y_fit, back, n_clipped = transform(variant, c, yc)
        q, borders = fit_predict(Xc, y_fit, Xte, seed)
        # exp and expm1 of the outer borders overflow to inf; that is where those
        # borders are, and the counts below only compare against the data range.
        with np.errstate(over="ignore", invalid="ignore"):
            q, borders = back(q), back(borders)
    else:
        raise ValueError(variant)

    lo, hi = float(ytr.min()), float(ytr.max())
    col = LEVELS.index
    b90 = np.clip(np.searchsorted(borders, q[:, col(0.9)]) - 1, 0, len(borders) - 2)
    b99 = np.clip(np.searchsorted(borders, q[:, col(0.99)]) - 1, 0, len(borders) - 2)
    num, den, mid = q[:, col(0.99)], q[:, col(0.9)], q[:, col(0.5)]
    ok = (den - mid) > 1e-12
    xi_imp = (metrics.xi_from_residual_ratio(
        float(np.median((num[ok] - mid[ok]) / (den[ok] - mid[ok]))))
        if ok.any() else np.nan)

    # The raw-ratio reading grid_occupancy.py uses. It is biased low whenever
    # the distribution has a location component (metrics.xi_from_ratio says as
    # much), so both are kept: the gap between them is how much of the negative
    # xi in part two belongs to the measure rather than to the model.
    okr = den > 1e-12
    xi_rat = (metrics.xi_from_ratio(float(np.median(num[okr] / den[okr])))
              if okr.any() else np.nan)

    r = dict(sd_shift=shift, n_clipped=n_clipped, xi_ratio=xi_rat,
             borders_in_data=int(((borders >= lo) & (borders <= hi)).sum()),
             same_bar_90_99=float((b90 == b99).mean()),
             xi_implied=xi_imp, reason="")
    for a, name in zip(LEVELS, ("50", "90", "99", "999")):
        r[f"q{name}"] = float(np.median(q[:, col(a)]))
        # The test set is always clean: this is the error a user of the interval
        # sees, whatever was done to the context.
        r[f"pb{name}"] = pinball(yte, q[:, col(a)], a)
    return r


def cells():
    for xi in XI_LIST:
        for target in SD_SHIFTS:
            for seed in [7000 + 1000 * i for i in range(SEEDS_N)]:
                yield xi, "raw", 0.0, target, seed
                yield xi, "robust", 0.0, target, seed
                for c in CLIP_C:
                    yield xi, "clip", c, target, seed
                yield xi, "log", 0.0, target, seed
                yield xi, "tail_log", TAIL_LOG_C, target, seed
                yield xi, "evt_trim", 0.0, target, seed
                yield xi, "sd_cap", TAU, target, seed
                yield xi, "rank_emp", 0.0, target, seed
                yield xi, "rank_gpd", 0.0, target, seed
                yield xi, "rank_gpd_trim", 0.0, target, seed
                yield xi, "rank_exp", 0.0, target, seed


def main():
    done = append.done(OUT, KEY)
    if done:
        print(f"resuming, {len(done)} cells already measured", flush=True)
    t0 = time.time()
    for xi, variant, c, target, seed in cells():
        k = dict(model=MODEL, xi=xi, variant=variant, clip_c=c,
                 sd_shift_target=target, seed=seed, n_est=N_EST,
                 n_train=N_TRAIN, n_test=N_TEST)
        if append.key(k, KEY) in done:
            continue
        t1 = time.time()
        try:
            r = one(xi, variant, c, target, seed)
        except Exception as e:
            r = dict(reason=f"{type(e).__name__}: {e}"[:110])
        r.update(k, seconds=round(time.time() - t1, 1))
        append.write(OUT, r, COLUMNS)
        tag = variant if variant != "clip" else f"clip_{c:g}"
        msg = r.get("reason") or (
            f"borders {r['borders_in_data']:>4}  xi {r['xi_implied']:+.3f}  "
            f"pb99 {r['pb99']:.4f}  pb999 {r['pb999']:.4f}  clipped {r['n_clipped']}")
        print(f"  xi={xi} sd={target:>4g} {tag:9s} s={seed}  {msg}  [{r['seconds']}s]",
              flush=True)
    report(t0)


def report(t0=None):
    path = paths.result(OUT)
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        print("no successful rows")
        return
    d["arm"] = np.where(d.variant == "clip", "clip_" + d.clip_c.map("{:g}".format),
                        d.variant)
    for col in ("borders_in_data", "xi_implied", "xi_ratio", "pb999"):
        print(f"\n=== {col}, median by shift ===")
        print(d.pivot_table(index=["xi", "sd_shift_target"], columns="arm",
                            values=col, aggfunc="median").round(4).to_string())
    raw = d[d.variant == "raw"].set_index(["xi", "sd_shift_target", "seed"])
    rob = d[d.variant == "robust"].set_index(["xi", "sd_shift_target", "seed"])
    j = raw[["q99", "xi_implied"]].join(rob[["q99", "xi_implied"]], rsuffix="_rob",
                                        how="inner")
    if len(j):
        gap = float((j.q99 - j.q99_rob).abs().max() / j.q99.abs().max())
        print(f"\nP1 control, robust vs raw: max relative gap in Q(0.99) = {gap:.2e}"
              f"  ({'holds' if gap < 1e-3 else 'FAILS: the reading of fit is wrong'})")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
