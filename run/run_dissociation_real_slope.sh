#!/bin/sh
# The real-data dissociation without a chosen feature and without a fragile quotient.
#
# WHY THIS REPLACES THE EXISTING REAL-DATA MEASURE
# ------------------------------------------------
# `dissociation_real.py` picks the feature with the largest tail gradient on the
# fit split and then divides the model's gradient by the reference one. Two
# measurements in this repository say that is not sound enough to carry the
# claim:
#
#   * `reference_instrument.csv`: on 9 of 10 datasets the selected gradient does
#     not clear the 95th percentile of what the selection produces from permuted
#     targets. The selection is doing most of the work.
#   * the same file's bootstrap intervals cross zero on most datasets, which is
#     the same fact stated a second way.
#
# And the quotient is not neutral between the two halves. When the reference
# shape gradient is noise, the shape share is far from 1 by construction, while
# the scale share along the same feature is genuine and near 1. The rank claim is
# therefore tilted towards the finding.
#
# WHAT THIS MEASURES INSTEAD
# --------------------------
# One fit per (dataset, model, seed), then the same predictions read through the
# terciles of EVERY usable feature. The share becomes the slope of the model's
# gradient on the reference gradient across features, through the origin: no
# selection, no division by a single small number.
#
# The slope is reported raw and corrected for measurement error in the reference,
# whose variance is estimated by bootstrap per feature. That correction is the
# point of the design, not a refinement: the shape reference is far noisier than
# the scale reference -- a first run on `houses` gives 0.164 of the reference
# variance as measurement error for the shape against 0.004 for the scale -- and
# error in a regressor pulls a slope towards zero. Without the correction, a
# noisier instrument alone could manufacture the entire dissociation.
#
# WHAT COUNTS AS WHICH ANSWER
# ---------------------------
#   scale slope near 1, shape slope small, both after correction
#       the dissociation survives the strongest available test on real data
#   both slopes move together under the correction
#       the difference was never the instrument
#   the shape slope rises to meet the scale slope once corrected
#       the real-data half of H1 was attenuation, and that is the finding
#
# The synthetic half is unaffected either way: there the true xi(x) is known.
#
# COST
# ----
# About 90 minutes on CPU: ten datasets, three models, three seeds, one fit each.
# The bootstrap is over the data, not the model, so it costs seconds per cell.
#
#     sh run/run_dissociation_real_slope.sh
#
# TabPFN needs a Prior Labs account before its weights load; see README, Setup.
# Until that is done, run the rest and add TabPFN-V3 later, since every cell is keyed:
#
#     MODELS=TabICLv2,GBM sh run/run_dissociation_real_slope.sh
#
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...

MODELS=${MODELS:-TabICLv2,TabPFN-V3,GBM} SEEDS=3 N_EST=4 N_BOOT=100 \
  OUTPUT=dissociation_real_slope.csv \
  $PY -u experiments/h1_shape_vs_scale/dissociation_real_slope.py
