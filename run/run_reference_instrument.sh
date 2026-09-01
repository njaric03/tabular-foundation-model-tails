#!/bin/sh
# The real-data reference for the shape, measured with the model's own instrument,
# and the null distribution of the feature selection.
#
# WHY
# ---
# The real-data half of the first part compares two quantities that are not
# produced the same way. The reference tail index per tercile comes from a Hill
# estimator on residuals; the model's comes from inverting the ratio
# Q(0.99)/Q(0.9). The share is their quotient. On synthetic data the difference
# between those instruments was checked against a known xi; on real data there is
# no known xi, so it has never been checked at all.
#
# The second question is the feature. It is chosen as the column with the largest
# |xi(t3) - xi(t1)| on the fit split, a maximum over every column of the table.
# The maximum of many noisy statistics is large even when nothing is there, which
# is the mechanism behind the share range of -2.80 to 6.16 that the findings
# currently attribute to "a weak reference gradient" without naming a cause. This
# run measures that null distribution directly, by repeating the whole selection
# on permuted targets, where no dependence of the tail on the covariates exists
# by construction.
#
# WHAT IT SETTLES
# ---------------
# Which of the ten datasets carry a shape gradient stronger than their own
# selection noise. Those are the datasets the rank claim may be quoted on. A
# first check on `diamonds` -- which contributes 2 of the 59 comparisons -- gives
# a Hill gradient of +0.051 against a null 95th percentile of 0.180, and the two
# instruments disagree in sign (+0.051 against -0.218). One dataset is not a
# verdict, but it is the reason to run the rest.
#
# COST
# ----
# About 90 minutes, and NO MODEL IS CALLED: this is estimators over the data
# only. The permutation count is the expensive part; N_PERM=20 is enough for a
# 95th percentile, N_PERM=50 for a smoother one.
#
#     sh run/run_reference_instrument.sh
set -x
PY=python

SEEDS=3 N_PERM=20 N_BOOT=200 OUTPUT=reference_instrument.csv \
  $PY -u experiments/h1_shape_vs_scale/reference_instrument.py
