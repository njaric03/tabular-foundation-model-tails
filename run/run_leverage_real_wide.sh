#!/bin/sh
# Widen the real-data leverage result from three datasets to nine.
#
# WHY THIS ONE FIRST
# ------------------
# The strongest claim in the second part currently rests on ONE dataset that has
# both leverage and a heavy tail: freMTPL2sev, sd shift up to 19.9x, Hill 0.79,
# where TabICLv2 loses 15.5% at a shift of 2 to 4x and 28.3% above 4x. On
# Brazilian_houses, which has leverage up to 6.0x but a lighter tail (Hill 0.30),
# the effect disappears. That is a hypothesis with two points, and
# findings/NALAZI.md section 8 names it as the largest open weakness of the work.
#
# The panel is widened by lowering the entry threshold from a 2.0x sd shift to
# 1.5x, which takes the datasets from 3 to 9. The extra six are lighter-tailed,
# so they are the informative cases: if the effect follows the tail index rather
# than the leverage, that is exactly what should show up.
#
# Either outcome is worth having before the thesis is written. A nine-point
# pattern turns the hypothesis into a result; no pattern narrows the claim to
# one dataset honestly, which is better than defending it at the viva.
#
# COST
# ----
# About 4 hours on CPU. Four models, nine datasets, four leverage buckets, five
# subsamples each. Resumable: every cell is keyed on
# (dataset, model, bin, repeat, n_est), so it can be stopped and restarted.
#
#     sh run/run_leverage_real_wide.sh
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=python

SETS="freMTPL2sev,Brazilian_houses,OnlineNewsPopularity,nyc-taxi-green-dec-2016,\
Job_Profitability,stock_fardamento02,Buzzinsocialmedia_Twitter,\
Airlines_DepDelay_10M,particulate-matter-ukair-2017"

MODELS=GBM,XGB,TabICLv2,TabPFN-V3 DATASETS="$SETS" \
  N_EST=1 KVOTA=5 OUTPUT=prevalence_models_wide.csv \
  $PY -u experiments/h2_leverage/prevalence_models.py
