#!/bin/sh
# Empirical coverage of the predicted quantiles, clean and under one leverage row.
#
# WHY
# ---
# The work opens with the claim that the predictive distribution of these models
# is read directly, for intervals and for risk, and then never measures the one
# number that follows from it: what fraction of held-out outcomes exceeds the
# predicted Q(0.99). Every existing measure is indirect -- an implied tail index
# from a quantile ratio, pinball, threshold-weighted CRPS. Those say how good the
# distribution is in some metric; none says whether the interval holds its
# nominal level.
#
# It is also the only table in which the first and the second part of the thesis
# meet on one number. Coverage is measured at dose 0 and dose 100, so the price
# of a single leverage row in the context is expressed in the unit a user of the
# interval actually cares about, rather than as a percentage shift of a quantile.
#
# WHAT TO EXPECT
# --------------
# Nominal 0.99 means 60 exceedances out of 6000. The count is recorded next to
# every coverage precisely so the reader can see when a level is not measurable:
# at 0.999 the expectation is six, and the difference between four and eight is
# noise, not a finding. That is the same measurability limit the third part
# argues about the public benchmark, applied to this work's own numbers.
#
# COST
# ----
# About 5 hours on CPU: ten datasets, four models, two doses, three seeds, with
# 3000 context and 6000 test rows. Resumable, keyed on
# (dataset, model, seed, dose, n_est).
#
# Add "sint" to DATASETS for the synthetic control, where the true coverage is
# known by construction and any deviation is the model rather than the data.
#
#     sh run/run_coverage.sh
#
# TabPFN needs a Prior Labs account before its weights load; see README, Setup.
# Until that is done, run the rest and add TabPFN-V3 later, since every cell is keyed:
#
#     MODELS=TabICLv2,GBM sh run/run_coverage.sh
#
set -x
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
PY=${PY:-venv-tfm/Scripts/python.exe}   # another environment: PY=... sh run/...

SETS="freMTPL2sev,OnlineNewsPopularity,diamonds,particulate-matter-ukair-2017,\
Buzzinsocialmedia_Twitter,CPS1988,218_house_8L,superconduct,houses,house_16H"

MODELS=${MODELS:-GBM,XGB,TabICLv2,TabPFN-V3} DATASETS="$SETS" \
  DOSES=0,100 SEEDS=3 N_EST=1 OUTPUT=coverage.csv \
  $PY -u experiments/h1_shape_vs_scale/empirical_coverage.py
