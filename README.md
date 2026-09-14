# The upper tail of the predictive distribution in tabular foundation models

[![tests](https://github.com/njaric03/tabular-foundation-model-tails/actions/workflows/tests.yml/badge.svg)](https://github.com/njaric03/tabular-foundation-model-tails/actions/workflows/tests.yml)

Master's thesis, Department of Mathematics and Informatics, University of Novi Sad: the
code, the inputs and the measured results. The write-up is in Serbian and kept outside
this repository.

Tabular foundation models lead the [TabArena](https://tabarena.ai) leaderboard, and their
predictive distribution is increasingly read directly, for intervals and for risk. This
repository asks how far that output can be trusted in the upper tail: what the models
track, what one row of the context does to it, and what can be repaired without
retraining. Five foundation models (TabPFN, TabICL, TabDPT, TabFM, EXAONE-Tabular) are
measured against gradient boosting, XGBoost and CatBoost, all on CPU, under the rules in
`RULES.md`.

## Findings

**1. The conditional scale is tracked, the conditional tail shape is not.** On a generator
whose tail index varies with a covariate, the models capture about 94% of the change in
scale and 40 to 50% of the change in shape, with a bootstrap spread of 0.15 to 0.20
across seeds. The sample is not the limit: an extreme-value oracle on the same 2000 rows
recovers 98%. The dissociation holds across four tail families (60 of 60 runs), with a
location-invariant estimator, after reseeding, and on real data by rank: 8 of 8 target
vectors, p = 0.0039 by a sign test. Ten dataset names were loaded, and `218_house_8L` and
`house_16H` serve one target, so "nine of nine datasets" counted one unit twice; the
row-level p = 3.4e-11 treats nested seeds and models as independent and is not quoted.
Read as a slope across every usable feature and corrected for noise in the reference, the
real-data shape share is 0.40 for TabPFN-V3 and 0.53 for TabICLv2, against 0.42 to 0.43
on the generator.

**2. One leverage row in the context changes the predictive distribution.** None of five
packages protects the target: it is standardised by its plain mean and sd, while the
features go through outlier-resistant transforms. One added row moves Q(0.99) by tens of
percent, and the predicted mean of some models by an order of magnitude, sign included.
Tree ensembles do not standardise the target and keep the tail family (CatBoost's implied
tail index rises with the dose from 0.34 to 0.43 while TabPFN's falls from 0.43 to -0.03),
but they are not inert: CatBoost's local Q(0.99) moves by up to 74%, sklearn GBM's by 3.9%.
The trigger is a data error rather than a heavy tail: an sd shift of 4 or more occurs in
2% of 99 public datasets.

For TabPFN the mechanism is one expression in `regressor.py`: the raw-space grid is
`borders * y_train_std_ + y_train_mean_`, with 341 borders per unit in the body and three
between 50 and 128 sd. A row that inflates the sd stretches the grid while the data stays
put, the data then spans few bars, and linear interpolation inside a bar reads as a
negative tail index. The half-normal tail beyond the last border is never read by the
quantile function the package calls.

**3. After pre-training the weakness can be reduced, not removed, and every repair has a
price.** Swapping the scale estimator after the fit makes everything worse, but only
because `fit` has already normalised the context by the contaminated sd; it does not test
a robust scale. What works acts before the fit. A cap that brings the context's own sd
shift to 1.2 restores the grid and beats the raw model at 0.999 in 20 of 20 seeds, and a
rank-transformed context is the sharpest option at 0.99 but can fail badly at 0.999 on
real tables, so the right repair depends on the level. On the output side, a GPD fitted to
the model's conformity scores brings coverage at 0.99 back under a corrupted row without
knowing which row it is, at the cost of sharpness. The one free recommendation is
`n_estimators=1`, worth about 27% of implied tail index at a true 0.7 and 0.9 on ten
seeds and 4% at 0.3; the loss happens at the first aggregation and only in TabPFN-V3.

Everything above is indirect, so coverage is measured on its own: the foundation models
hold their nominal level better than the tree controls, and one leverage row costs a
fraction of a point at 0.99 while moving TabPFN by five points at the median.

## Models

| model | quantiles | why it is here |
|---|---|---|
| TabPFN-V3 | yes | top of TabArena; bar-distribution head |
| TabICLv2 | yes | 999-quantile head |
| EXAONE-Tabular | yes | second on TabArena; a quantile head like TabICL's |
| TabDPT | from its bin head | 2048 fixed bins, read out in `common/adapters/tabdpt.py` |
| TabFM | no, mean only | first on TabArena in regression, publishes no distribution |
| XGBoost, sklearn GBM | yes | controls that do not standardise the target |
| CatBoost | yes | best tree model in TabArena regression; tracks conditional shape at 0.60 |
| TabPFN-v2.5, v2.6 | yes | selected through `TABPFN_PATHS`, measured in part three |

LimiX and Mitra return a point prediction only, and RealTabPFN-2.5 is not public. TabPFN-V3
and TabICLv2 carry most of the results; EXAONE, TabDPT and TabFM have the fewest rows, and
the drivers in `run/` fill that gap.

## Structure

| folder | contents |
|---|---|
| `common/` | all shared code, below |
| `experiments/` | one folder per hypothesis (`h1_shape_vs_scale`, `h2_leverage`, `h3_repair`), plus `datasets/` and `side/` |
| `run/` | drivers for the long runs, with the parameters and their reasons |
| `results/` | one CSV per measurement, mirroring `experiments/`, plus `provenance.csv` |
| `analysis/` | notebooks that recompute every quoted number from the CSVs |
| `data/` | inputs: dataset lists, OpenML ids, fingerprints of the tables |
| `tests/` | tests of `common/`, and an import of every experiment |
| `requirements/` | frozen package sets per virtualenv |
| `figures/` | thesis figures |

`common/`:

| module | contents |
|---|---|
| `models` | the only place a model is built and asked for quantiles or a mean |
| `adapters/` | EXAONE, TabDPT and TabFM, which need patches or special loading |
| `datasets`, `external` | loading, preparation and fingerprints of the real tables |
| `generator` | the synthetic GPD generator |
| `metrics`, `stats` | tail-index estimators, scoring rules, clustered paired tests |
| `gradients`, `leverage`, `transforms`, `evt`, `selection` | the procedures several experiments share |
| `append`, `runner`, `env`, `provenance`, `files` | resumable CSV writing, the measurement loop, environment knobs, the record of what produced a row |
| `paths`, `quiet`, `tables` | repo-relative paths, the warning filter, notebook aggregations |

## Setup

TabDPT, TabFM and EXAONE need mutually incompatible torch versions, so there are several
virtualenvs, and the package in `common/` is installed into each:

| virtualenv | models |
|---|---|
| `venv-tfm` | TabPFN, TabICL, XGBoost, CatBoost |
| `venv-tabfm` | TabFM, EXAONE |
| `venv-tabdpt` | TabDPT |
| `venv-data` | none; `pyreadr` for converting the external tables |

```bash
python -m venv venv-tfm
venv-tfm/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv-tfm/Scripts/python.exe -m pip install "tabpfn==8.4.0" "tabicl==2.1.1" xgboost pandas scipy scikit-learn
venv-tfm/Scripts/python.exe -m pip install -e .
```

The pins matter: part two is a statement about the preprocessing of exactly these
versions, and pip serves newer ones by default. TabPFN 8.4.0 loads its weights only after a
Prior Labs login in a browser, which only the account holder can do; every driver takes a
`MODELS=` override, so TabPFN-V3 can be added to a run later. TabFM's 6.6 GB of weights
come from `run/download_tabfm.sh` and need about 10 GB free.

What produced a row is recorded twice: `results/provenance.csv` gets package versions,
commit, interpreter and every environment knob from the first write of each process, and
`sh run/freeze_envs.sh` writes `requirements/<env>.txt` for every virtualenv it finds.

## Running

From the repository root, with `-u` so output is not buffered. Settings are environment
variables, and each one that changes a result is a column of the output:

```bash
MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv python -u experiments/h1_shape_vs_scale/shape_of_x.py
sh run/tabpfn_reseed.sh
```

Runs are resumable: a cell whose key is already in the output is skipped.

Tests and lint need numpy, scipy, pandas, scikit-learn, matplotlib and ruff:

```bash
python -m pytest tests -q
ruff check .
```

`TFM_STRICT=1` turns a numeric warning into an error and a changed dataset fingerprint
into a failure; worth one pass after touching an estimator.
