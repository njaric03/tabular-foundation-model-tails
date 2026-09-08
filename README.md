# The upper tail of the predictive distribution in tabular foundation models

Master's thesis, Department of Mathematics and Informatics, University of Novi Sad.
Measurements, results and text.

Tabular foundation models now sit at the top of the [TabArena](https://tabarena.ai)
leaderboard, and their predictive distribution is increasingly read directly, for
intervals and for risk. This repository asks how far that output can be trusted in the
upper tail: what the models track, what one row in the context can do to it, and what can
be repaired after pre-training. Five foundation models are measured (TabPFN, TabICL,
TabDPT, TabFM, EXAONE-Tabular) against gradient boosting and XGBoost as controls, all on
CPU, without retraining any of them.

The detailed write-up is in Serbian, the language of the thesis, and is kept outside this
repository. What lives here is the code, the inputs and the measured CSVs, under the seven
operating rules in `RULES.md`.

## What the three parts claim

**1. The conditional scale is tracked, the conditional tail shape is not.**
On a generator where the true tail index varies with a covariate, the models capture
about 94% of the change in scale and only about 40 to 50% of the change in shape, with
a bootstrap spread of 0.15 to 0.20 across seeds. This is not a limit of the sample: an
extreme-value oracle on the same 2000 rows recovers 98%. The dissociation survives four
distribution families (60 of 60 runs), a location-invariant estimator, a reseeding
control, and it holds by rank on real data (8 of 8 target vectors, p = 0.0039 by a
sign test at the level of the target, which is the free unit; ten dataset names were
loaded, two of which -- `218_house_8L` and `house_16H` -- serve one target under two
feature sets, so counting nine of nine datasets counted one unit twice; the row-level
p = 3.4e-11 counts 3 seeds and 3 models per dataset as independent and should not be
quoted). The number travels too, but only once the quotient is dropped: the feature
carrying the gradient was chosen as a maximum over columns, and on 9 of 10 datasets that
gradient does not clear its own selection noise. Read instead as the slope of the model's
gradient on the reference across every usable feature, corrected for the noise in the
reference, the shape share on real data is 0.40 for TabPFN-V3 and 0.53 for TabICLv2,
against 0.42 to 0.43 on the generator.

**2. The predictive distribution is not robust to a single leverage row in the context.**
Read from the source of five packages: none of them protects the target variable, which
is standardised by its plain mean and standard deviation, while features are as a rule
put through outlier-resistant transforms. One added row can move Q(0.99) by tens of
percent, and the predicted mean of some models by an order of magnitude including a sign
change. Tree ensembles, which do not standardise the target, do not lose the tail: CatBoost's
implied tail index rises with the dose, from 0.34 to 0.43, while TabPFN's falls from 0.43
to -0.03. They are not inert, though, and an earlier version of this page said they were:
CatBoost's local Q(0.99) moves by up to 74% where sklearn GBM moves by 3.9%. Trees absorb
the row; the foundation models lose the family.
The trigger is a data error rather than a heavy tail as such: a standard-deviation shift
of 4x or more occurs in 2% of 99 public datasets.

**3. After pre-training the weakness can be reduced, not removed, and the cost is
measurable.** Winsorizing the target helps at alpha = 0.99 under contamination, by up to
23.5%, but consistently hurts TabICL at alpha = 0.999. Replacing the scale estimator with
a robust one makes everything worse by an order of magnitude, because the output support
is calibrated on exactly that non-robust scale: the vulnerability and the safety mechanism
are the same thing. The one free recommendation is `n_estimators=1`, worth about 27% of
implied tail index at a true 0.7 and 0.9 on ten seeds, and only 4% at 0.3, so it applies
when the tail is heavy. The whole loss happens at the first aggregation and only in
TabPFN-V3; TabICLv2 does not respond to ensemble size at all.

Everything above is indirect, so the fraction of held-out outcomes that falls below the
predicted quantile is measured on its own. There the foundation models hold their nominal
level better than the tree controls, and one leverage row costs a fraction of a point at
0.99 while moving TabPFN five points at the median.

The first two weaknesses share a birthplace, the layer that processes the target, but not
a trigger. They are not causally linked, and the thesis does not claim they are. For
TabPFN the birthplace is now a single expression, read out of `regressor.py`: the largest
value the model can report is `borders[-1] * y_train_std_ + y_train_mean_`, a pre-training
constant times a non-robust scale. The border grid behind it holds 341 borders per unit in
the body and three between 50 and 128 standard deviations, so above about 17 the whole tail
is five bars, and the reported 0.999 quantile jumps from 3.3 to 105.8 on a half-per-mille
change in one bar. That is the mechanism behind the bimodal responses reported in part two.
The head's tail is half-normal, lighter than TabICL's exponential, and the quantile function
the package actually calls does not read even that. Reading it costs no GPU and no
training.

## Models

| model | quantiles | why it is here |
|---|---|---|
| TabPFN-V3 | yes | top of TabArena; bar-distribution head |
| TabICLv2 | yes | 999-quantile head, different output construction |
| EXAONE-Tabular | yes | second on TabArena; shares TabICL's head |
| TabDPT | via its bin head | 2048 fixed bins, recovered in `common/adapters/tabdpt.py` |
| TabFM | no, mean only | first on TabArena in regression, but publishes no distribution |
| XGBoost, sklearn GBM | yes | controls that do not standardise the target |
| CatBoost | yes | best tree model in regression on TabArena; tracks conditional shape at 0.60, better than any foundation model here |
| TabPFN-v2.5, v2.6 | yes | wired up through `TABPFN_PATHS`; **no rows in `results/`** |

Left out on purpose: LimiX and Mitra return a point prediction only, so there is nothing
distributional to measure; RealTabPFN-2.5 is not public. The boundary of this work is
which models expose a predictive distribution at all.

How much of each model is actually measured is uneven, and the table above does not show
it. Counted over the `model` column of every file in `results/`, after the runs of
2026-09-08: TabPFN-V3 in 38, TabICLv2 in 29, GBM in 23, EXAONE in 7, XGBoost in 3, CatBoost
in 3, TabDPT in 3, TabFM in 1. Five foundation models are wired up; two and a half carry
the results, and EXAONE and TabFM are the two that a reader will ask about first. The
runners that close the gap are in `run/`, each with its choice of parameters argued in the
header.

## Structure

| folder | what is inside |
|---|---|
| `common/` | the only shared code: paths, CSV append, generator, model dispatcher, metrics, clustered tests, dataset loading, provenance, model adapters |
| `experiments/` | one folder per hypothesis, plus `datasets/` and `side/` |
| `tests/` | property tests for the estimators, the generator and the append guards |
| `run/` | drivers for the long measurement runs, with the choice of parameters argued in the header |
| `data/` | inputs: the ScoringBench name and id lists, the selected datasets |
| `results/` | one CSV per measurement, mirroring `experiments/`, plus `provenance.csv` |
| `requirements/` | frozen package sets per virtualenv, written by `run/freeze_envs.sh` |
| `analysis/` | three notebooks that reproduce every table in the write-up from the CSVs |
| `figures/` | figures for the thesis |
| `archive/` | outside git: logs, notebooks, superseded results |

## Setup

The package in `common/` is installed into every environment, so scripts run from any
working directory:

```bash
pip install -e .
```

Four virtualenvs, because TabDPT, TabFM and EXAONE require mutually incompatible torch
versions: `venv-tfm` (TabPFN, TabICL, XGBoost), `venv-tabfm` (TabFM, EXAONE),
`venv-tabdpt`, `venv-graph`. Install the package into each of them.

`venv-tfm` pins the pair the second part is a statement about:

```bash
python -m venv venv-tfm
venv-tfm/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
venv-tfm/Scripts/python.exe -m pip install "tabpfn==8.4.0" "tabicl==2.1.1" xgboost pandas scipy scikit-learn
venv-tfm/Scripts/python.exe -m pip install -e .
```

The pins are the point: the preprocessing of those exact versions was read line by line,
and pip serves newer ones (8.5.0, 2.2.0) by default.

TabPFN 8.4.0 will not load its weights without a Prior Labs account: the first call opens
a browser login, asks for a licence to be accepted and then for an API key. Only the
account holder can do that, so every driver in `run/` takes a `MODELS=` override, and
because every cell is keyed, TabPFN-V3 can be filled in afterwards without remeasuring
anything else.

Whatever is installed is recorded twice, because the central claim of the second part is a
statement about what specific package versions do to the target variable.
`results/provenance.csv` gets one row per process, written by `common/append.py` on its
first write: package versions, commit, interpreter, virtualenv and every environment knob.
`sh run/freeze_envs.sh` writes `requirements/<env>.txt` for each virtualenv it can find,
which is the set to reinstall from.

Model weights are never stored in the repo; TabFM is fetched by `run/download_tabfm.sh`
and needs about 10 GB free.

## Running

From the repo root, with `-u` so output is not buffered. Settings go through environment
variables, and every one of them is recorded as a column in the output CSV:

```bash
MODELS=EXAONE N_EST=1 SEEDS=20 OUTPUT=shape_of_x_exaone_nest1.csv python -u experiments/h1_shape_vs_scale/shape_of_x.py
```

Long runs are kept as scripts under `run/`:

```bash
sh run/run_tabpfn_reseed.sh
```

Every number quoted in the write-up is produced by the notebooks in `analysis/`, which
read the CSVs in `results/` and print any disagreement with the text.

The estimators have tests, because three numerical bugs in them are on record and all
three were found by reading rather than by running. They need numpy, scipy, pandas and
matplotlib, the last only because one test imports every script in `experiments/` and one
of those draws figures:

```bash
python -m pytest tests -q
```

`TFM_STRICT=1` turns a numeric warning into an exception and makes a changed dataset
fingerprint fail the run instead of printing. Worth one pass after touching an estimator.
