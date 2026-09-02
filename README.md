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

Detailed findings are written in Serbian, the language of the thesis, under `findings/`.
`FINDINGS.md` is the English summary with every headline number.

## What the three parts claim

**1. The conditional scale is tracked, the conditional tail shape is not.**
On a generator where the true tail index varies with a covariate, the models capture
about 94% of the change in scale and only about 40 to 50% of the change in shape, with
a bootstrap spread of 0.15 to 0.20 across seeds. This is not a limit of the sample: an
extreme-value oracle on the same 2000 rows recovers 98%. The dissociation survives four
distribution families (60 of 60 runs), a location-invariant estimator, a reseeding
control, and it holds by rank on nine real datasets (9 of 9 datasets, p = 0.002 by a
sign test at the level of the dataset, which is the free unit; the row-level
p = 3.4e-11 counts 3 seeds and 3 models per dataset as independent and should not be
quoted). What does not travel to real data is the number 43%: the feature carrying the
gradient there is chosen as a maximum over columns, and on 9 of 10 datasets that
gradient does not clear its own selection noise.

**2. The predictive distribution is not robust to a single leverage row in the context.**
Read from the source of five packages: none of them protects the target variable, which
is standardised by its plain mean and standard deviation, while features are as a rule
put through outlier-resistant transforms. One added row can move Q(0.99) by tens of
percent, and the predicted mean of some models by an order of magnitude including a sign
change. Tree ensembles, which do not standardise the target, do not react at any dose.
The trigger is a data error rather than a heavy tail as such: a standard-deviation shift
of 4x or more occurs in 2% of 99 public datasets.

**3. After pre-training the weakness can be reduced, not removed, and the cost is
measurable.** Winsorizing the target helps at alpha = 0.99 under contamination, by up to
23.5%, but consistently hurts TabICL at alpha = 0.999. Replacing the scale estimator with
a robust one makes everything worse by an order of magnitude, because the output support
is calibrated on exactly that non-robust scale: the vulnerability and the safety mechanism
are the same thing. The one free recommendation is `n_estimators=1`, which buys about 9%
of tail accuracy at no cost to the median.

Not yet measured, and the most direct question the framing raises: what fraction of
held-out outcomes exceeds the predicted Q(0.99). Everything above is indirect.

The first two weaknesses share a birthplace, the layer that processes the target, but not
a trigger. They are not causally linked, and the thesis does not claim they are.

## Models

| model | quantiles | why it is here |
|---|---|---|
| TabPFN-V3, v2.6, v2.5 | yes | top of TabArena; bar-distribution head |
| TabICLv2 | yes | 999-quantile head, different output construction |
| EXAONE-Tabular | yes | second on TabArena; shares TabICL's head |
| TabDPT | via its bin head | 2048 fixed bins, recovered in `common/adapters/tabdpt.py` |
| TabFM | no, mean only | first on TabArena in regression, but publishes no distribution |
| XGBoost, sklearn GBM | yes | controls that do not standardise the target |
| CatBoost | yes | best tree model in regression on TabArena; wired up, not yet measured |

Left out on purpose: LimiX and Mitra return a point prediction only, so there is nothing
distributional to measure; RealTabPFN-2.5 is not public. The boundary of this work is
which models expose a predictive distribution at all.

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
| `findings/` | one document per phenomenon, in Serbian, plus `NALAZI.md` as the overview |
| `analysis/` | three notebooks that reproduce every table in the findings from the CSVs |
| `figures/` | figures for the thesis |
| `thesis/` | the topic application, the literature search, working notes |
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

The pins are the point: `findings/NALAZI.md` section 2.1 reads the preprocessing of those
exact versions line by line, and pip serves newer ones (8.5.0, 2.2.0) by default.

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

Every number in `FINDINGS.md` and in `findings/` is produced by the notebooks in
`analysis/`, which read the CSVs in `results/` and print any disagreement with the text.

The estimators have tests, because three numerical bugs in them are on record and all
three were found by reading rather than by running. They need numpy, scipy, pandas and
matplotlib, the last only because one test imports every script in `experiments/` and one
of those draws figures:

```bash
python -m pytest tests -q
```

`TFM_STRICT=1` turns a numeric warning into an exception and makes a changed dataset
fingerprint fail the run instead of printing. Worth one pass after touching an estimator.

## Six rules

Each one has been broken once already and cost measurements.

1. One process writes one output file. Every parallel stream gets its own `OUTPUT=`.
2. Every adjustable parameter is a column in the CSV and part of the resume key. `N_EST`
   moves the captured shape share for TabPFN from 19% to 42%; when it is not in the key,
   a re-run with a different value silently skips the work. `common/append.py` now
   refuses to resume when a key column is missing rather than skipping quietly.
3. A result enters the repo only when a finding cites it or a script reads it. Everything
   else stays in `archive/`.
4. The same seed for every model in the same comparison, and `random_state=seed` for all
   of them. This lives in one place, `common/models.py`.
5. Warnings are filtered by category, never blanket. `filterwarnings("ignore")` at the
   top of every script hid a divide-by-zero in the tail-index inversion for months.
   `common/quiet.py` silences package noise and leaves `RuntimeWarning` visible.
6. A test may not be counted more often than its free unit varies. Three seeds on one
   dataset are not three independent comparisons; `common/stats.py` reports the
   cluster-level p-value next to the row-level one, and the text quotes the first.
