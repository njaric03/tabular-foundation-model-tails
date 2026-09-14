# Task: is the upper-tail failure in the prior or in the architecture?

A self-contained prompt for a Claude Code session on a machine with an NVIDIA GPU and CUDA.
Written 14.9.2026 on the CPU machine that owns the rest of the measurements. Read the whole
file before running anything.

---

## 0. First, before anything else

```bash
git pull origin master
git log --oneline -3            # must include c5dadd4 "docs: keep the predictions and outcomes..."
git status                      # must be clean
```

Work on `master`. **Do not merge `claude/code-review-refactor-894248`**; that is being handled
on the other machine. The other machine is still writing `results/h3_repair/unit_error_real.csv`
and `results/provenance.csv`, so:

- write **only new output files** (always pass `OUTPUT=...` with a new name; never reuse an
  existing CSV name);
- before every push run `git pull --rebase origin master`;
- never edit existing files under `results/` or `findings/`.

Read `RULES.md` and `README.md` first. Rules that matter most here: every knob that changes a
result is a column and part of the resume key (rule 2), and statistics are computed at the level
of the free unit, never over nested rows (rule 6).

Commit messages: short, plain English, in the existing style (`feat: ...`, `data: ...`,
`fix: ...`, `docs: ...`). **No `Co-Authored-By` trailer, no em dashes, no ellipsis.**

---

## 1. Background you need (the findings files are not in git)

The thesis measures the upper tail of the predictive distribution of tabular foundation models
(TabPFN v2.5/v2.6/V3, TabICLv2, TabDPT, EXAONE) on heavy-tailed positive targets. Three findings
motivate this task:

1. **Shape vs scale.** Models track the conditional scale of the target (slope about 1) but
   capture only 0.28 to 0.47 of the conditional tail shape, on all four foundation models with a
   quantile path; a GBM control tracks shape (about 1.2). An EVT oracle on the same sample recovers
   0.96 to 0.99 of the shape gradient, so the signal is learnable.
2. **One leverage row breaks the tail.** All packages standardise the context target by plain mean
   and sd. One extreme row inflates the sd; TabPFN's output head is a fixed grid of bar borders in
   standardised space, so the grid stretches, the number of borders over the data falls from about
   2560 to 224 at an sd shift of 50, and the implied tail index turns negative. On real insurance
   data TabPFN v2.5 and v2.6 lose about half of the predicted Q(0.99), V3 8%, TabICLv2 15%.
   EXAONE standardises the same way and loses nothing; TabICLv2 has no grid and still loses 15%.
   So standardisation alone costs part of the tail, the grid multiplies it, and something else
   (plausibly the pre-training prior) lets one model avoid it.
3. **Repairs at inference time exist but each has a price.** Capping the context, rank-transforming
   the target with a GPD tail, and a conformal GPD patch on the output all help at some level and
   fail at another.

The open question this task answers: **is the failure a property of the pre-training prior or of
the architecture (non-robust target standardisation plus a fixed bar grid)?** A small model of the
same architecture, pre-trained under controlled priors, can separate the two. It shows the
**direction** of an effect, not its size in the production models, and must be reported that way.

---

## 2. Tooling: TFM-Playground

Use https://github.com/automl/TFM-Playground (nanoTabPFN itself is classification only).
Already verified from its source:

- `tfmplayground/models/nanotabpfn.py`: TabPFNv2-style model; decoder emits `num_outputs` logits;
  features standardised by plain mean/sd and clipped to +-100.
- `pretrain_regression.py`: criterion `pfns.bar_distribution.FullSupportBarDistribution` over
  `--n_buckets` (default 100) bucket edges from `utils.make_global_bucket_edges`, which standardises
  each table's y by mean and sd (ddof=1) before computing the edges. Model defaults: 6 heads,
  embedding 192, hidden 768, 6 layers, lr 1e-4.
- `tfmplayground/train.py` lines 78 to 86: during pre-training y is standardised per table with the
  mean and sd of the **train part only**, and test targets with the same statistics.
- `tfmplayground/interface.py` lines 201 to 222: `NanoTabPFNRegressor.fit` standardises y by mean
  and sd; `predict` returns only `dist.mean(logits)` un-standardised. **No quantile output**; you
  must compute quantiles yourself from the same bar distribution (verify the `icdf` method name in
  the installed `pfns` version, otherwise invert the discrete CDF of the bucket probabilities).
- `tfmplayground/external_priors/base.py`: `PriorDumpDataLoader` reads HDF5 dumps with keys `X`,
  `y`; batches carry `x`, `y`, `target_y`, `train_test_split_index`. `PriorDataLoader` accepts a
  custom `get_batch` function.
- `python -m tfmplayground.external_priors --lib tabicl --prior_type mix_scm ...` generates dumps;
  flags include `--num_batches`, `--batch_size`, `--min_features`, `--max_features`,
  `--min_seq_len`, `--max_seq_len`, `--min_eval_pos`, `--np_seed`, `--torch_seed`, `--save_path`.

**Do not pre-train on the released dump `50x3_1280k_regression.h5` for the experiment.** It holds
tables of 50 rows and 3 features; our contexts are hundreds to thousands of rows. It may be
downloaded only for a smoke test of the pipeline (step 4).

---

## 3. Environment

Use a new virtualenv `venv-tfmp` (ignored by git through `venv-*/`). Keep the repo's other
environments untouched. On Windows the interpreter is `venv-tfmp/Scripts/python.exe`, on Linux
`venv-tfmp/bin/python`; below it is `$PY`.

```bash
python -m venv venv-tfmp
# TFM-Playground pins torch==2.9. Install the CUDA build matching this machine's driver first,
# e.g. (pick the right cuXXX index for the installed CUDA):
$PY -m pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/automl/TFM-Playground .cache/tfmp/TFM-Playground
git -C .cache/tfmp/TFM-Playground rev-parse HEAD      # record this sha
$PY -m pip install "pfns==0.3.0" h5py schedulefree "openml==0.15.1" "tabicl[pretrain]>=2.1.0" \
    scikit-learn requests pandas scipy pyreadr
$PY -m pip install -e .cache/tfmp/TFM-Playground --no-deps
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must be 2.9.x, True
```

Record in a new file `data/tfmp_artifacts.json` (commit it): the TFM-Playground commit sha, the
torch and pfns versions, the GPU model, and the sha256 of every file you download or generate
(dumps, checkpoints). Store every dump and checkpoint under `.cache/tfmp/` (ignored by git);
`.h5` and `.pth` are not otherwise ignored, so never put them elsewhere.

The real tables outside OpenML must be converted once, with checksums already recorded in
`data/external_sources.json`:

```bash
$PY -m common.external          # downloads the raw files and refuses if a checksum differs
```

---

## 4. Smoke test (small downloads only)

Download the released regression checkpoint (`nanotabpfn_regressor.pth`, 6.9 MB) and its bucket
edges (`nanotabpfn_regressor_buckets.pth`, 1.5 KB) from
`https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/`. Write the
quantile adapter (step 6) and confirm on a toy regression that the median and the quantiles are
monotone and sensible. This only tests the code path; the released model is trained on 50-row
tables and is not an experimental arm.

---

## 5. The design: prior x target encoding, 2 x 2, at least 2 pre-training seeds per arm

All four arms use the **same model size, the same number of optimiser steps, the same batch size,
the same generated base tables** (same dump, same `--np_seed/--torch_seed`), and differ only in
the factor under test. Measure throughput with a short pilot first, then fix one budget for all
arms (for example 2 to 4 GPU hours each) and record it.

**Base dump.** Generate once with the TabICL `mix_scm` prior at `--max_seq_len 1024`,
`--max_features 32`, `--min_eval_pos` so that at least 512 rows are context, enough tables for the
budget. Record every flag.

**Factor A: prior.** Implemented as a wrapper around the batches of the base dump, so A0 and A1 see
identical base tables.

- **A0 standard**: the base dump as generated.
- **A1 heavy tails plus contaminated contexts**, applied per table with recorded probabilities:
  - *heavy tail with covariate-dependent shape* on a fraction `p_tail` (suggest 0.5) of tables:
    map each table's y through its within-table ranks to a generalised Pareto quantile function
    whose shape xi depends on one randomly chosen feature, xi(x) in [0, 1] (for example
    xi = sigmoid(w * x_j) with random w). Apply it to context and test targets alike: it is the
    data-generating distribution.
  - *leverage contamination* on a fraction `p_contam` (suggest 0.3) of tables: pick one row of the
    **context part only** and scale its target so that the context sd shift
    `sd(all) / sd(without that row)` lands in [4, 50]. **Never contaminate test targets**: the loss
    must teach the clean conditional distribution.

**Factor B: target encoding**, changed consistently in all three places (bucket edges, pre-training
loss in `train.py`, inference in the adapter):

- **B0 mean/sd**: as shipped.
- **B1 robust**: median and IQR / 1.349 (+ epsilon) instead of mean and sd, computed on the context
  part. Everything else unchanged.

Name models `nanoTabPFN-A0B0-s1`, `nanoTabPFN-A1B0-s1`, and so on. Keep each arm's training config
(all flags, `p_tail`, `p_contam`, xi range, budget, seed) in `data/tfmp_artifacts.json`.

---

## 6. Adapter into this repo

Add a `nanoTabPFN-*` branch to `common/models.py:quantiles` that loads the checkpoint for the given
name from `.cache/tfmp/checkpoints/`, applies the arm's encoding (B0 or B1) to the context target,
computes quantiles from the bar distribution, un-standardises, and chunks test rows so memory stays
bounded. It must refuse contexts longer than the pre-training `max_seq_len` and feature counts
above `max_features` rather than silently truncating. Add any new environment knob to
`common/provenance.py:ENV_KNOBS`. Add a test in `tests/` that the adapter returns monotone
quantiles.

---

## 7. Predictions: write them before any evaluation

Create `experiments/h3_repair/prior_pretraining_eval.py` with a module docstring in the style of
`experiments/h3_repair/clip_context.py` (WHY, WHAT IS MEASURED, PREDICTIONS WRITTEN BEFORE THE RUN)
and commit it **before** running the evaluation. Proposed predictions, adjust wording but keep them
falsifiable:

- **Q1, the gate.** Under A0B0 a single context row with an sd shift of 20 or more lowers the
  predicted Q(0.99) and pushes the implied tail index towards or below zero, as in TabPFN. **If A0B0
  does not reproduce the failure, stop: the small model is not a valid proxy, report that as the
  result, and do not interpret the other arms.**
- **Q2.** A1B0 reduces that leverage loss substantially against A0B0, while pinball at 0.99 on
  clean contexts gets worse by at most a few percent.
- **Q3.** A1B0 captures a larger share of the conditional tail shape than A0B0 on the generator.
- **Q4.** A0B1 removes the resolution loss under leverage (borders over the data stay flat across
  sd shifts), and A1B1 gives the best pinball at 0.99 under leverage.
- **Q5.** No arm closes the shape gap to the GBM control.

---

## 8. Evaluation, all with new OUTPUT names

Contexts must fit the pre-training size: use 768 context rows unless the pilot says otherwise.

1. **Generator, leverage and resolution** (new script `prior_pretraining_eval.py`): reuse
   `common/generator.py` and `common/metrics.py` (`y0_for_sd_shift`, `sd_shift`,
   `xi_from_residual_ratio`). xi in {0.7, 0.9}, sd shifts {1, 4, 20, 50}, at least 10 evaluation
   seeds per pre-training seed. Report implied xi, pinball at 0.99 and 0.999 on clean test rows, and
   borders over the data (the bucket edges mapped back through the arm's encoding).
2. **Part one, shape vs scale**: run `experiments/h1_shape_vs_scale/shape_of_x.py` and
   `scale_of_x.py` with `MODELS=nanoTabPFN-...` and a 768-row training size (check their knobs;
   add one if needed and make it a column).
3. **Real data, natural leverage**: `experiments/h3_repair/clip_context_real.py` with
   `MODELS=nanoTabPFN-...`, `N_FIT=768`, `OUTPUT=clip_context_real_nanotabpfn.csv`.
4. **Real data, unit error**: `experiments/h3_repair/unit_error_real.py` with
   `MODELS=nanoTabPFN-...`, `N_FIT=768`,
   `DATASETS=freMTPL2sev,beMTPL97,ausprivauto0405,norauto,AutoClaims,MEPS_2016_TOTEXP`
   (BlogFeedback has 280 features and exceeds `max_features`), and
   `OUTPUT=unit_error_real_nanotabpfn.csv`.

Statistics: the free unit is the pre-training seed crossed with the evaluation dataset or
generator seed. Compare arms only on cells where all arms exist; pooled medians over different cell
sets have misled this project five times.

---

## 9. What to deliver

- Commit the new script, the adapter, the tests, `data/tfmp_artifacts.json` and the new result CSVs.
- Append an `OUTCOME` block to the docstring of `prior_pretraining_eval.py` marking each prediction
  `HELD`, `FAILED`, `WRONG` or `HALF`, with the numbers, and copy that block into a new section at
  the end of `PREREGISTRATION.md`.
- Keep a short summary table: rows are the four arms, columns are the loss in Q(0.99) at an sd shift
  of 20 or more, implied xi at a shift of 50, clean pinball at 0.99, and captured shape share.
- `git pull --rebase origin master`, then push. Stop and report if the rebase touches any file you
  did not create.

If anything in this file turns out to be wrong in the installed code (a flag name, a method name, a
file path), do not work around it silently: note the discrepancy in the final report.
