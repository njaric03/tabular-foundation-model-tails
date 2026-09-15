# Three framings of the topic

Three ways to frame one topic. They draw on the same measurements and differ in which
question leads and what is left out.

## F1. Where a tabular foundation model loses the upper tail: prior, target encoding and output head

The predictive distribution of a tabular foundation model passes through three layers
before a user reads a quantile: the prior the network was pre-trained on, the encoding that
brings the context target into the network, and the output head that turns logits into a
distribution. The upper tail can be lost in each of them, for different reasons and under
different triggers. The framing does not claim a causal link between the three; it locates
each loss in one layer.

Consists of:

- **Prior.** On a generator with a known conditional tail index, the models track about 94%
  of the change in scale and 40 to 50% of the change in shape, while an extreme-value oracle
  on the same rows recovers 98%. Tree ensembles show the same deficit (GBM 0.31 to 0.45,
  CatBoost 0.60), so the layer asks what any learner without extreme-value structure
  misses, and whether a prior can supply it.
- **Target encoding.** Every package standardises the target by a non-robust mean and sd,
  while features go through outlier-resistant transforms. One leverage row in the context
  moves Q(0.99) by tens of percent.
- **Output head.** TabPFN's bar distribution has half-normal tails that its quantile path
  never reads, and five bars above 17 sd. TabICL's exponential tail and its disabled GPD
  branch cannot report what the quantile bank does not carry. TabDPT's uniform grid shows
  no model-specific deficit.
- **Separating the layers.** Source reading for the head, and a 2 x 2 pre-training
  experiment (prior x target encoding) for the first two.

## F2. One row is enough: the breakdown point of the predictive tail in tabular foundation models

Tabular foundation models protect their features against outliers and leave their target
unprotected. A single context row that inflates the sd, such as a claim entered in cents
instead of units, changes the predicted upper quantile. The framing measures that
influence, locates its mechanism in the code, measures how often it occurs, and prices the
repairs that work without retraining.

Consists of:

- **Mechanism.** In TabPFN the raw-space grid is `borders * y_train_std_ + y_train_mean_`,
  so the borders over the data fall from 2559 to 217 at an sd shift of 50 and the implied
  tail index turns negative.
- **Natural experiment across six models on real claims data.** TabPFN v2.5 and v2.6 lose
  about half of Q(0.99), V3 8%, TabICLv2 15%, and EXAONE nothing under the same
  standardisation.
- **Prevalence.** An sd shift of 4 or more occurs in 2 of 99 public datasets; a unit error
  in one row is visible at 4 to 33% of rows on seven heavy-tailed tables.
- **Repairs and their price.** A cap derived from the context's own sd shift restores the
  grid at no measurable cost on a clean context. A rank-transformed context is the sharpest
  option at 0.99 and can fail badly at 0.999. A GPD fitted to conformity scores restores
  coverage at 0.99 without knowing which row is wrong, and does not reach 0.999 for the
  foundation models.
- **Coverage as the counterweight.** On clean contexts the foundation models hold their
  nominal level better than the tree controls, and one leverage row costs a fraction of a
  point of coverage at 0.99.

## F3. Prior or architecture? Controlled pre-training of the upper tail in tabular foundation models

The tail failures of production models can come from what the network learned or from how
it is built. Production models cannot separate the two, because prior, encoding and head
change together from one model to the next. The framing pre-trains models of a fixed
architecture under controlled priors and target encodings, and checks whether the effect
survives scale.

Consists of:

- **Motivation.** The leverage failure and the conditional shape deficit measured in
  TabPFN, TabICLv2 and EXAONE.
- **2 x 2 design on nanoTabPFN** (TFM-Playground), with identical base tables and budget.
  Prior: A0 standard, or A1 heavy tails with covariate-dependent shape plus contaminated
  contexts. Target encoding: B0 mean and sd, or B1 median and IQR. At least two
  pre-training seeds per arm.
- **Scale ladder.** The same architecture scaled up with the same head, then continued
  pre-training of TabICLv2 from its released weights on its public v2 prior, each with the
  control arm and the best arm only.
- **Controls.** The control arm gets the same extra training steps; cost is measured on
  standard regression tasks (TabArena) and not only on the generator; results are reported
  per head type; the shape share is measured against the extreme-value oracle rather than a
  tree control.
- **Predictions** written before evaluation in
  `experiments/h3_repair/prior_pretraining_eval.py`: whether the small model reproduces the
  failure, whether each factor removes it, and at what cost.
- **Release.** A tail-robust checkpoint, if an arm removes the failure at no cost.
