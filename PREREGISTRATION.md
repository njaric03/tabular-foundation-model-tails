# Predictions written before each run and what happened

Extracted verbatim from the module docstrings of the scripts below at commit `578a2f3`,
before a refactor shortened those docstrings. Nothing here was edited: the text is what
sat at the top of each script, predictions first and outcomes appended after each run.
The git history of each script shows when a prediction block was committed relative to
the result rows it describes.

These are predictions written into the code before running, not a formal
preregistration in a public registry. Where a prediction was drafted after a smoke test,
or a threshold was picked after a run, the block says so itself.

Outcomes of runs after the refactor is merged are appended below the last section, in
the same form.

The first four sections were written in Serbian and are kept as written. References to
`findings/` and `FINDINGS.md` point to working notes that are not published.

## Contents

Sections are in the order they were written, grouped here by part of the thesis.

- Scale and tail shape: `xi_residual.py`.
- Extreme context target: `prevalence_models.py`, `grid_occupancy.py`,
  `head_tail_tabicl.py`.
- Capping: `clip_context.py`, `clip_context_real.py`, `unit_error_real.py`,
  `pinball_reference.py`; the confirmation of 1.10.2026, `unit_error_confirm.py` and
  `legit_segment.py` with their outcomes; `recalibration.py`, 2.10.2026;
  `unit_error_confirm.py` with native categories, 13.10.2026; both scripts with
  TabPFN-V3's default ensemble and a log arm, 20.10.2026, with the outcome of 21.10.2026;
  both on TabPFN-3.5, 21.10.2026, with its outcome; `clip_context.py` at the default
  ensemble for V3 and 3.5, 22.10.2026, with its outcome; the correction to the expected
  pinball of the generator, 22.10.2026.
- Scale and tail shape, added: `dissociation_real_slope.py` across the Hill share,
  22.10.2026, with its outcome.
- Robustness of the leverage result, 27.10.2026: the position of the row, the context
  size, the default ensemble with two more tables and the repairs' loss at 10 000 rows,
  each with its outcome; TabICLv2 at its default ensemble, 28.10.2026, with its
  outcome; TabPFN-V3 without its target transform, 28.10.2026, with its outcome.
- Corrections of 1.10.2026: the sd shift quoted for freMTPL2sev, the fine-tuning protocol
  and MEPS_2016_TOTEXP. Thread count, 15.9.2026.
- Outside the thesis: `architecture_tabdpt.py`, `architecture_tabfm.py`,
  `tail_splice.py`; pre-training of nanoTabPFN (`prior_pretraining_eval.py`,
  `prior_pretraining_real.py`) and fine-tuning of TabICLv2 (`tabicl_prior_finetune.py`),
  15 to 17.9.2026.

---

## `experiments/h1_shape_vs_scale/xi_residual.py`

```text
Slabost 2 iz TEMA-POTVRDA-v2: merni instrument potcenjuje uz lokacijsku komponentu.

PROBLEM
-------
Ceo prvi deo rada meri implicirano xi iz odnosa Q(0.99)/Q(0.9). Taj odnos
pretpostavlja SKALNU familiju. Cim postoji lokacija mu, odnos se skuplja ka 1 i
implicirano xi ispadne manje: nominalno 0.7 vratilo je referencu 0.43
(findings/NALAZI.md 3.5). To je centralna alatka poglavlja 1 i nije provucena kroz nalaze.

ISPRAVKA
--------
Umesto odnosa nivoa, uzeti odnos RAZLIKA u odnosu na medijanu:

    r_res = (Q(0.99) - Q(0.5)) / (Q(0.9) - Q(0.5))

Dodavanje bilo kakvog mu se skrati i u brojiocu i u imeniocu, pa je mera
invarijantna na lokaciju po konstrukciji. xi se resava iz

    (core_q(0.99, xi) - core_q(0.5, xi)) / (core_q(0.9, xi) - core_q(0.5, xi)) = r_res

STA SE MERI
-----------
DEO 1 (bez modela, sekunde): na uzorcima iz GPD-a sa poznatim xi i dodatom
lokacijom mu, koliko svaki estimator promasi. Ocekivanje: sirovi pada sa mu,
rezidualni ne.

DEO 2 (sa modelima): ponoviti merenje udela pracenog gradijenta xi(x) -- glavni
broj rada, 42-45% -- sa OBA estimatora. Pitanje nije da li se broj menja nego
da li se menja ZAKLJUCAK (skala >> oblik).

PREDVIDJANJE, ZAPISANO PRE MERENJA
----------------------------------
  Q1. Rezidualni estimator vraca xi bez pristrasnosti po mu; sirovi pada
      monotono sa mu.
  Q2. Udeo pracenog gradijenta ce se promeniti za manje od 10 procentnih poena
      i zakljucak (skala ~94%, oblik znatno manje) ostaje.
  Ako Q2 padne -- dakle ako se pod ispravnim estimatorom oblik prati jednako
  dobro kao skala -- glavni rezultat rada pada i to se tako pise.

POKRETANJE
----------
    python -u xi_residual.py
    PARTS=1 python -u xi_residual.py
```

---

## `experiments/h2_leverage/architecture_tabdpt.py`

```text
Predvidja li ARHITEKTURA glave ponasanje u repu?

PREDVIDJANJE IZ KODA, PRE MERENJA
---------------------------------
Iz izvornog koda tri modela:

  TabPFN   `regressor.py`: raw_space_bardist_ = borders * y_train_std_ + y_train_mean_
           -> fiksna mreza korpi u z-prostoru, skalirana empirijskom sd
  TabDPT   `estimator.py`: edges = linspace(bin_min=-10, bin_max=+10, 2048)
           uz train_y = normalize_data(train_y)
           -> ISTO, i jos sa TVRDOM granicom nosaca na mean +- 10*std
  TabICL   999 kvantila, bez mreze korpi
           -> nema sta da izgubi rezoluciju skokovito

Odatle predvidjanje koje se moze oboriti:

  (1) TabPFN i TabDPT pokazuju ODSECEN NOSAC -- sredina im je bliza odsecenoj nego pravoj;
  (2) TabPFN i TabDPT gube rezoluciju kad se ubaci outlier, TabICL ne;
  (3) TabDPT ima TVRDU granicu: ne moze da predvidi iznad mean + 10*std, bez obzira na
      podatke. To je proverljivo analiticki.

Ako TabDPT prati TabPFN a ne TabICL, arhitektura predvidja ponasanje.

MERI SE (samo preko javnog API-ja, dakle sredina)
-------------------------------------------------
  odnos_prava    sredina modela / prava uslovna sredina
  odnos_odsec    sredina modela / E[Y | Y <= max(y_train)]
  uticaj         promena sredine kad se ubaci jedna tacka sa y = 100*max
  granica        koliki deo prave sredine uopste staje u nosac mean +- 10*std

POKRETANJE (TabDPT ima svoj venv)
---------------------------------
    venv-tabdpt/Scripts/python.exe architecture_tabdpt.py
```

---

## `experiments/h2_leverage/architecture_tabfm.py`

```text
TabFM kao peti model: odsecen nosac i funkcija uticaja.

ZASTO BAS OVA DVA TESTA
-----------------------
`TabFMRegressor.predict(X)` vraca samo sredinu, pa se implicirano xi i disocijacija
nivo/oblik ne mogu meriti. Ali dva testa traze samo sredinu:

  ODSECEN NOSAC   model / prava sredina  naspram  model / E[Y | Y <= max(y_train)]
  UTICAJ          promena sredine kad se ubaci jedna tacka sa y = 100*max

PREDVIDJANJE, ZAPISANO PRE MERENJA
----------------------------------
`TabFMRegressor.__init__` ima `outlier_threshold: float = 4.0` -- dakle EKSPLICITNO
odsecanje outliera u pretprocesiranju, cega nema ni kod TabPFN-a ni kod TabDPT-a.

Odatle:
  (1) TabFM treba da bude ROBUSTAN na ubacenu tacku -- za razliku od TabDPT-a kome
      sredina menja znak i TabPFN-a koji je bimodalan;
  (2) ali bas zbog odsecanja treba da PODBACI na pravoj sredini pod teskim repom,
      jer odseca upravo one vrednosti koje sredinu i cine.

Ako oba prodju, taksonomija dobija cetvrtu kategoriju: eksplicitno klipovanje kupuje
robusnost po ceni nivoa.

POKRETANJE
----------
    venv-tabfm/Scripts/python.exe architecture_tabfm.py
```

---

## `experiments/h2_leverage/prevalence_models.py`

```text
FAZA 2 studije prevalencije: uslovljeno na polugu, pomera li se stvarno Q(0.99)?

KONTEKST
--------
Faza 1 (`prevalence_data.py`, 99 skupova, 19.800 merenja) je pokazala da
pomeraj sd >= 4x postoji u 2% tabela, a najgori je freMTPL2sev sa max 19,9x i
11% poduzoraka preko 4x.

`findings/NALAZI.md` 4.2 je ranije zakljucio "na uzorcima sa polugom >= 3x nema efekta",
ali je tada uslovljavano na POGRESNU velicinu -- odnos max/drugi umesto pomeraja
sd. Ovde se to ispravlja.

DIZAJN
------
Samo skupovi koje je faza 1 oznacila (pomeraj sd >= 2x bar jednom). Za svaki:
  * uzorkuj poduzorke od N_FIT redova i izracunaj pomeraj sd
  * zadrzi po KVOTA poduzoraka u svakoj korpi pomeraja: <1.2, 1.2-2, 2-4, >=4
    (odbacivacko uzorkovanje -- inace bi visoke korpe bile prazne)
  * u svakom: fituj model SA najvecom tackom i BEZ nje, izmeri promenu Q(0.99)

To je uslovna verzija 4.2. Pitanje nije "da li stvarni podaci lome modele"
(odgovor je vec ne, u proseku) nego "lome li ih ONI poduzorci koji nose polugu".

PREDVIDJANJE, ZAPISANO PRE MERENJA (isto kao u findings/h2/prevalence_data.md)
--------------------------------------------------------------------
  R1. U korpi pomeraj sd >= 4x ocekujem pomeraj Q(0.99) reda 10-30% kod
      TabICL-a i bimodalan odgovor kod TabPFN-a.
  R2. U korpi < 1.2x ocekujem ispod 5%.
  R3. GBM i XGBoost ravni u svim korpama.
  Ako se korpe ne razdvoje, mehanizam ne stoji na stvarnim podacima i tako se pise.

POKRETANJE
----------
    python -u prevalence_models.py
    DATASETS=freMTPL2sev MODELS=GBM N_PER_BIN=5 python -u prevalence_models.py
```

---

## `experiments/h2_leverage/grid_occupancy.py`

```text
Where the predictive mass sits on the bar grid, under a dosed leverage row.

WHY THIS EXISTS

`findings/h1/head_tail_family.md` established, from the source of `tabpfn` 8.4.0, three
properties of the output head: its outer tail is half-normal and its scale is not learned;
`icdf`, which `regressor.py` calls for quantiles, is not overridden and therefore never
reads that tail; and the border grid holds 341 borders per unit in the body but three
between 50 and 128 standard deviations, so above about 17 the whole tail is five bars of
22 sd each. Placing mass in the last bar by hand then moved the reported Q(0.999) from
3.29 to 105.78 on a half-per-mille change.

All of that is a statement about what the head *can* do. It says nothing about whether a
fitted TabPFN on real data ever reaches that regime, and that is the difference between an
architectural curiosity and an explanation of the measurements this repository already has.
This script asks the empirical half, on the treatment of `influence.py` in `sd` mode.

WHAT IT IS TRYING TO EXPLAIN

`findings/NALAZI.md` section 2.6: dosed on the sd shift itself, at a true xi = 0.7, TabPFN's
implied xi goes 0.576 / 0.515 / 0.356 / -0.173 / -0.189 across shifts 2 / 4 / 10 / 20 / 50.
GBM is flat to the third decimal at every dose. At a shift of 20 -- which `freMTPL2sev`, the
worst of 99 public datasets, actually carries at 19.9 -- TabPFN's predictive distribution
stops being heavy-tailed and reports as bounded. The finding is currently recorded as a
property of the model, with no mechanism attached to it.

The candidate mechanism is arithmetic. The raw-space grid is
`borders * y_train_std_ + y_train_mean_`, so a row that inflates the standard deviation by
20x stretches the whole grid by 20x while the data stays where it was. The data then
occupies a small number of bars, and the inherited `icdf` interpolates *linearly* inside a
bar. Linear interpolation inside one bar is locally uniform, and a uniform tail is exactly
what a negative xi reports. If Q(0.9) and Q(0.99) end up inside the same bar, the ratio
between them carries no model belief at all -- only the geometry of that bar.

PREDICTIONS, WRITTEN BEFORE THE RUN

P1. `borders_in_data`, the number of borders falling inside the training range, collapses
    roughly in proportion to the sd shift. This one is nearly arithmetic and is here as a
    sanity check on the instrument rather than as a finding.

P2. `same_bar_90_99`, the share of test rows whose Q(0.9) and Q(0.99) land in one bar,
    is near zero in the clean context and rises sharply by shift 20. If it does, section
    2.6's negative xi is bar geometry rather than a belief the model holds, and the two
    parts of the thesis are joined by a measurement instead of by an expression.

P3. `mass_beyond_inner`, the probability mass at or past `borders[-2]`, stays negligible
    at every dose. The half-normal branch would then be latent: real, unreachable here, and
    honestly reported as an architectural property that this treatment does not trigger.
    P2 and P3 are independent; P2 can hold while P3 fails, and the write-up must say which.

If P2 fails, the head finding stays architectural and section 2.6 keeps looking for a
mechanism. That outcome is worth the hour too, and is the reason the predictions are here
rather than written afterwards.

RUNNING

    MODELS=TabPFN-V3 XI=0.7 SD_SHIFTS=1,2,4,10,20,50 SEEDS=5         python -u experiments/h2_leverage/grid_occupancy.py

Only models with a bar grid can be measured: TabPFN exposes it through
`predict(output_type="full")`, which returns the raw-space criterion and the logits.
TabICL and EXAONE have a 999-quantile head and no grid, so they are refused by name rather
than silently producing empty columns.
```

---

## `experiments/side/head_tail_tabicl.py`

```text
What TabICL's head can express against what it is configured to express.

WHY THIS EXISTS

`findings/h1/head_tail_family.md` did this for TabPFN, whose bar head cannot represent
xi > 0 at all: the outer tail is half-normal, and the quantile path does not even read
that. TabICL's head is a different construction -- 999 quantile levels, a monotone spline
between them, and a parametric tail past the outermost level -- so the same question has
to be asked separately, and the answer is not the same.

WHAT THE SOURCE SAYS, `tabicl` 2.1.1

`_model/quantile_dist.py` implements **both** tail families:

    def estimate_exp_tail_params(...)     # Q(a) = -beta * ln(1-a) + c,  xi = 0
    def estimate_gpd_tail_params(...)     # generalised Pareto, with a shape parameter

and the configuration carries the shape parameter explicitly:

    MIN_ETA: float = -0.49    # eta > 0 gives heavy tails, eta = 0 exponential
    MAX_ETA: float =  0.49    # must be < 0.5 for finite variance

That `eta` is the extreme-value index this thesis calls xi: same parameter, and the
comment about finite variance is the GPD moment condition k < 1/xi.

The switch is `QuantileDistribution(..., tail_type: Literal["exp", "gpd"] = "exp")`.
`_model/tabicl.py:200` builds the head as `QuantileToDistribution(num_quantiles=...)`,
so it takes that default, and `TabICLRegressor.__init__` carries no parameter with
"tail" or "quantile" in its name. The branch cannot be reached through the public API.

WHAT IS MEASURED

The model is fitted once on a context with a known xi. Its native bank of 999 quantiles
is read out, and the *same* vector is then wrapped in `QuantileDistribution` twice, once
per tail type. Nothing about the network changes between the two: the weights, the
context and the predicted quantiles are identical, and only the extrapolation past level
0.999 differs. Whatever separates the two columns is therefore the tail family alone.

Implied xi is read the way the rest of the thesis reads it, by inverting the ratio of
quantiles, at levels far enough out that the tail rather than the spline decides.

PREDICTIONS, WRITTEN BEFORE THE RUN

Q1. `exp` returns an implied xi near zero at every true xi, including 0.8. If it does,
    the ceiling is a configuration default and not a property of the trained network.
Q2. `gpd` returns an implied xi that rises with the true one. If it does, the capability
    is present in the shipped package and switched off, which is a far cheaper
    recommendation than "the head needs a tail parameter".
Q3. `gpd` saturates near 0.49, its configured maximum, so it cannot follow the heaviest
    tails either. The recommendation is then "expose it", not "it solves the problem".

If Q2 fails -- if the GPD branch does not recover more than the exponential one -- then
the branch is dead code and no issue should be filed upstream. That is the outcome this
script exists to rule out before anything is reported.

    XI=0.2,0.5,0.8 SEEDS=3 python -u experiments/side/head_tail_tabicl.py
```

---

## `experiments/h3_repair/clip_context.py`

```text
Clip the context before TabPFN sees it, and test whether resolution comes back.

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

`q999_in_outer_five` in that same file is 0 at shifts of 20 and 50 and at most
0.0011 at 10 (the median is 0 at every shift above 4; remeasured 13.9.2026 with
the border counts reproducing exactly), so the half-normal tail and the `icdf`
mismatch are real but practically never reached.
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
```

---

## `experiments/h3_repair/clip_context_real.py`

```text
Clip the context on real data, on the subsamples that carry the leverage.

WHY
---
`clip_context.py` found a fix on the generator: cap the context at
`median + C * robust_sd` before the fit, and one leverage row no longer stretches
TabPFN's grid off the data. There the true tail index is known, so the question
was whether the reported xi comes back. It does. But that is TabPFN-V3 alone, a
row injected at the centre of x, and a C picked from three values afterwards.

The real-data evidence for part two is one dataset and one model: on
freMTPL2sev TabICLv2 loses about 15% of Q(0.99) on subsamples whose largest
claim shifts the sd by two or more. This asks whether the same clip removes
that loss where it was measured, without a synthetic row and without a known
truth.

WHAT IS MEASURED
----------------
The subsamples are drawn exactly as `prevalence_models.py` draws them: rejection
sampling into four bins of the natural sd shift, seed 31337, N_FIT + N_TEST rows,
five per bin. The raw arm therefore has to reproduce the wide sweep's `d_q99`
on the same cells, and that is the control that the instrument is the same one.

For each subsample, three contexts, one fit each:

    without_max   the largest row removed; the reference the wide sweep uses
    raw           the full context, as shipped
    clip_C        the full context with y capped at median + C * robust_sd

`d_q*` is the median relative change of the predicted quantile against
`without_max`, exactly as in `prevalence_models.py`. If clipping is a fix, a
context that still holds the leverage row behaves like one that never had it:
`d_q99` of clip_C near zero where raw is far from it.

There is no true tail here, so the test set is the arbiter of cost: pinball at
each level on N_TEST held-out rows the context never saw. At 0.99 that is about
ten exceedances per cell; 0.999 is recorded with its count and is not a finding
at this size.

`robust_sd` and the cap are imported from `clip_context.py`, so C means the same
thing in both.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  P1. The raw arm reproduces `prevalence_models_wide.csv` for the same model,
      bin and repeat on freMTPL2sev. Same seed, same sizes, same code path.
  P2. In the bins at two sd and above, TabICLv2's raw `d_q99` is negative and
      clip_200's is near zero.
  P3. Below 1.2 sd the cap rarely binds at C = 200, so clipping costs close to
      nothing in pinball at 0.99. At C = 50 it binds more and may cost.
  P4. GBM moves little under any arm; trees do not standardise the target.

Results in `clip_context_real.csv`.

OUTCOME, 11.9.2026
------------------
  P1. Holds for GBM exactly and for TabICLv2 to 1e-5; `sd_shift` agrees on all
      60 cells. Fails for TabPFN-V3 by up to 0.03 per cell, 0.009 on a bin
      median. The sweep ran before venv-tfm was rebuilt (Python 3.11.0 then,
      3.12.8 now, the same package versions). Within one environment TabPFN is
      deterministic: 6 and 12 threads give identical output.
  P2. Holds. TabICLv2 at 4 sd and above: raw -15.4%, clip_200 +3.6%. At 2 to
      4 sd: -5.2% and +5.8%. TabPFN-V3 at 4 sd and above: -7.8% and +0.4%.
  P3. Holds. Below 2 sd the cap binds on 3 to 5 of 2000 values and pinball at
      0.99 moves by -0.3% to -1.2%, significant for no model. At 2 sd and above
      clip_200 beats raw in 9/10, 9/10 and 10/10 subsamples (TabICLv2,
      TabPFN-V3, GBM; sign test p = 0.021, 0.021, 0.002; medians -6.1%, -2.4%,
      -9.7%). C = 50 costs TabICL 25 to 34% of Q(0.99) in every bin.
  P4. Wrong. GBM's raw d_q99 has a median of +13% at 2 to 4 sd, and GBM gains
      the most from the clip in pinball. Not all of the gain is the grid.

SECOND RUN, 11.9.2026: THE ARMS THAT DO NOT NEED A PICKED C
-----------------------------------------------------------
log, tail_log (C = 50) and evt_trim from `clip_context.transform`, on the same
60 cells. No separate predictions were written here; P5 to P8 of
`clip_context.py` are the ones that apply. Pinball at 0.99 against raw, over
the ten subsamples at 2 sd and above:

    clip_200   TabICLv2 9/10 (p = 0.021)  TabPFN-V3 9/10 (0.021)  GBM 10/10 (0.002)
    log                 8/10 (0.109)               9/10 (0.021)        8/10 (0.109)
    tail_log            9/10 (0.021)               6/10 (0.754)        9/10 (0.021)
    evt_trim   touches no context in any of the 60 cells; identical to raw

log moves TabPFN-V3's Q(0.99) by +20 to +37% against without_max in every
bin, the clean ones included: it gives a different model, not the reference
back. For TabICLv2 at 4 sd and above it makes the loss larger (-19.7% against
raw's -15.4%). tail_log costs TabPFN-V3 a pinball of 854 against 660 at 2 to 4
sd. clip_200 stays the only arm that helps all three models where the leverage
is and moves nothing where it is not.

THIRD RUN, 12.9.2026: SIX MODELS, AND THE MECHANISM AS A NATURAL EXPERIMENT
--------------------------------------------------------------------------
TabPFN-v2.5, TabPFN-v2.6 and EXAONE-Tabular were added on the same 20 subsamples.
EXAONE runs in `venv-tabfm`, which is where its package lives; the run in
`venv-tabfm` is the one `run/run_exaone_a.sh` has always used, and a first attempt
in `venv-tfm` failed outright with ModuleNotFoundError, 240 rows that were dropped.

Raw `d_q99` against a context without the largest row, median by bin:

| model | 2 to 4 sd | 4 sd and above |
|---|---|---|
| TabPFN-v2.5 | -29.7% | **-51.6%** |
| TabPFN-v2.6 | -54.6% | -48.5% |
| TabICLv2 | -5.2% | -15.4% |
| TabPFN-V3 | -6.8% | -7.8% |
| GBM | +13.3% | -1.7% |
| EXAONE | +2.8% | +0.9% |

Two things follow, and the second is the stronger.

First, the loss is not one model's quirk: two TabPFN generations lose about half
of the predicted Q(0.99) on real subsamples whose largest claim shifts the sd, and
V3 is the least affected of the three. The mechanism weakens across generations
without being removed.

Second, EXAONE does not lose anything, and its source says why it is the right
control. `regressor.py:341` centres the target with `selected_y.mean()` and scales
it with `selected_y.std(correction=1)`: the same non-robust standardisation the
other packages use. What differs is the head, a bank of predicted quantiles rather
than a fixed grid of borders in z-space.

TabICLv2's head was read on 13.9.2026 and it splits that conclusion in three.
`_sklearn/regressor.py:412` fits a plain StandardScaler on the target, the network
predicts 999 quantile values in that scaled space, and line 769 inverts the scaler
affinely: no grid anywhere, and still a 15% loss. So non-robust standardisation
alone costs part of the tail, because an inflated sd squashes the body towards zero
and the network conditions on a nearly degenerate context; the fixed grid then
multiplies that loss to 48 to 52%; and a quantile head can avoid it entirely, as
EXAONE does, for a reason the head type does not explain, since EXAONE and TabICLv2
standardise identically and both predict quantile values. The earlier claim here,
that the damage is in the discretisation and not the standardisation, is withdrawn
as too strong.

Repairs scale with the damage, which is the third piece of evidence for the same
mechanism. Pinball at 0.99 against raw on the ten subsamples at 2 sd and above:

    clip_200   v2.6 10/10 (-17.9%)  v2.5 10/10 (-9.5%)  V3 9/10 (-2.4%)  EXAONE 10/10 (-1.8%)
    sd_cap     v2.6 10/10 (-15.9%)  v2.5  9/10 (-9.4%)  V3 7/10 (-1.7%)  EXAONE  9/10 (-1.6%)
    rank_gpd   v2.6  9/10 (-10.8%)  v2.5 10/10 (-11.6%) V3 9/10 (-5.0%)  EXAONE  6/10 (-1.9%)
    evt_trim   0/10 on every one of the six models: it touches nothing, anywhere.

RUNNING
-------
    python -u experiments/h3_repair/clip_context_real.py
    DATASETS=OnlineNewsPopularity MODELS=TabICLv2 python -u clip_context_real.py
```

---

## `experiments/h3_repair/unit_error_real.py`

```text
A unit error in one row of the context, on real heavy-tailed tables.

WHY
---
`clip_context_real.py` tests the repairs where leverage occurs naturally, and in
the whole pool that is one table, freMTPL2sev. `external_selection.py` added six
heavy-tailed tables from insurance, health expenditure and online popularity,
with Hill indices from 0.48 to 1.05, and none of them reaches an sd shift of 2
in 200 subsamples of 2000; the largest is 1.81. A heavy tail alone does not
produce the leverage that part two measures. What produces it is an error, so
the test on those tables has to bring one: a single context row whose target is
multiplied by 100, a value entered in cents instead of units.

This also gives the repairs a truth that the natural test lacks. The corrupted
row is known, so it can be checked whether an arm touched that row and whether
it touched anything else, and the clean context of the same subsample is the
reference: the prediction the model would have made had the error not been
there.

DESIGN
------
Datasets: freMTPL2sev and `data/external_selected.txt`. On each, REPEATS
random subsamples of N_FIT + N_TEST rows with a positive target, and one row of
the context chosen as the one to corrupt. Both are drawn up front from seed
31337, so a resumed run corrupts the same row.

The row is drawn uniformly among those whose error would reach an sd shift of
at least 2, the level at which part two sees a loss. A first draft drew it
uniformly from all rows; its smoke test on AutoClaims put the corrupted value
at rank 9 with a shift of 1.03, which tests nothing, because a unit error on a
small value is invisible. The condition is the same kind as the bins of the
natural test, and what it conditions away is kept: `share_visible` is the
fraction of context rows at which a unit error would reach that shift.

    factor 1     the clean context; what each arm costs when nothing is wrong
    factor 100   the chosen context row times 100

Arms, all from `clip_context.transform`: raw, clip_50, clip_200, log, tail_log
(C = 50), evt_trim. The reference is raw on the clean context, same subsample,
same seed. `d_q*` is the median relative change of the predicted quantile
against it; pinball is on the clean held-out rows.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  U1. Measured, not predicted: `share_visible`, how often a unit error in a
      random row would be visible at all.
  U2. Under the corruption, raw worsens pinball at 0.99 against the clean
      reference more for the two foundation models than for GBM.
  U3. clip_200 brings pinball at 0.99 back to within 2% of the clean reference
      in most corrupted cells, and costs less than 2% on the clean context.
  U4. evt_trim touches the corrupted row wherever it lands at least ten times
      above the second largest value, and touches a clean context in at most
      5% of subsamples. (The smoke test of `clip_context.py`, seen before this
      was written, showed it missing an injected row at a shift of 20 on the
      generator; the prediction is kept as it was drafted.)
  U5. log removes most of the corruption's effect, and costs more than
      clip_200 at 0.999 on the clean context.

Results in `unit_error_real.csv`.

OUTCOME, 12.9.2026
------------------
Seven tables, three models, ten subsamples each, 2520 rows, none failed. The unit
of the counts below is a dataset-model pair, 21 of them.

  U1 measured. A unit error is visible at 4% of rows on freMTPL2sev, 8 to 11% on
     BlogFeedback, beMTPL97 and MEPS, and 20 to 33% on ausprivauto0405,
     AutoClaims and norauto. Where it lands it lifts the sd shift to 2.8 to 5.5
     and sits 4 to 15 times above the second largest value.
  U2 WRONG. The error costs GBM most, not the foundation models: median pinball
     at 0.99 rises by 35% on BlogFeedback and 25% on MEPS, against TabICLv2's
     9 and 14% and TabPFN-V3's 8 and 3%. On freMTPL2sev it costs nothing
     measurable, because that context's own largest claim already dominates.
  U3 HELD, and it is the strongest argument for the cap that exists. clip_200
     beats raw in 17 of 21 units (p = 0.007, median -2.3%), its worst cell is
     +5.1%, and on a clean context it costs 0.00% at the median and +1.4% at
     worst. C = 200 was picked after a run, but it behaves as a safe default.
  U4 FAILED. evt_trim touches the corrupted row in 10% of cells and nothing else
     in any cell, on any table. Same as on the generator.
  U5 HALF. log repairs as often (17 of 21, -1.8%) and is not safe: 22 of 210
     cells overflow, the worst by 2.8e43, because a quantile over-predicted in
     log space is exponentiated. clip_50 repairs more (-3.7%) and costs more
     (+64% at worst under the error, +85% on a clean context). tail_log is
     neither reliable (15 of 21) nor cheap (+63% at worst).

SECOND RUN, 12.9.2026: THE RANK ARMS UNDER THE SAME ERROR
---------------------------------------------------------
420 cells where every arm is present, 210 of them corrupted. The unit for the
counts is a dataset-model pair, 21; the blow-up shares are over cells, because
that is where the risk lives.

| arm | pinball 0.99 | worst 0.99 | cells >100% worse at 0.999 | worst 0.999 |
|---|---|---|---|---|
| rank_exp | 19/21 (-4.0%) | +0.3% | 5.2% | +84% |
| rank_gpd_trim | 19/21 (-3.6%) | +3.0% | 11.9% | +10191% |
| rank_gpd | 17/21 (-1.9%) | +12.3% | 16.2% | +86195% |
| sd_cap | 17/21 (-2.3%) | +4.9% | 0.5% | +1% |
| clip_200 | 16/21 (-1.3%) | +4.0% | 0.0% | +1% |

So a rank-transformed context repairs the level a user reads most, 0.99, better
than any cap does, and it is the only family that can fail catastrophically two
levels up. The guards split the failure: the trim moves the median (127 of 210
cells better than plain rank_gpd, p = 0.003), the exponential scale moves the
worst case (two orders of magnitude).

Two limits on the rank arms, both from this run:
  * On a clean real context they buy nothing at 0.99 (11 to 13 of 21 units, median
    around zero), unlike on the generator where they were sharper everywhere. Their
    advantage here is conditional on an error being present.
  * The blow-ups are not only TabPFN. Of the 25 cells where rank_gpd_trim is more
    than 100% worse at 0.999, 5 are TabICLv2 on beMTPL97 and norauto. An earlier
    reading of this file's first run said "all TabPFN-V3", and that was wrong.

THIRD RUN, 14.9.2026: TABPFN V2.5 AND V2.6
-----------------------------------------
3080 rows, seven tables, ten repeats, none failed. The unit is a dataset-model pair,
14 of them.

The error costs the two older generations little accuracy: median pinball at 0.99
moves by -6% to +16% by table (BlogFeedback +15.7% and MEPS +8.5% on v2.6, at most
+5.9% on v2.5). That fits the split seen on freMTPL2sev, where grid-headed models
mis-report the upper quantile far more than they lose in pinball.

    clip_200       11/14 (p = 0.057, median -4.0%, worst +2.7%)
    sd_cap         11/14 (p = 0.057, median -4.0%, worst +6.4%)
    clip_50        10/14 (median -1.9%, worst +9.5%)
    rank_exp        8/14 (p = 0.79, median -1.3%, worst +8.3%)
    rank_gpd_trim   8/14 (median -2.0%, worst +60.8%)
    log             6/14 (median +5.3%, worst +3.96e6%)
    evt_trim        3/14 (touches almost nothing)

The caps carry over to the older generations. The rank transform, the best arm at
0.99 on TabPFN-V3, TabICLv2 and GBM (19 of 21, median -5.9%), does not, so the
level-dependent recommendation also depends on the model.

RUNNING
-------
    python -u experiments/h3_repair/unit_error_real.py
    DATASETS=beMTPL97 MODELS=GBM REPEATS=2 python -u unit_error_real.py
```

---

## `experiments/h3_repair/tail_splice.py`

```text
Put the tail back on the output instead of cleaning the input.

WHY
---
Every repair measured so far acts on the context: clip, log, tail_log, evt_trim.
The best of them, a cap at median + 200 robust sd, works but cannot tell a data
error from a genuine extreme, and its threshold was picked from three values.
The literature's answer to the same problem is the other side: leave the context
alone, take the model as a black box, and repair its upper quantiles afterwards
by fitting a generalised Pareto tail to its own errors. Two references carry it,
both checked against Crossref and arXiv before this was written:

  * Pasche, Lam and Engelke (2026), "Extreme conformal prediction: reliable
    intervals for high-impact events", Extremes 29:129-155,
    doi 10.1007/s10687-026-00536-9 (arXiv 2505.08578). Fits a GPD to the
    conformity scores and extrapolates the score quantile past the calibration
    data, which is the only way to get a level beyond 1 - 1/(n_calib + 1).
  * Romano, Patterson and Candes (2019), conformalised quantile regression, the
    base method whose empirical score quantile degenerates at exactly that
    level.

This also tests the mechanism from the other end. If a GPD on the scores
restores Q(0.99) under a context error, then the damage is in the model's
quantile and not in anything the repair needs to know about the error: the
output layer never sees which row was wrong.

WHAT IS MEASURED
----------------
The context of 2000 rows is split once, 1500 to fit and 500 to calibrate, and
the test set of 1000 clean rows is the arbiter. Two treatments of the context,
crossed with two error states, four fits per cell:

    raw         the context as it is
    clip_200    capped at median + 200 * robust_sd of the fit part
    factor 1    clean
    factor 100  one row of the fit part times 100, a value in cents

The corrupted row is drawn among those where the error would reach an sd shift
of 2, as in `unit_error_real.py`, and it always lands in the fit part, so the
calibration rows are clean.

For each level a in 0.9, 0.99, 0.999 the model's own quantile is the base, and
the one-sided conformity score on the calibration rows is s_i = y_i - q_a(x_i):

    none        the model's quantile, as shipped
    cqr_emp     q_a(x) + the ceil((n+1)a)-th smallest score; undefined when that
                index exceeds n, which at n = 500 is exactly a = 0.999, and then
                it falls back to the largest score and is marked degenerate
    cqr_gpd     q_a(x) + a GPD quantile of the scores: peaks over the 90th
                percentile of s, shape and scale by maximum likelihood, the
                level converted with the exceedance rate

Reported per cell: coverage of the bound on the test rows (the number a user
reads), pinball at the same level, the median bound, and the fitted shape.

PREDICTIONS, WRITTEN BEFORE THE RUN
-----------------------------------
  T1. cqr_emp at 0.999 is degenerate at n_calib = 500 and under-covers.
  T2. cqr_gpd covers closer to nominal than cqr_emp at 0.999, and closer than
      the model alone at 0.99 and 0.999.
  T3. Under factor 100 the model's own coverage at 0.99 falls, and both
      conformal layers bring it back without being told that a row is wrong. If
      they do, an output repair replaces the input repair, and the trade-off
      that the clip cannot avoid disappears.
  T4. clip_200 plus cqr_gpd is not better than cqr_gpd on the raw context: once
      the output carries a tail, cleaning the input adds little.

Results in `tail_splice.csv`.

OUTCOME, 12.9.2026
------------------
840 cells, none failed. The unit of the counts is a dataset-model pair, 21.

  T1 HELD structurally. At n_calib = 500 all 840 cqr_emp cells at 0.999 are
     degenerate, since ceil(501 * 0.999) = 501 > 500. It still covers 0.9985,
     because the largest of 500 heavy-tailed scores is large, but the level is
     not identified by the data.
  T2 HELD. cqr_gpd is closer to nominal coverage than the model alone in 16 of
     21 units at 0.999 (p = 0.027), 15 of 21 at 0.99 (p = 0.078), and 16 of 21
     at 0.99 under the error (p = 0.027).
  T3 HELD where it was needed. At 0.99 the error takes TabPFN-V3's coverage from
     0.9855 to 0.9805 and the patch returns it to 0.9910, without being told
     which row is wrong. GBM goes 0.9845 to 0.9900. TabICLv2 was already at
     0.9915 and stays there.
  T4 HELD except on GBM. Under the error at 0.99, clip_200 plus cqr_gpd gives a
     pinball of 328 against 364 for the patch alone on GBM, but 326 against 327
     on TabICLv2 and 284 against 277 on TabPFN-V3.

The price is sharpness, and it is the opposite trade from the clip's. At 0.99 the
patch costs 3.7% of pinball on a clean context and 6.9% under the error; at 0.999
it gains 6.1% and 2.6%. Coverage is bought with a wider bound, which is what a
conformal bound is for. The clip cannot tell an error from an extreme; the patch
does not need to, and never gets sharper at 0.99.

The shape fitted to the conformity scores is 0.14 to 0.61 by dataset, median
0.34. The model's own errors are heavy-tailed, which is why the empirical score
quantile cannot reach 0.999 and why a GPD on those scores can.

SECOND RUN: THE THRESHOLD AS A GRID AND AS A RULE
-------------------------------------------------
The 90th percentile was picked as freely as C = 200 was, so it is now measured
over 0.80, 0.85, 0.90 and 0.95, and chosen by two rules from the scores
themselves: `ks` by the Kolmogorov-Smirnov distance of the fitted tail, `cv` by
out-of-fold pinball of the extrapolated score quantile. Predictions:
  T5. The conclusion does not depend on the threshold: coverage at 0.99 across
      the four fixed thresholds stays within 0.003 of itself.
  T6. Neither rule beats the fixed 0.90 in coverage by more than that. If one
      does, the fixed choice goes and the rule takes its place.
  T7. The two rules disagree with each other more often than either disagrees
      with 0.90, because the KS distance judges the fit of the whole tail while
      the cross-validated loss judges one quantile of it.

THIRD RUN: SHARPNESS, AND WHETHER THE TWO REPAIRS COMPOSE
--------------------------------------------------------
Two additions, both answers to a literature review of the second run.

The review proposed switching to the empirical conformal quantile at 0.99, where
it is not degenerate, on the assumption that it is sharper there. On 1467 cells
where every arm is present it is not: pinball at 0.99 is 414.9 for cqr_emp
against 394.2 to 394.6 for every GPD variant and 405.0 for no patch at all. The
empirical bound is the widest of the three, because at n = 500 the 496th order
statistic of a heavy-tailed score sample is itself an extreme. So the switch is
not made, and the sharpness question is attacked where it belongs: the score.

    cqr_gpd_norm   the score divided by the model's own spread for that row,
                   Q(0.9 | x) - Q(0.5 | x), so
                   the bound scales with each row's spread instead of being one
                   additive constant for all of them. The standard locally
                   adaptive conformal construction, applied to the GPD
                   extrapolation.

    treatment rank_gpd   the third context treatment: the context target becomes
                   its own normal scores and the prediction is mapped back
                   through the spliced empirical-Pareto inverse. The review
                   argued that this reduces to the output patch with a different
                   body model. It does not: the rank transform removes the
                   leverage before the network sees it (borders inside the data
                   3547 against 421), while the patch leaves the context
                   contaminated and repairs the output. Crossing the two settles
                   whether they compose or are redundant.

Predictions:
  T8. cqr_gpd_norm lowers pinball at 0.99 against cqr_gpd_0.9 at coverage no
      worse than nominal, on the datasets whose predicted spread varies most
      across rows.
  T9. On the rank_gpd context the patch adds less than it adds on the raw
      context, because the leverage is already gone; if instead it adds the same
      amount, the two repairs are independent and should be reported as a pair.

FOURTH RUN: THE SPLICE POINT ITSELF CONDITIONAL
-----------------------------------------------
Every arm above splices at one number: a quantile of the pooled scores. The
conditional version moves the splice with the row. `cond_gpd` takes the model's
own Q(0.9 | x) as the threshold, divides the exceedances over it by that row's
spread Q(0.9 | x) - Q(0.5 | x), pools those, fits a GPD, and extrapolates per
row:

    Q(a | x) = Q(0.9 | x) + spread(x) * (sigma / xi) * (((1 - a) / zeta) ** -xi - 1)

with zeta the measured exceedance rate of the model's own Q(0.9 | x) on the
calibration rows, which also corrects for that quantile being miscalibrated. This
is the classical conditional peaks-over-threshold construction with an
intermediate conditional quantile, the same shape as gradient boosting for
extreme quantile regression and extremal random forests, except that the
intermediate quantile comes from the model under test rather than from a purpose
built quantile regressor. Run with GBM as that model it is a baseline in the
spirit of gbex, built from parts already here; it is not the published
implementation and is not claimed to be.

It differs from the conformal arms in what it gives up: there is no exchangeability
argument behind it, so no coverage guarantee, only an EVT extrapolation. That is
the trade to report.

Predictions:
  T10. cond_gpd is sharper at 0.99 than cqr_gpd_0.9, because the bound moves with
       the row instead of adding one constant to every row.
  T11. Its coverage at 0.999 is closer to nominal than the model alone but less
       reliable than the conformal arms, since it has no finite-sample guarantee.
  T12. Under the unit error it degrades less than the additive arms, because the
       threshold it splices at is itself re-estimated per row.

OUTCOME OF THE SECOND, THIRD AND FOURTH RUNS, 12.9.2026
-------------------------------------------------------
50400 rows, none failed. Every comparison is paired on the same cell, and the
unit is a dataset-model pair, 21 of them. That matters here: a pooled median over
arms measured on different cell sets suggested a 3% gap between thresholds that
the paired test says is 0.06%.

  T5 HELD, strongly. At 0.99 the four fixed thresholds differ by at most 0.06% of
     pinball, paired, and coverage is 0.990 to 0.991 for all of them.
  T6 HELD. Neither rule beats the fixed 0.90: the KS rule is better in 11 of 21
     units (p = 1.0, median -0.01%) and the cross-validated rule in 11 of 21
     (median -0.06%). They cost nothing either, so a rule can replace the hand
     choice for free, which is the useful form of this result.
  T7 STRONGER than predicted. The rules disagree about the threshold, often: KS
     picks 0.80 in 49% of cells and 0.95 in 12%, cross-validation picks 0.90 in
     35% and 0.80 in 25%. The outcome does not move. The threshold is not a knob
     worth arguing about at this calibration size.
  T8 FAILED. Dividing the score by the model's own spread is not sharper: at 0.99
     it is better in 7 of 21 units on a clean context (median +0.5%, worst +25%),
     and at 0.90 it is significantly worse (5 of 21 under the unit error, two
     sided p = 0.027; 4 of 21 on a clean context). Locally adaptive scores buy
     nothing here, and add tail risk.
     What the 0.90 row actually compares, which the arm names hide. At a
     threshold of 0.9 the level 0.90 sits inside the calibration data, so
     `gpd_quantile` returns None for BOTH arms and both fall back -- by design,
     not by failure: 100% of rows at that level carry no fit. So that line is
     locally adaptive EMPIRICAL conformal against plain empirical conformal, with
     no Pareto tail on either side. It is still the textbook test of the locally
     adaptive score, and the verdict stands; it is simply not a statement about
     the GPD. Only the 0.99 row, where both arms really fit, is. A re-run cannot
     move the 0.90 number, and `cqr_gpd_norm` is fixed at a 0.9 threshold, so the
     locally adaptive score is never measured against a live tail at that level.
     Doing that needs a `cqr_gpd_norm_0.8` arm, the way cqr_gpd_0.8 and 0.85 do
     fit at 0.90; it is not measured here.
  T9 NOT HELD, and that settles an argument. The patch adds about as much on a
     rank-transformed context as on a raw one (12 of 21 units better at 0.99,
     median -1.6%, and 12 of 21 at 0.999), so the rank transform is not the patch
     with a different body model: the two act on different objects and compose.
     The rank context is also better on its own at 0.99 (15 of 21, -2.8%) and
     unusable at 0.999 (worst +175232%), which is the instability the fifth run of
     `clip_context.py` addresses.
  T10 and T12 FAILED. The conditional splice is worse than the additive patch:
     at 0.99 it is better in 5 of 21 units on a clean context (p = 0.027, median
     +14.4%) and at 0.999 under the error in 3 of 21 (p = 0.001, +13.5%), with
     worst cells of +72800% and +1.7e8%. Splicing per row multiplies a pooled
     shape by a per-row spread, and where either is large the product is not a
     quantile any more. Conditionality is not free.
  T11 HELD for coverage only. cond_gpd covers 0.990 and 0.991 at 0.99 and 0.998
     to 0.999 at 0.999, as close to nominal as the conformal arms, while being
     the least sharp of them. Coverage alone does not rank these methods.

So the output side keeps exactly one recommendation, the plain conformal patch
with a GPD on the scores, and its threshold can be chosen by a rule instead of by
hand. The two attempts to improve it, locally adaptive scores and a conditional
splice point, both failed, and both are reported.

FIFTH RUN, 13.9.2026: 5000 TEST ROWS, WHERE 0.999 CAN BE SEEN
-------------------------------------------------------------
`DATASETS=freMTPL2sev N_TEST=5000 OUTPUT=tail_splice_n5000.csv`. At 1000 test rows,
coverage at 0.999 moves in steps of 0.001, so 0.996 and 0.999 are three
exceedances apart. The unit is the repeat: all test rows of a repeat share one
calibration sample, and a binomial test over pooled test rows is anti-conservative.

At 0.99 the patch holds. The model alone is below nominal in 10 of 10 repeats for
GBM and TabPFN-V3 (p = 0.002); with the patch 5, 5 and 7 to 8 of 10, none
significant. cqr_emp stays inside its Beta(496, 5) band [0.9818, 0.9961].

At 0.999 it does not, for the foundation models. Every model alone is below in 10
of 10. The patch brings GBM to nominal (0.9992, 0.9995) but leaves TabICLv2 at
0.9956 to 0.9961 (9 of 10 below under the error, p = 0.021) and TabPFN-V3 at 0.9955
to 0.9960 (8 of 10): about four times the nominal exceedance rate, and no better
than the degenerate empirical bound. T2's "16 of 21 at 0.999" was measured on 1000
test rows and is withdrawn for the foundation models. The likely cause is the
calibration size: 500 rows give about 50 exceedances, and 0.999 is two orders
beyond them.

RUNNING
-------
    python -u experiments/h3_repair/tail_splice.py
    DATASETS=AutoClaims MODELS=GBM REPEATS=2 python -u tail_splice.py
```

---

## Thread count, 15.9.2026

```
No prediction was written before this run. It checks one sentence of an earlier outcome,
`clip_context_real.py` P1 of 11.9.2026: "Within one environment TabPFN is deterministic:
6 and 12 threads give identical output." That holds only at a fixed thread count.

    XI=0.7 SEEDS=2 SD_SHIFTS=1,20 OMP_NUM_THREADS=t MKL_NUM_THREADS=t \
      python -u experiments/h3_repair/clip_context.py        t = 3, 4, 6, 10, 12

At a7f3ae9 in venv-tfm, 52 cells per thread count, all in `results/side/thread_count.csv`.

OUTCOME
-------
6, 10 and 12 threads give the same output bit for bit. 3 and 4 do not: against 6
threads, pinball at 0.99 moves by a median 0.3% (worst 4%), at 0.999 by 1.2 to 1.3%
(worst 18% and 52%), the implied xi by at most 0.05, the borders over the data not at all.

Arm effects move less. An arm's median pinball at 0.99 against clip_200 moves across the
five thread counts by at most 0.12 points for sd_cap, 0.4 to 1.0 for the rank arms and
log, and 2.4 for raw at a shift of 20, where the effect is +10 to +12.5%. Only rank_exp
at a shift of 20 changes sign, within half a point of zero.

The committed `clip_context.csv` equals the 6-thread run on 32 of the 36 non-rank cells
the two share and the 4-thread run on the other 4. Its rank cells differ from the current
code by a median 0.4% and at most 1.6%, the splice fix of 3cb55cc.

Nothing is remeasured. A difference between arms under one point at 0.99 is not read as
an effect, and 0.999 is not read per cell. Runs from here on set OMP_NUM_THREADS=6.
```

## `experiments/h3_repair/prior_pretraining_eval.py`, 15.9.2026

```
Is the upper-tail failure in the pre-training prior or in the architecture? A nanoTabPFN
of TFM-Playground's architecture, pre-trained on a GPU machine under a 2 x 2 of prior (A0
TabICL's SCM prior, A1 with heavy tails and contaminated contexts) and target encoding
(B0 mean and sd, B1 median and IQR / 1.349). The predictions were committed in 66aebe1,
before any model existed.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Q1, the gate. Under A0B0, a shift of 20 or more lowers the median predicted Q(0.99) against
the clean context of the same seed in at least 8 of 10 generator seeds per pre-training
seed, at both xi, and moves the implied xi at a shift of 50 down by at least 0.2 or below
zero. If A0B0 does not reproduce the failure, the small model is not a valid proxy: that is
the result, and Q2 to Q5 are not interpreted.

Q2. A1B0 loses less of Q(0.99) at shifts 20 and 50 than A0B0, by at least half of A0B0's
loss, paired over cells; its pinball at 0.99 on a clean context is at most 5% worse than
A0B0's.

Q3. A1B0 captures a larger share of the conditional tail shape than A0B0 in
`shape_of_x.py` with the same seeds.

Q4. A0B1 keeps the borders over the data within 10% of its clean-context count at every
shift, where A0B0 loses more than half at a shift of 50; A1B1 has the lowest pinball at
0.99 under shifts 20 and 50 of the four arms.

Q5. No arm reaches the shape share of the GBM control in `shape_of_x.py`.

OUTCOME, 15.9.2026: THE GATE FAILS ON BOTH PRE-TRAINING SEEDS
-------------------------------------------------------------
Only arm A0B0 was evaluated, on pre-training seeds 1 and 2 (7000 steps of 8 tables each,
1000 buckets), 10 generator seeds, xi 0.7 and 0.9, 768 context rows. The rows are in
`results/h3_repair/prior_pretraining_eval.csv`. Seed 1 was read on the CPU at
OMP_NUM_THREADS=4 and seed 2 on the GPU; the effects below are factors of 12 and more,
where the thread count moved TabPFN's pinball at 0.99 by at most 4%.

Q1  FAILED. A shift of 20 or 50 lowers the median predicted Q(0.99) against the clean
    context in 0 of 10 generator seeds, for both pre-training seeds and both xi. It raises
    it instead, 12 to 16 times at a shift of 20 and 34 to 47 times at 50. The implied xi
    at a shift of 50 rises, from +0.76 to +1.56 on the clean context to +2.22 to +2.26,
    and falls by 0.2 or below zero in 0 of 10 seeds. The grid does stretch as in TabPFN:
    the borders over the data fall from 654 on a clean context to 170 at a shift of 50
    (xi 0.7; 612 to 200 at xi 0.9), the same for both seeds, since the edges and the
    encoding are the same. The small model reads the stretched grid as a heavier tail
    where TabPFN reads it as a lighter one. By the rule above it is not a valid proxy for
    this failure, and Q2 to Q5 are not interpreted.

    It is no proxy for part one either. In `shape_of_x.py` and `scale_of_x.py` at 768
    training rows, 20 seeds, one member, A0B0 reproduces -0.002 and -0.010 of the true
    shape slope and -0.002 and -0.001 of the true change in the conditional median: it
    ignores x. The GBM control at the same size keeps 0.31 of the shape and 0.85 of the
    scale. After 56000 tables the small model has learned a marginal distribution, not a
    conditional one.

Q2, Q3, Q4, Q5  NOT TESTED, because the gate failed. A1B0-s1 was stopped at step 250 of
    7000. A0B1-s1 had diverged before the gate was read (median loss 1.49 at step 3000,
    12.47 at 7000, only the decoder grown) with the encoded target capped at 1e4, so B1
    would have needed another fix before any comparison.

What this does not say: that the failure lies in the architecture rather than the prior.
A model this weak cannot separate the two. What it says: the direction of the grid effect
depends on what the network has learned to put into the stretched bars, so the question
needs a model that at least tracks the conditional scale, which this budget on an RTX 2060
did not reach.

    medians over xi 0.7 and 0.9 and 10 generator seeds
    arm       Q(0.99) at shift >= 20  implied xi at 50  clean pinball 0.99  shape share
    A0B0-s1   +3163%                  2.255             3.615               -0.002
    A0B0-s2   +3421%                  2.225             4.444               -0.010
    A0B1-s1   diverged                -                 -                   -
    A1B0-s1   stopped at step 250     -                 -                   -
    A1B1      not trained             -                 -                   -
```

## `experiments/h3_repair/tabicl_prior_finetune.py`, 16.9.2026

```
Continue TabICLv2's pre-training under a heavy-tailed, contaminated prior, and measure
whether its tail under leverage on freMTPL2sev comes back. The nanoTabPFN route of
`prior_pretraining_eval.py` never produced a valid proxy at usable table sizes on this
GPU; TabICLv2 is the model the thesis measured. The predictions were committed in
104cf50, before any arm was trained.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
On `clip_context_real.py`, freMTPL2sev, N_FIT=2000, the raw arm against without_max, the
same 20 subsamples as the base model, paired per subsample; medians over both pre-training
seeds.

F1, the control. FT-A0 keeps the base model's behaviour: its median d_q99 in the bin at 4
sd and above is within 5 points of the base model's -15.4%, and its median pinball at 0.99
in the two bins below 2 sd is within 5% of the base model's.

F2. FT-A1 at least halves the loss: its median d_q99 at 4 sd and above is -7.7% or closer
to zero, and closer to zero than the base model's in at least 4 of those 5 subsamples.

F3. FT-A1 costs little where there is no leverage: its median pinball at 0.99 in the two
bins below 2 sd is at most 5% above the base model's.

F4. Where the leverage is, FT-A1 predicts better than the control: its pinball at 0.99 in
the bins at 2 sd and above is lower than FT-A0's in at least 8 of those 10 subsamples.

OUTCOME, 16.9.2026
------------------
Four arms fine-tuned, FT-A0 and FT-A1 on seeds 1 and 2, 20000 tables each, 17:12 to 19:17;
evaluated by run/tabicl_prior_finetune_eval.sh, rows in
`results/h3_repair/clip_context_real_tabicl_ft.csv`. The base model was measured again in
the same environment (tabicl 2.2.0) and reproduces its committed rows (tabicl 2.1.1) to
7.8e-06 on all 20 cells. A subsample's value is the mean over the two seeds.

F1  FAILS, narrowly and usefully. FT-A0's median d_q99 at 4 sd and above is -11.1%
    against the base model's -15.4%, 4.3 points and inside the bound, but its median
    pinball at 0.99 below 2 sd is 5.2% lower than the base model's, outside it by 0.2.
    Continued training on the model's own prior is not a no-op: it improves the clean
    contexts and takes more than a quarter of the leverage loss away with no heavy tail
    and no contamination. The comparison for the prior is FT-A1 against FT-A0.

F2  FAILS. FT-A1's median d_q99 at 4 sd and above is -9.0%, not -7.7% or closer, though
    closer to zero than the base model's in 5 of 5 subsamples (-15.4, -14.4, -19.2, -13.0
    and -27.7% to -1.2, -9.0, -14.3, -7.0 and -20.1%). The seeds disagree: -13.0% for
    seed 1, -5.0% for seed 2.

F3  HOLDS. FT-A1 costs nothing on clean contexts: its median pinball at 0.99 below 2 sd
    is 6.5% below the base model's, and above it in 2 of 10 subsamples.

F4  HOLDS, at the threshold. At 2 sd and above FT-A1's pinball at 0.99 is lower than
    FT-A0's in 8 of 10 subsamples (one-sided sign test p = 0.055), median ratio 0.966;
    against the base model in 9 of 10, median ratio 0.952.

What this says. Heavy tails and contaminated contexts in TabICLv2's prior move its tail
under leverage in the predicted direction, but the part owed to the prior beyond
fine-tuning as such is small (median d_q99 at 4 sd and above -9.0% against FT-A0's
-11.1%), not consistent across the two seeds, and at the edge of significance on ten
subsamples of a single dataset. A cap on the context still removes the rest in every
arm (clip_200 at 4 sd and above: +2.4% and +2.5% for FT-A1, -3.6% and -3.7% for FT-A0).
The prior is not the whole of the failure, and the evidence that it is part of it is
weak; more seeds and a second dataset would be needed to say more.

    medians, subsample values averaged over the two seeds
    model      d_q99 at 4 sd+   pinball 0.99 below 2 sd   pinball 0.99 at 2 sd+ vs base
    base       -15.4%           504                        1
    FT-A0      -11.1%           478 (-5.2%)
    FT-A1       -9.0%           471 (-6.5%)                0.952 (lower in 9 of 10)
```

## `experiments/h3_repair/prior_pretraining_eval.py`, second outcome, 17.9.2026

```
The predictions Q1 to Q5 are those of the section above, committed in 66aebe1.

SECOND OUTCOME, 17.9.2026: A CURRICULUM THAT LEARNS, AND THE PRIOR DECIDES THE COLLAPSE
-------------------------------------------------------------------------------------
Every later run from scratch at more than 3 features or more than 50 rows stayed at the
loss of the context marginal (`data/tfmp_artifacts.json`). A model that learned at 50 rows
and 3 features was carried on instead: A0B0-s42 to 8 features (A0B0-s91, linear-task R2
+0.911 at 35 rows), then to 256 rows for 3755 steps of 16 tables in each arm
(`run/prior_pretraining_curriculum.sh`). Every arm starts from s91, so the arms differ only
in the prior and the encoding of that last stage. Seed 92 reads two 256-row shards, seed 93
the other two. Evaluation at N_TRAIN=200, inside the 128 to 223 rows of pre-training.
Learning check (linear task, 200 rows): A0B0 +0.950 and +0.948, A1B0 +0.939 and +0.935;
A0B1-s92 and A1B1-s92 end at NaN and learn nothing. Rows in
`results/h3_repair/prior_pretraining_eval.csv` (s92) and `prior_pretraining_eval_curriculum_1.csv`
and `_2.csv`; shares in `shape_of_x_nanotabpfn_n200_curriculum.csv` and its scale twin.

Q1  HALF. A0B0-s92 holds: Q(0.99) below the clean value in 10, 10, 10 and 9 of 10 seeds at
    shifts 20 and 50 and both xi, falling to 2 to 4% of it, the implied xi at 50 down by
    2.50 to -0.06 in 20 of 20. A0B0-s93 holds at a shift of 50 (10 and 10 of 10) and for
    the implied xi (down by 2.09 in 20 of 20) but not at 20 (6 and 7 of 10). The borders
    over the data fall from 64 to 9, as in TabPFN.

Q2  HOLDS on both seeds. A0B0 loses a median 97% and 94% of Q(0.99) at shifts 20 and 50;
    A1B0 loses none and rises by 71% and 38%, at most half of A0B0's loss in 31 and 30 of 40
    cells. Median Q(0.99) over both xi, clean / 20 / 50: A0B0-s92 1991 / 105 / 29, A1B0-s92
    25 / 40 / 43, against true values near 35 (xi 0.7) and 69 (xi 0.9). Pinball at 0.99
    under shifts 20 and 50 is
    lower for A1B0 in 40 of 40 cells on each seed (sign test p = 9e-13); on the clean
    context it is 5.6 times lower, not 5% higher.

Q3  HOLDS on both seeds. Shape share at 200 rows, 20 seeds: A1B0 +1.093 and +1.071, A0B0
    +0.100 and -0.080. Scale share: A1B0 0.410 and 0.384, A0B0 0.324 and 0.356.

Q4  NOT TESTED. Both B1 arms reached NaN when the encoding changed at the last stage.

Q5  WRONG as written, and empty at this size. GBM at 200 rows has a shape share of -0.428
    (scale 0.782): 200 rows do not give a tree a usable Q(0.99), so no arm could fail to
    reach it. A1B0 exceeds it at +1.09.

What this says. In a model of TabPFN's architecture that learns, trained identically up
to its last stage, one leverage row collapses the upper tail under the standard prior and
does not under a prior with heavy tails and contaminated contexts, while the grid stretches
exactly the same in both (64 to 9 borders over the data). Stretching the grid is not
enough for the collapse; what the network has learned to put in the stretched bars
decides it. That supports the reading of `clip_context_real.py`, that EXAONE and TabICLv2
differ by what they learned and not by their pipeline, and it matches the direction of the
TabICLv2 fine-tuning (`tabicl_prior_finetune.py`), where the same prior helped a little.

What this does not say. A1 adds two things at once, a GPD tail whose shape follows a
feature and a contaminated context row, and the generator of this experiment is a GPD
with a covariate-dependent shape: part or all of A1B0's gain may be that its prior now
resembles the test family, which also explains its clean calibration and shape share. An
arm with the tail alone and one with the contamination alone would separate the two. The
arms share their first two stages, the contexts are 200 rows against the thesis's 768 to
2000, the model is small, and A0B0's clean tail is far off (Q(0.99) near 1900 against a
true 35 to 69), so the size of the effect does not transfer to the production models.
```

## `experiments/h3_repair/prior_pretraining_eval.py`, the treatments of A1 alone, 17.9.2026

```
PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
A1B0 adds two things at once, a GPD tail whose shape follows a feature and a contaminated
context row, and the generator of the evaluation is a GPD with a covariate-dependent
shape. Two arms separate them: ATB0 with the tail alone and ACB0 with the contamination
alone (`run/prior_pretraining_curriculum.sh`, streams 3 and 4). Both start from A0B0-s91
with s92's budget and shards, and both take A1's own draws on the same tables
(`prior_arms.apply_a1`), so A1B0, ATB0 and ACB0 of one seed differ only in what is applied.
Seeds 92 and 93, the learning check, N_TRAIN=200, 10 generator seeds, xi 0.7 and 0.9, and
`shape_of_x.py` and `scale_of_x.py` with 20 seeds, all as for A1B0.

What each prior shows the network, measured on the first 2000 tables of small_s5026 with
seed 92 before any training step was read: the context sd shift (`metrics.sd_shift`) is 4
or more in 0.0% of contexts under A0B0, 3.6% under ATB0, 29.2% under ACB0 and 31.7% under
A1B0; 20 or more in 0.0%, 0.0% (largest 27), 9.8% and 9.8%.

Every comparison is with A0B0 of the same pre-training seed, paired by xi and generator
seed. "Meets Q2" is Q2's criterion as it was read for A1B0: a loss of Q(0.99) at shifts 20
and 50 at most half of A0B0's in more than half of the 40 cells.

R0, the gate. An arm whose linear-task R2 is below 0.9 has not learned and is not read.

R1. ACB0 meets Q2 on both seeds, and its pinball at 0.99 under shifts 20 and 50 is lower
    than A0B0's in more than 30 of 40 cells on each seed.

R2. ATB0 does not meet Q2 on either seed. Its prior never shows a context row at a shift of
    20 or more.

R3. The shape share comes from the tail: above 0.5 for ATB0 and below 0.5 for ACB0, on both
    seeds.

R4. So does the clean calibration: ATB0's pinball at 0.99 on a clean context is at most
    half of A0B0's, median over the 20 cells of each seed, and ACB0's is not.

Reading fixed in advance. R1 with R2: the collapse is prevented by leverage rows in the
prior, not by a prior that resembles the test family, and the tail supplies only the
shape (R3, R4). ATB0 meeting Q2 while ACB0 does not: the resemblance carries it, and the
claim narrows to that. Both meeting Q2: either treatment suffices. Neither: only the two
together, and the ablation does not separate them.

What a hold would not say. ACB0's contamination is one row placed above the mean at a
chosen sd shift, the same construction as the leverage row of the evaluation
(`leverage.add_row` at `metrics.y0_for_sd_shift`). R1 holding shows that a network shown
that perturbation discounts it, not that it withstands contamination of another kind.
```

## `experiments/h3_repair/prior_pretraining_eval.py`, the treatments of A1 alone, outcome, 17.9.2026

```
The predictions R0 to R4 are those of the section above, committed in e7b763e.

THIRD OUTCOME, 17.9.2026: THE TAIL ALONE STOPS THE COLLAPSE, THE CONTAMINATION ALONE ONCE
----------------------------------------------------------------------------------------
Arms ATB0 (A1's heavy tail alone) and ACB0 (its contaminated context alone), streams 3 and
4 of `run/prior_pretraining_curriculum.sh`, as A1B0 in every other respect and with A1's
own draws on the same tables: at step 200 a tail in 0.4913 of seed 92's tables and a
contamination in 0.2981, the shares of A1B0-s92. The predictions R0 to R4 were committed
in e7b763e, before any step was read. Rows in `prior_pretraining_eval_curriculum_3.csv`
and `_4.csv`; shares appended to the curriculum shape and scale files. "Meets Q2" is Q2's
criterion as it was read for A1B0, a reading that reproduces A1B0's 31 and 30 of 40.

               meets Q2   median Q(0.99)         clean pinball    shape share
               (of 40)    clean / 20 / 50        / A0B0's
    A0B0-s92       -      1991 /  105 /  29       1                +0.100
    A1B0-s92      31        25 /   40 /  43       0.19             +1.093
    ATB0-s92      35        20 /   77 / 121       0.16             +1.419
    ACB0-s92      14       189 /   37 /  23       0.50             +0.673
    A0B0-s93       -      1860 /  202 /  79       1                -0.080
    A1B0-s93      30        25 /   36 /  30       0.22             +1.071
    ATB0-s93      33        20 /  147 / 279       0.18             +0.702
    ACB0-s93      24        45 /   58 /  33       0.39             +0.562

R0  HOLDS. Linear-task R2 at 200 rows: ATB0 +0.947 and +0.944, ACB0 +0.950 and +0.944.

R1  WRONG. ACB0 meets Q2 on seed 93 (24 of 40) but not on 92 (14 of 40), where its Q(0.99)
    still falls by a median 77% and 85% at shifts 20 and 50. Its pinball at 0.99 under
    those shifts is lower than A0B0's in 40 of 40 cells on both seeds.

R2  WRONG. ATB0 meets Q2 on both seeds (35 and 33 of 40), although its prior showed no
    context row at a shift of 20 in the 2000 tables measured before the run.

R3  HALF. ATB0's shape share is above 0.5 (+1.419 and +0.702), but so is ACB0's (+0.673 and
    +0.562). The scale shares do not separate the arms (0.307 to 0.411; A0B0 0.324, 0.356).

R4  HALF. ATB0's clean pinball is 0.16 and 0.18 of A0B0's, but ACB0's is at most half too,
    0.495 and 0.386 (ratio of the medians over 20 cells).

The reading fixed in advance, seed by seed: on 92 the tail meets Q2 and the contamination
does not, "the resemblance carries it"; on 93 both do, "either treatment suffices". Only
the tail meets it on both seeds, so the claim narrows. In this experiment a prior with a
GPD tail stops the collapse, and that tail cannot be told apart from the GPD family of the
test. That contaminated contexts in the prior protect against a leverage row is not shown.

Not predicted, read after the outcome. Q2's criterion counts a rise as no loss, and the
tail alone meets it by overshooting: its median Q(0.99) climbs to 121 and 279 at a shift of
50 against a true 35 to 69, and its implied xi at 50 rises (+0.09 and +0.66), where A1B0's
moves by -0.12 and -0.33. Shown heavy tails but never a contaminated row, the network
reads the leverage row as a heavier tail. Under shifts 20 and 50 the pinball at 0.99 of
A1B0 is lower than ATB0's in 29 and 34 of 40 cells (median ratio of ATB0 to A1B0 1.06 and
1.49) and lower than ACB0's in 33 and 32 (1.03 and 1.05). Neither treatment alone does as
well under leverage as the two together, while on the clean context the tail alone is a
little better than both (lower than A1B0 in 13 and 17 of 20). The borders over the data
fall from 64 to 9 in all four arms on both seeds: the grid stretches the same under four
priors, and what the network puts in the stretched bars differs.
```

## `prior_pretraining_eval.py` on other tail families and `prior_pretraining_real.py`, 17.9.2026

```
PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
The tail of the prior alone (ATB0) kept Q(0.99) under a leverage row on both seeds, but
that prior is a GPD and so is the test. The same eight models (A0B0, A1B0, ATB0 and ACB0 on
seeds 92 and 93) are read twice more, without further training, at 200 context rows:

  families  `prior_pretraining_eval.py`, FAMILIES=frechet,burr,studentt: Q2's design,
            leverage row, shifts 1, 4, 20 and 50, xi 0.7 and 0.9 and 10 seeds, with a tail
            of the same index and scale and another body (`generator.family`, the samplers
            of `tail_families.py` moved unchanged). One seed gives every family the same X
            and U.
  real      `prior_pretraining_real.py`: 20 subsamples of 200 context and 1000 held-out
            positive rows per dataset, drawn from seed 31337 with a natural sd shift below
            1.2, from the five claim tables with at most 16 features (freMTPL2sev,
            beMTPL97, ausprivauto0405, norauto, AutoClaims). The same added row at the
            context's feature mean, at shifts 4, 20 and 50. Neither tail nor body chosen.

Why not the natural leverage of `clip_context_real.py`. At 200 rows the natural sd shifts
of freMTPL2sev reach 4.4 to 14.9, and at a shift of 4 on the GPD, in rows already
committed, A0B0's Q(0.99) does not collapse but rises (it falls in 3 and 2 of 20 cells).
A test there could not show a collapse to prevent.

Seen before this was written, and disclosed: the gpd branch of the changed
`prior_pretraining_eval.py` reproduces two committed A0B0-s92 cells in every compared digit;
the three families give finite quantiles for ATB0-s93 at one seed and xi 0.9 (values
printed); all five datasets give 20 clean subsamples where a shift of 50 is reachable; one
real cell of AutoClaims was checked for finite output only. And one natural freMTPL2sev
subsample of `clip_context_real.py` (bin 4-inf, repeat 0, shift 5.9) was fitted before the
design above was chosen: median Q(0.99) without its largest row / with it, A0B0-s92 728412 /
1947485, ATB0-s92 837518 / 22139, GBM 6371 / 7783; the Q(0.99) of all claims is 18278.

Criteria, as for A1B0 and the ablation. The loss is 1 - Q(0.99) / Q(0.99) of the clean
context of the same cell. "Meets Q2": a loss at shifts 20 and 50 at most half of A0B0's
(same pre-training seed, same cell) in more than 20 of 40 cells, per family or dataset and
per seed. On the families a cell is (xi, seed, shift), on real data (subsample, shift).

F0, gate. A0B0's Q(0.99) falls at shifts 20 and 50 in more than 20 of 40 cells, per family
    and seed. F1 and F2 are read on the families that pass.
F1. ATB0 and A1B0 meet Q2 on every family that passes, on both seeds.
F2. ATB0, A1B0 and ACB0 each have a lower pinball at 0.99 than A0B0 under shifts 20 and 50
    in more than 30 of 40 cells, per family and seed.

R0, gate. As F0, per dataset and seed. R1 to R3 are read on the datasets that pass.
R1. ATB0 meets Q2 on at most half of the passing datasets on each seed: on claims its tail
    no longer resembles the test, and the one natural subsample above showed it collapsing.
R2. A1B0 meets Q2 on at least half of the passing datasets on each seed.
R3. Pinball at 0.99 under shifts 20 and 50: ATB0 lower than A1B0 in fewer than 20 of 40
    cells on more than half of the passing datasets, on each seed.

ACB0 is read and not predicted on either set: on the GPD its two seeds disagreed.

Reading fixed in advance.
  F1 and ATB0 meeting Q2 on more than half of the passing datasets on both seeds: the tail
    protects beyond the GPD, and the
    resemblance reading of the ablation is rejected.
  F1 and R1: the protection holds for any tail of the prior's index but not for claims;
    the claim stays with synthetic tails.
  F1 failing: it depends on the GPD body, and the ablation's narrowing stands as written.
  R0 failing on most datasets: the small model does not reproduce the failure on claims,
    and real data does not decide between the readings.
```

## `prior_pretraining_eval.py` on other tail families and `prior_pretraining_real.py`, outcome, 17.9.2026

```
OUTCOME, 17.9.2026: THE CRITERION FAILS BEFORE THE PRIORS DO
-----------------------------------------------------------
The predictions are those committed in 77f3176. Rows in
`results/h3_repair/prior_pretraining_eval_families.csv` (1920) and
`prior_pretraining_real.csv` (3200); no cell failed. "Meets Q2" counts, of 40 cells, those
where the loss of Q(0.99) at shifts 20 and 50 is at most half of A0B0's; it takes more
than 20.

                          seed   A0B0 falls    meets Q2:  A1B0   ATB0   ACB0
    families  burr         92        39                    19     12     22
              frechet      92        39                    25     20     22
              studentt     92        35                    18     22     12
              burr         93        32                    18     18     24
              frechet      93        32                    19     22     24
              studentt     93        29                    22     25     16
    claims    AutoClaims   92        40                     0      0      0
              ausprivauto  92        39                     0     10      0
              beMTPL97     92        38                     4      6      0
              freMTPL2sev  92        40                     0      0      0
              norauto      92        40                     0      0      0
              AutoClaims   93        20                     0      0      0
              ausprivauto  93        20                     0      5      0
              beMTPL97     93        25                     4      8      9
              freMTPL2sev  93        21                     0      0      0
              norauto      93        20                     0      0      0

F0  HOLDS on every family and seed (29 to 39 of 40).
F1  WRONG. A1B0 meets Q2 on 2 of the 6 family-seed pairs and ATB0 on 3; neither on burr.
    ACB0, not predicted, meets it on 4.
F2  HOLDS. Pinball at 0.99 under shifts 20 and 50 is lower than A0B0's in 40 of 40 cells for
    each of the three arms, on every family and seed.

R0  HOLDS on all five tables for seed 92, and on seed 93 only on beMTPL97 and freMTPL2sev.
    On the other three, A0B0's Q(0.99) falls at a shift of 50 only (20 of 40); at 20 it
    rises.
R1  HOLDS. ATB0 meets Q2 on none of the passing tables (0 of 5, 0 of 2).
R2  WRONG. A1B0 meets it on none either (0 of 5, 0 of 2), and nor does ACB0.
R3  HALF. ATB0's pinball under leverage is lower than A1B0's in fewer than 20 of 40 cells on
    2 of 5 tables for seed 92 (12 to 27 of 40, median ratio 0.98 to 1.06) and on 2 of 2
    for seed 93 (8 and 17).

By the reading fixed in advance, F1 failing leaves the ablation's narrowing as written.

Not predicted, read after the outcome, and it changes what Q1, Q2, F1 and R1 to R3 can
mean. Their loss is measured against each model's own clean Q(0.99), and that is far off.
Median over cells of the predicted Q(0.99) over the true one (GPD and families: the true
conditional quantile; claims: the empirical 0.99 quantile of the 1000 held-out rows), and
the median pinball at 0.99 over the same model's clean pinball, seed 92 / 93:

                    Q(0.99) / true                            pinball / clean pinball
                    clean         shift 20      shift 50      shift 20      shift 50
    GPD       A0B0  42.4 / 40.0   2.30 / 5.51   0.71 / 1.68   4.36 / 5.95   7.02 / 6.18
              A1B0  0.58 / 0.54   0.79 / 0.78   0.87 / 0.62   0.66 / 0.62   0.67 / 0.63
              ATB0  0.51 / 0.52   1.70 / 3.23   2.38 / 6.01   0.83 / 0.97   0.84 / 1.54
              ACB0  4.65 / 0.95   0.75 / 1.14   0.46 / 0.66   0.23 / 0.52   0.25 / 0.41
    families  A0B0  41.3 / 38.9   2.42 / 12.1   0.76 / 2.15   4.79 / 6.09   7.36 / 6.71
              A1B0  0.94 / 2.86   0.84 / 0.85   0.90 / 0.72   0.29 / 0.31   0.28 / 0.31
              ATB0  7.57 / 5.35   1.73 / 3.28   2.48 / 6.14   0.29 / 0.33   0.34 / 0.66
              ACB0  2.92 / 0.99   0.81 / 1.14   0.54 / 0.69   0.24 / 0.47   0.25 / 0.36
    claims    A0B0  34.1 / 36.8   2.21 / 235    0.59 / 0.72   3.50 / 7.38   2.85 / 4.88
              A1B0  32.2 / 32.6   1.17 / 1.15   1.13 / 1.00   0.06 / 0.05   0.06 / 0.05
              ATB0  31.1 / 30.1   1.46 / 2.26   0.95 / 1.60   0.07 / 0.09   0.08 / 0.08
              ACB0  32.4 / 30.8   0.80 / 0.90   0.67 / 0.84   0.06 / 0.06   0.06 / 0.06

A0B0 puts the clean Q(0.99) about 40 times too high on every test set, so its "collapse"
under leverage is a fall towards the truth: 0.59 to 2.15 of it at a shift of 50, 2.2 to 12
at 20, and 235 once. On the claims every arm is 30 times too high, and Q2 scores each fall
towards the truth as a failure. Relative Q(0.99) does not measure a collapse when the clean
prediction is off by more than the collapse. Pinball does, and by pinball the picture is
the same on all three test sets and both seeds. One leverage row at a shift of 20 or 50
makes A0B0's pinball at 0.99 2.9 to 7.4 times worse than on its clean context. Under each
of the three modified priors it gets no worse, 0.05 to 0.97 of the clean pinball, in 35 of
the 36 arm-seed-shift-test-set cells of the table; the exception is ATB0-s93 at 50 on the
GPD (1.54). Under leverage, A1B0's Q(0.99) lands at 0.62 to 1.17 of the truth, ACB0's at
0.46 to 1.14, ATB0's at 0.95 to 6.14: the tail alone overshoots on every test set.

What carries over from the GPD, then, is the leverage response: a prior with heavy tails, a
contaminated context row or both keeps the pinball at 0.99 from degrading under one
leverage row, on tails of another body and on five claim tables nobody built to resemble
the priors. What does not carry over is the clean calibration. A1B0's clean Q(0.99) is 0.54
to 0.58 of the truth on the GPD, 0.94 and 2.86 on the families and 32 on the claims, as bad
as A0B0's there; its clean pinball is 0.19 and 0.22 of A0B0's on the GPD, 0.30 to 0.42 on
the families and 0.65 to 1.20 on the claims. The resemblance the ablation feared explains
the clean calibration, and on this evidence not the response to leverage.

Limits. Five tables are five units per seed, three families three; the 40 cells of one
table are not independent. The pinball ratios are medians over cells without a test, and
the empirical Q(0.99) of 1000 held-out claims is itself noisy. On the claims every arm is
calibrated so badly that a leverage row improves it, so "no worse under leverage" is a
weaker statement there than on the GPD. The models are small, the contexts are 200 rows,
and the clean miscalibration of every arm off the GPD says the 256-row stage did not teach
a usable tail for real claims.
```

## `experiments/h3_repair/pinball_reference.py`, read after the outcomes, 17.9.2026

```
READ AFTER THE OUTCOMES, 17.9.2026: PINBALL OVER A REFERENCE NO MODEL ENTERS
--------------------------------------------------------------------------
Not predicted. Median over cells of a model's pinball over the oracle's (generator) or over
the context quantile's (claims); 1 is the reference. On 900 or 1000 test rows 0.999 sees
about one exceedance, so that level reads mostly overshoot.

nanoTabPFN arms at 200 context rows, pre-training seed 92 / 93, at 0.99:

                   clean          shift 4        shift 20       shift 50
    GPD      A0B0  12.0 / 11.8    30.9 / 26.6    54.4 / 48.5    84.1 / 70.8
             A1B0  1.71 / 1.73    1.13 / 1.15    1.15 / 1.12    1.11 / 1.14
             ATB0  1.35 / 1.35    1.11 / 1.12    1.15 / 1.42    1.28 / 2.18
             ACB0  4.74 / 3.20    4.09 / 1.66    1.11 / 1.17    1.24 / 1.19
    families A0B0  11.7 / 11.7    28.9 / 27.5    58.8 / 53.8    85.5 / 72.5
             A1B0  4.19 / 4.20    1.20 / 1.16    1.12 / 1.10    1.08 / 1.13
             ATB0  4.02 / 4.01    1.09 / 1.10    1.20 / 1.43    1.28 / 2.28
             ACB0  4.57 / 3.19    3.15 / 1.53    1.10 / 1.22    1.22 / 1.18
    claims   A0B0  19.8 / 21.7    50.6 / 39.6    69.7 / 157     55.6 / 105
             A1B0  17.3 / 17.7    2.04 / 1.14    1.06 / 0.99    1.01 / 0.99
             ATB0  13.7 / 15.1    14.4 / 14.9    1.12 / 1.54    1.02 / 1.36
             ACB0  18.7 / 17.3    2.15 / 1.16    1.00 / 1.00    1.03 / 0.99

At 0.999 and a shift of 50: A0B0 188 to 205 on the generator, 298 and 320 on the claims;
A1B0 1.22 to 1.55 and 1.18 / 0.84; ACB0 1.53 to 2.01 and 0.94 / 0.81; ATB0 75 to 131 and
13.7 / 106. At a shift of 4 on the generator: ATB0 3.9 to 5.8, A1B0 3.8 to 7.7, ACB0 9.0
to 13.5, A0B0 17.1 to 17.8.

TabPFN-V3 at 2000 context rows (`clip_context.csv`, 20 seeds, xi 0.7 and 0.9), clean and
at shifts 4 / 20 / 50:

    raw        0.99  1.05   1.09 / 1.19 / 1.23     0.999  1.17   1.27 / 1.70 / 2.03
    sd_cap     0.99  1.05   1.06 at every shift     0.999  1.12   1.03 at every shift
    clip_200   0.99  1.04   1.04 at every shift     0.999  1.14   1.08 at every shift
    rank_gpd   0.99  1.02   1.01 at every shift     0.999  1.18   1.25 / 1.31 / 1.36
    rank_emp   0.99  1.02   1.04 at every shift     0.999  1.15   2.20 / 8.14 / 19.5

What the reference adds.

1. Against the oracle the small model's failure is larger than its own clean pinball
   showed: A0B0 is 12 times the oracle on a clean context and 71 to 86 times at a shift of
   50. Under a shift of 20 or 50 the modified priors sit at 1.1 to 1.4 (ATB0-s93 at 50,
   2.2 and 2.3), in most cells closer to the oracle than on their own clean contexts
   (1.35 to 4.74).
2. At 0.999 the ablation's order reverses at large shifts. The tail alone is 75 to 131
   times the oracle at 50, in A0B0's range, while both arms that carry the contamination
   stay within about 2. At a shift of 4 the tail alone and both together are close (3.8 to
   7.7) and the contamination alone is worst (9.0 to 13.5). Which treatment matters depends
   on the level and on the shift.
3. On the claims no arm beats the quantile of its clean context, a forecaster that ignores
   every feature: 14 to 22 times worse at 0.99 on a clean context, and at 0.5 and 0.9 within
   3% of it. Under a shift of 20 or 50, A1B0 and ACB0 match it (0.99 to 1.06). At 200 rows
   the small models add almost nothing to the marginal of these tables.
4. The production model is a different case. TabPFN-V3 is within 5% of the oracle at 0.99
   on a clean context, and one leverage row at a shift of 50 moves it to 1.23, and at 0.999
   from 1.17 to 2.03. In the same cells its implied xi falls from 0.61 and 0.75 to -0.30
   and -0.29, and the borders over the data from about 2550 to about 200. The mechanism
   measures show a change of tail family; the pinball shows what that costs a forecast, 17%
   more at 0.99 and 74% more at 0.999. Both belong in part two, and they are not
   interchangeable. The caps hold it at 1.03 to 1.08 at both levels under every shift. The
   rank transforms and log are the sharpest at 0.99 (1.01 to 1.04), and the empirical rank
   reaches 19.5 times the oracle at 0.999 at a shift of 50: the level dependence of part
   three, in these units.

The oracle is exact on the generator. The context quantile on the claims is a baseline and
not an oracle, and 1000 held-out claims give it noise of its own. Medians over cells, not
tests; the units are those of the sections above.
```

## Correction to the sd shift quoted for freMTPL2sev, 1.10.2026

```
Two predictions above quote 19.9 as the sd shift of freMTPL2sev, the worst of 99 public
datasets. That value is from the survey of 27.8.2026, which read the table one row per
claim: 26 639 rows, as OpenML serves it. Every run since the re-measurement of 11.9.2026
reads claims summed per policy and joined to the policy covariates, 24 944 rows. On that
table the worst sd shift is 14.5, still the largest of the 99 (Brazilian_houses follows at
6.0). The predictions are left as they were written; outcomes read against them use 14.5.
```

## Correction to the fine-tuning protocol, 1.10.2026

```
Both training loops cut every table of a batch at the smallest split in that batch
(`tabicl_prior_finetune.py:258`, `prior_pretraining_train.py:241`), while
`prior_arms.contaminate` picks the corrupted row below each table's own split. When a
table's own split is larger than the batch's, the corrupted row can land among the query
targets. That contradicts the statement in the TabICLv2 fine-tuning section that test
targets are never contaminated. The bucket edges of `prior_pretraining_train.py` read each
table at its own split and are not affected. FT-A0 has no contamination; the shorter
context is the intended effect of the batch cut there.

Counted without training, over the 20 000 tables each arm of the TabICLv2 fine-tuning saw
(batch 4, the starting offset of the seed, A1 defaults):

    seed   split changed   contaminated   corrupted row in the query
    1      2131            6086           104 (1.7%)
    2      1905            6018            91 (1.5%)

A corrupted query target sits 100 to 1500 context sds above the context mean; the clip
at 1e4 does not reach it. Where the row stays in the context, the shift on the final
context equals the drawn one on the median (largest ratio 14.5 for seed 1, 7.2 for seed 2).
The nanoTabPFN runs share the pattern; their counts were not made.

The outcomes of 16.9.2026 stand as measurements of those four checkpoints. They are not a
clean test of contamination confined to the context. How much the deviation moved them is
not measured. The pinball gradient with respect to a quantile is bounded, so the size of
these query targets in the loss says nothing about their weight in training.
```

## Correction to MEPS_2016_TOTEXP, 1.10.2026

```
pandas reads a SAS XPORT zero as 5.397605346934028e-79, so 6933 of the 34 655 persons in
MEPS HC-192 with no expenditure passed `positive_only` as positive amounts: 20% of the
table that `external_selection.py`, `unit_error_real.py` and the prevalence runs used.
`common/external.py` now sets every value below 1e-70 to zero; the converted table keeps
27 722 positive rows. The fingerprint in `data/dataset_fingerprints.json` was updated by
hand (y_sha1 6deebfe5085bc766 before, b6eb1afb17df43f4 after). Earlier MEPS rows are left
as they were measured and are not comparable with later ones.
```

## `experiments/h3_repair/unit_error_confirm.py`, 1.10.2026

```
The frozen cap clip_200 on subsamples no earlier run has seen (draw seed 20261001), the
five insurance tables, TabPFN-V3, TabICLv2 and GBM, 20 repeats. Three contexts per
subsample: clean, the earlier stress test (a row whose x100 error reaches an sd shift of
2) and a row drawn uniformly over the context, times 100.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
A unit is a (dataset, model) pair, 15 in all; a unit's value is its median over repeats.

C1, the stress test replicates. Under the visible error clip_200's pinball at 0.99 is
below raw's in at least 10 of 15 units (16 of 21 on the earlier draws).

C2, the clean cost stays small. On the clean context clip_200's pinball against raw is
within 2% at 0.99 in every unit, and at most 10% above it at 0.999 in every unit.

C3, a random error matters less. For each of the ten foundation-model units the rise of
raw's pinball at 0.99 over the clean context is smaller under the random row than under
the visible row.

Reported without a prediction: D0 and D1, the change in mean pinball at 0.99 from the cap
on clean contexts and under the random error, and the break-even rate p* = D0 / (D0 - D1)
where D0 > 0 > D1.

After the predictions were written, a smoke test of the code ran GBM on two repeats of
AutoClaims and norauto into a scratch directory. It showed that clip_200 touched no clean
context and neither corrupted row there. The predictions were not changed.
```

## `experiments/h3_repair/legit_segment.py`, 1.10.2026

```
The cost of the frozen cap to a legitimate high-risk segment, on the GPD generator at
xi = 0.7: a visible binary feature that multiplies the conditional scale by M in 5% of
rows, M in {1, 5, 20}, clean and under an added row at an sd shift of 20. The loss is the
expected pinball under the true conditional law, in closed form, so the 0.999 level
carries no Monte Carlo noise from the test targets. Seeds 7500 to 26500, unused before.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Expected pinball, paired over seeds, foundation models (TabPFN-V3, TabICLv2) unless said.

S1, the control. At M = 1 on a clean context clip_200 changes the expected pinball at 0.99
over the whole test set by less than 2% on the median for each foundation model.

S2, the benefit replicates. At M = 1 under the shift of 20 clip_200 lowers the expected
pinball at 0.99 in at least 18 of 20 seeds for each foundation model.

S3, the segment pays. At M = 20 on a clean context:
  (a) at least half of the clipped context rows are segment rows, on the median;
  (b) the segment's expected pinball at 0.999 is higher under clip_200 than under raw in at
      least 14 of 20 seeds for each foundation model;
  (c) for each foundation model the median relative change of the segment's expected
      pinball at 0.99 is larger than that of the whole test set.

S4, the benefit survives the segment. At M = 20 under the shift of 20 clip_200 lowers the
expected pinball at 0.99 over the whole test set in at least 15 of 20 seeds for TabPFN-V3.

GBM carries no prediction, nor does M = 5. The closed form was checked against 4 million
Monte Carlo draws, and a smoke test ran GBM on one seed into a scratch directory, both
after the predictions were written.
```

## `experiments/h3_repair/legit_segment.py`, outcome, 1.10.2026

```
720 cells, none failed, 6584 s on CPU beside the unit-error run (6 threads each).
Medians over 20 seeds of clip_200 against raw, expected pinball.

S1  HOLDS. At M = 1 on a clean context the cap changes the expected pinball at 0.99 by
    -0.17% (TabPFN-V3) and -0.31% (TabICLv2).

S2  HALF. TabPFN-V3 is lower in 20 of 20 seeds (-9.6%). TabICLv2 is lower in 12 of 20
    (-2.2%), and at 0.999 the cap costs it 7.4%.

S3  HOLDS. (a) 3 of the 4 clipped context rows are segment rows on the median (share
    0.82). (b) The segment's expected pinball at 0.999 rises in 18 of 20 seeds for
    TabPFN-V3 (+5.3%) and in 20 of 20 for TabICLv2 (+15.9%). (c) At 0.99 the segment pays
    more than the whole test set: +0.45% against -0.01% for TabPFN-V3, +10.0% against
    +5.7% for TabICLv2.

S4  HOLDS. 20 of 20 seeds for TabPFN-V3, -9.4% over the whole test set.

What this says. The cost of the cap falls on the segment as predicted. Its size, and
whether the cap pays at all, depends on what the added row does to the raw model:

    raw, shift 20 against clean    TabPFN-V3        TabICLv2         GBM
    0.99,  M = 1                   +10.0%           +2.5%            0.0%
    0.999, M = 1                   +4.0%            -9.3%            +23.4%
    0.999, M = 20                  +22.3%           -28.5%           +1.0%

The row hurts TabPFN-V3, so the cap repairs it: -9.4% at 0.99 at M = 20 under the error,
-6.0% and -3.6% on the segment at 0.99 and 0.999, for a clean-context cost of +0.45% and
+5.3% on the segment. The row helps TabICLv2 at 0.999, because its clean Q(0.999) sits at
0.42 of the truth (0.12 on the segment at M = 20) and the inflated context lifts it. The cap
removes the lift: at M = 20 under the error it raises TabICLv2's expected pinball by 9.6% at
0.99 and 54% at 0.999 over the whole test set, by 21.6% and 72.6% on the segment. That
the error helps TabICLv2 is a consequence of its clean underestimate, not a robustness.
GBM moves by no more than 2.6% at 0.99; at 0.999 the cap gains under the error at M = 1 and 5
and costs 5.6% on a clean context at M = 20.
```

## `experiments/h3_repair/unit_error_confirm.py`, outcome, 1.10.2026

```
1800 cells, none failed, 6952 s on CPU beside the segment run (6 threads each).

C1  FAILS. clip_200 is below raw under the visible error in 7 of 15 units, not 10. The
    count splits by model: TabPFN-V3 in 4 of 5 (freMTPL2sev +0.07%), TabICLv2 in 1 of 5
    (freMTPL2sev -2.4%; elsewhere +1.0% to +5.5%), GBM in 2 of 5 and unchanged in 3. The
    earlier 16 of 21 was the same split plus BlogFeedback and MEPS, where the cap helped
    every model. On the five insurance tables of the earlier draws TabPFN-V3 was lower in
    5 of 5 and TabICLv2 in 2 of 5. The prediction was set on the pooled count.

C2  HOLDS. On a clean context the cap touches nothing on four tables, so its pinball equals
    raw's. On freMTPL2sev it touches 85% of clean contexts and lowers the pinball at 0.99 by
    0.07% to 0.6% on the median, at +0.2% and +0.9% at 0.999.

C3  FAILS. The random row raises raw's pinball less than the visible row in 6 of 10
    foundation-model units. TabPFN-V3 follows the prediction except on freMTPL2sev, where
    neither error moves it. TabICLv2 does not: a random x100 row raises its pinball at 0.99
    about as much as a visible one (norauto 5.9% against 5.7% by mean, ausprivauto0405
    3.1% against -0.1%).

Two regimes. D0 is zero on four tables because the cap never fires there, and negative on
freMTPL2sev for all three models. No unit has D0 > 0, so the break-even rate p* is
undefined everywhere: on these tables at 0.99 clip_200 costs nothing on a clean context
and the only question is whether it helps under an error. Under the random error the cap
fires in 15 to 35% of contexts on the four small-claim tables, leaving the corrupted value
in place in the rest, and in every context on freMTPL2sev. Under the visible error it fires
in 75 to 100%.

Where it fires on a visible error, the cap lowers the pinball at 0.99 for TabPFN-V3 in 70
of 90 contexts (median -1.5%) and raises it for TabICLv2 (lower in 33 of 90, median
+1.4%). The generator of `legit_segment.py` gives the same split: the cap repairs
TabPFN-V3 and costs TabICLv2. A recommendation for clip_200 is a recommendation for
TabPFN-V3, not for the family.
```

## `experiments/h3_repair/recalibration.py`, 2.10.2026

```
One global recalibration on 2000 clean held-out rows, beside the frozen cap, on the design
of legit_segment.py (xi = 0.7, segment of 5% at scale M in {1, 20}, clean and at an sd
shift of 20), TabPFN-V3, TabICLv2 and GBM, seeds 7700 to 26700. Primary: one factor c on
the quantile, c the q-weighted tau-quantile of y / q. Secondary: one shift, the tau-quantile
of y - q. Both are exact pinball minimisers on the calibration rows and share every fit.
The calibration rows are new information; the result measures what is repairable when such
a sample exists. Replaces the optional fine-tuning branch (decision of 2.10.2026).

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Expected pinball at 0.99, multiplicative calibration unless said, paired over 20 seeds.
Differences under 2% are calibration noise at this sample size and are not read as effects.

K1. For TabPFN-V3 and TabICLv2, raw + mul is lower than raw over the whole test set in at
    least 15 of 20 seeds in each of the four (M, shift) cells.
K2. At M = 1 under the shift of 20, cap + mul is lower than raw + mul for TabPFN-V3 in at
    least 14 of 20 seeds, and the median relative gap between them is smaller than the gap
    between cap and raw without calibration.
K3. At M = 20 on a clean context, after raw + mul the segment's excess over the oracle is
    larger than the excess outside the segment in at least 15 of 20 seeds for TabICLv2.
K4. At M = 20 on a clean context, cap + mul gives TabICLv2's segment a higher expected
    pinball than raw + mul in at least 14 of 20 seeds.
K5. At M = 20 on a clean context the segment's excess over the oracle after raw + add is
    larger than after raw + mul in at least 15 of 20 seeds for each foundation model.

GBM, TabPFN-V3 in K3 and K4 and the 0.999 level carry no prediction. After the predictions
were written a smoke test ran GBM on one seed into a scratch directory; at M = 20 its
multiplicative factor fell to 0.46 and raised the loss over the whole test set. The
predictions were not changed.
```

## `experiments/h3_repair/unit_error_confirm.py` with CATEGORICAL=native, 13.10.2026

```
The confirmation of 1.10.2026 handed categorical columns to every model as integer codes.
TabICL 2.1.1 encodes categories with its own OrdinalEncoder, so codes are what it would see
anyway, and trees split on codes either way; only TabPFN has a categorical path
(`categorical_features_indices`). This run repeats TabPFN-V3 on the same draws (draw seed
20261001, same subsamples, same rows corrupted) with the coded columns passed as
categories, into unit_error_confirm_native.csv. On freMTPL2sev alone the natural-extreme
run of 12.9.2026 (clip_context_real_native.csv, read only on 13.10.2026) already showed
the cap lowering TabPFN-V3's pinball at 0.99 in 9 of 10 subsamples either way.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
A unit is a table; its value is the median over 20 repeats, as in C1.

N1, the benefit does not depend on the encoding. Under the visible error clip_200's
pinball at 0.99 is below raw's for TabPFN-V3 in at least 4 of 5 tables, as with codes.

N2, the encoding moves the raw model little. On the clean context TabPFN-V3's raw pinball
at 0.99 with categories is within 3% of the coded run's, on the median of the paired
ratio over repeats, in every table.

TabICLv2 and GBM are not rerun. No smoke test was run before these were written.
```

## `experiments/h3_repair/recalibration.py`, outcome, 13.10.2026

```
480 cells, none failed. 122 cells ran on 2.10.2026, the other 358 on 13.10.2026; seed 7700
of TabPFN-V3 rerun on 13.10 reproduced the stored rows bit for bit in all four cells.

K1  FAILS. raw + mul is lower than raw in 17 and 19 of 20 seeds for TabPFN-V3 at M = 1,
    9 and 14 at M = 20; TabICLv2 12, 12, 12 and 9. The correction helps TabPFN-V3 without a
    segment and is noise for TabICLv2.
K2  HOLDS. At M = 1 under the shift of 20 cap + mul beats raw + mul in 20 of 20 seeds;
    the median gap is -3.2% against -6.7% without calibration.
K3  HOLDS, narrowly. 16 of 20 seeds for TabICLv2.
K4  HOLDS. cap + mul raises TabICLv2's segment loss over raw + mul in 17 of 20 seeds.
K5  FAILS. 14 of 20 seeds for TabICLv2 and 10 of 20 for TabPFN-V3.

Calibration takes half of the cap's benefit to TabPFN-V3 and removes none of the cap's
cost to the segment. GBM's factor raises its loss as often as it lowers it.
```

## `experiments/h3_repair/unit_error_confirm.py` with CATEGORICAL=native, outcome, 13.10.2026

```
600 cells, none failed; same subsamples, corrupted rows and sd shifts as the coded run.

N1  HOLDS. Under the visible error the cap lowers TabPFN-V3's pinball at 0.99 in 5 of 5
    tables (-0.2% to -4.0%), against 4 of 5 with codes; freMTPL2sev moves from +0.07% to
    -0.18%.
N2  HOLDS. The clean raw pinball at 0.99 moves by -0.5% to +0.4% against the coded run.
    On beMTPL97 every cell is identical to the last digit, which suggests TabPFN's own
    modality detection already treats those columns as categorical.

The encoding does not carry the result of the confirmation.
```

## `unit_error_confirm.py` and `legit_segment.py` with the default ensemble and a log arm, 20.10.2026

```
Two gaps in the confirmation of 1.10.2026, both raised by a review of the thesis plan.
Every H2 and H3 run fits TabPFN-V3 with one ensemble member. The package default is
eight: in tabpfn 8.4 with this checkpoint four of the eight members fit a Yeo-Johnson
("safepower") transform of the target before standardising it, while one member fits
none. That was checked on 20.10.2026 by fitting both settings on random data and reading
the members' target transforms. The vendor's insurance cookbook fits claim amounts on
log(1 + y) and caps them at 200 000 before the split, so the plain rival of clip_200 is a
log of the target, which no confirmation has measured.

Same draws as 1.10.2026: draw seed 20261001 on the five insurance tables, seeds 7500 to
26500 on the segment generator, categorical columns as codes. Arms raw, clip_200 and log
(fit on log y, exp of the quantiles), selected with ARMS=raw,clip,log. TabPFN-V3 runs on
CPU at 6 threads, as before. Four outputs:

    unit_error_confirm_log.csv      TabPFN-V3, TabICLv2, GBM    one member
    unit_error_confirm_nest8.csv    TabPFN-V3                   eight members
    legit_segment_log.csv           TabPFN-V3, TabICLv2, GBM    one member
    legit_segment_nest8.csv         TabPFN-V3                   eight members

The one-member files repeat raw and clip_200, which should match the stored rows of
1.10.2026 bit for bit (R0, a check rather than a prediction). If they do not, every
comparison below stays within the new files. The cap's one-member rows on these draws
were seen on 1.10.2026; what is new is the log arm and the ensemble.

Read before the predictions were written, from development runs at one member.
clip_context.csv (TabPFN-V3, xi = 0.7, 20 seeds, drawn test targets): log against raw at
0.99 is -4.0% on a clean context (lower in 19 of 20 seeds) and -13.6% at a shift of 20
(20 of 20); clip_200 gives -0.1% and -10.4%. clip_context_real.csv (freMTPL2sev, natural
extremes, 5 subsamples per bin): at a shift of 4 or more log gives -3.7% for TabPFN-V3
and -5.3% for TabICLv2, clip_200 -2.5% and -6.0%; below a shift of 1.2 log gives -0.9%
and -2.5%, clip_200 0.0% and -0.5%. No run has measured log under the unit error or on
the segment, nor eight members under contamination.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Units as on 1.10.2026. Unit error: a (table, model) pair, its value the median over 20
repeats of the paired relative change in pinball at 0.99. Segment: expected pinball at
0.99 over the whole test set unless said, paired over 20 seeds. The verdicts come from
analysis/confirm_ensemble_log.py, written with these predictions.

The default ensemble, TabPFN-V3 at eight members.

E1, the sensitivity survives. Segment, M = 1: raw's expected pinball under the shift of
    20 is above raw's on the clean context in at least 15 of 20 seeds, with a median rise
    of at least 2% (+10.0% at one member).
E2, the ensemble damps it. That rise is smaller at eight members than at one on the same
    seed in at least 14 of 20 seeds: the four transformed members see a compressed row,
    the other four see it as one member does.
E3, the cap still repairs TabPFN-V3. (a) Segment, M = 1, shift 20: clip_200 is below raw
    in at least 18 of 20 seeds. (b) Unit error, visible row: clip_200 is below raw in at
    least 4 of 5 tables.

The log arm, one member.

L1, log helps on a clean context. Segment, M = 1, clean: log is below raw in at least 15
    of 20 seeds for each foundation model.
L2, log repairs the error for TabPFN-V3. Segment, M = 1, shift 20: log is below raw in at
    least 18 of 20 seeds.
L3, log repairs TabICLv2 where the cap did not. Unit error, visible row: log is below raw
    for TabICLv2 in at least 3 of 5 tables (clip_200: 1 of 5).
L4, log beats the cap. (a) Unit error, visible row: log is below clip_200 in at least 3
    of 5 tables for each foundation model. (b) Segment, M = 20, clean: the segment's
    expected pinball is lower under log than under clip_200 in at least 14 of 20 seeds
    for each foundation model, since log truncates no segment row.

Reported without a prediction: GBM, the 0.999 level, M = 5, the random row, log at eight
members, log against raw on the unit error's clean context and D0 and D1 for log.

No model was run on the data of this experiment before these were written. The changed
reports of both scripts were rerun on the stored files of 1.10.2026 and reproduce the
counts of C1 to C3. The verdict script was run once on stand-ins built from those files,
with log rows copied from clip rows, which tests the code and nothing else.
```

## `unit_error_confirm.py` and `legit_segment.py` with the default ensemble and a log arm, outcome, 21.10.2026

```
5040 cells, none failed, on CPU at 6 threads in two parallel streams. Unit error 11196 s
at one member and 47230 s at eight; segment 9288 s and 47463 s.

R0  The raw and clip_200 rows at one member equal the stored rows of 1.10.2026 bit for bit
    (1800 unit-error rows, 720 segment rows).

E1  HOLDS, narrowly. At eight members the shift of 20 raises TabPFN-V3's expected pinball
    at 0.99 in 19 of 20 seeds, by +2.3% on the median (+10.0% at one member).
E2  HOLDS. The rise is smaller at eight members in 20 of 20 seeds.
E3  HOLDS. (a) clip_200 is below raw in 20 of 20 seeds, by -2.6% on the median against
    -9.6% at one member. (b) Under the visible unit error it is below raw in 5 of 5
    tables (-0.5% to -5.1%), against 4 of 5 at one member.
L1  HOLDS. On a clean context log is below raw in 20 of 20 seeds for TabPFN-V3 (-2.2%)
    and in 18 of 20 for TabICLv2 (-2.1%).
L2  HOLDS. 20 of 20 seeds, -11.3%.
L3  HOLDS. Under the visible error log is below raw for TabICLv2 in 5 of 5 tables (-1.0%
    to -7.3%), where clip_200 was below raw in 1 of 5.
L4  HOLDS. (a) Log is below clip_200 in 5 of 5 tables for TabPFN-V3 and in 4 of 5 for
    TabICLv2 (freMTPL2sev +0.3%). (b) On the segment at M = 20 log is lower in 15 of 20
    seeds for TabPFN-V3 (-2.2%) and in 20 of 20 for TabICLv2 (-11.8%).

Not predicted, read after the verdicts.

The ensemble damps the added row on the generator, not the unit error on the insurance
tables. Raw's rise at 0.99 under the visible x100 row against the clean context, TabPFN-V3
at one member and at eight: ausprivauto0405 +3.2% and +8.1%, beMTPL97 +6.4% and +7.8%,
AutoClaims +2.2% and +2.9%, norauto +1.7% and +1.5%, freMTPL2sev -0.3% and +0.2%. E2 was
written for the segment design only and says nothing about these tables.

Eight members lower TabPFN-V3's clean upper quantiles. On the segment design at M = 1 its
predicted Q(0.999) outside the segment is 0.35 of the truth at eight members against 0.52
at one, the flattening by ensemble averaging seen in H1. Log lifts it to 0.63 at either
size.

Log costs TabPFN-V3 the segment's extreme quantile. At M = 20 its predicted Q(0.999) in
the segment is 1.9 times the truth under log at one member and 2.0 at eight, against 0.55
and 0.35 under raw. The segment's expected pinball at 0.999 then rises over raw by +10% on
a clean context and +59% under the shift of 20 at one member, by +0.2% and +43% at eight.
TabICLv2 moves the other way: log lifts its too-low Q(0.999) outside the segment from 0.42
to 0.69 of the truth at M = 1 and lowers its expected pinball at 0.999 in every clean cell,
by 14% to 24% (20 of 20 seeds each). At M = 20 under the shift TabICLv2's loss at 0.99 is
higher under log than under raw (+3.0%, lower in 5 of 20 seeds), though still 7.2% below
clip_200.

On the insurance tables log is not free at 0.999 for TabPFN-V3 either. On a clean context
it costs +14.6% on ausprivauto0405 and +10.7% on beMTPL97 at one member; under the visible
error ausprivauto0405 costs +40% at one member and +75% at eight. For TabICLv2 log is
lower at 0.999 on every clean table (-9.5% to -16.2%).

D0 and D1 for log, from mean pinball at 0.99 within each unit. Medians over repeats favour
log on a clean context in all ten foundation-model units at one member, but means do not:
D0 > 0 for TabPFN-V3 on freMTPL2sev (2.0) and norauto (0.8; 7.7 at eight members), for
TabICLv2 on beMTPL97 (0.7) and for GBM on ausprivauto0405, beMTPL97 and norauto. Where
D0 > 0 > D1 the break-even error rate p* is 0.33 for TabPFN-V3 on freMTPL2sev, 0.23 on
norauto, 0.24 for TabICLv2 on beMTPL97, 0.23 and 0.99 for GBM on norauto and
ausprivauto0405. On norauto at eight members and on beMTPL97 for GBM log costs under the
random error too (D1 > 0).

GBM moves little at 0.99. Log changes its expected pinball on the segment design by -0.5%
to -1.8% and clip_200 by -0.6% to +2.5%; under the visible unit error log is below raw in
5 of 5 tables (-0.1% to -4.5%).

A caveat on one column. n_clipped_seg in legit_segment_log.csv and legit_segment_nest8.csv
counts the segment's context rows whose fitted value is below the raw one. For clip_200
those are the clipped rows; for log they are every segment row above 1, though log clips
none. No verdict reads the column on log rows. The code was fixed on 21.10.2026 after both
runs had finished, so the stored log rows still carry the old count.

What this says. One member overstated how much TabPFN-V3 needs the cap: at the package
default the added row costs it about a quarter as much on the generator and the cap's
benefit shrinks with it, though on the insurance tables the visible unit error costs the
ensemble at least as much as one member. A log of the target, the vendor's advice for
claim amounts, beats clip_200 at 0.99 for both foundation models on these draws and helps
TabICLv2 under a unit error where the cap did not. It is not free: it changes every clean
prediction, its clean cost by mean loss is positive on several tables and it doubles
TabPFN-V3's extreme quantile of a legitimate high-scale segment. These are the draws of
1.10.2026 and log was picked after development runs, so this replicates those runs with
predictions written first; it is not a test on a new domain.

Corrected the same day, two phrases of the paragraph above. "At least as much as one
member" holds on 4 of 5 tables: on norauto eight members rise by +1.5% against +1.7%.
"Doubles" means that log puts TabPFN-V3's Q(0.999) in the segment at 1.9 to 2.0 times the
truth, 3.4 and 5.7 times its raw value at one and eight members.
```

## `unit_error_confirm.py` and `legit_segment.py` on TabPFN-3.5, 21.10.2026

```
TabPFN-3.5 (`TabPFN-v3.5`, tabpfn 9.0.0) is the package default since 15.9.2026. It
handles the target as V3 does: checked on 21.10.2026 in venv-tfm9 by fitting it on random
data, REGRESSION_Y_PREPROCESS_TRANSFORMS is (None, "safepower"), eight members give four
a Yeo-Johnson target and one member gives none. The runs use venv-tfm9cu, the packages of
venv-tfm9 without causilo and with torch 2.9.1+cu126 in place of 2.14.0+cpu. The question is whether the outcome of
21.10.2026 for V3 carries over to the model a user now gets by default.

Same draws as 1.10.2026 and 21.10.2026, arms raw, clip_200 and log, one and eight
members. No TabPFN-3.5 row exists on these draws or anywhere in results/. It runs on the
RTX 2060 (TABPFN_DEVICE=cuda), every earlier TabPFN row on CPU, so its numbers are
compared with V3's only through the verdicts below, never row by row. Four outputs:

    unit_error_confirm_tabpfn35_nest1.csv    legit_segment_tabpfn35_nest1.csv
    unit_error_confirm_tabpfn35_nest8.csv    legit_segment_tabpfn35_nest8.csv

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Units and losses as in the section of 20.10.2026. Each prediction says that 3.5 behaves
as V3 did; the V3 value is in brackets.

F1, one member is sensitive. Segment, M = 1: raw's expected pinball at 0.99 under the
    shift of 20 is above raw's on the clean context in at least 15 of 20 seeds at one
    member, with a median rise of at least 5% (+10.0%).
F2, the ensemble damps it. That rise is smaller at eight members than at one on the same
    seed in at least 14 of 20 seeds (20 of 20).
F3, the cap still helps at eight members. (a) Segment, M = 1, shift 20: clip_200 is below
    raw in at least 15 of 20 seeds (20 of 20, -2.6%). (b) Unit error, visible row:
    clip_200 is below raw in at least 4 of 5 tables (5 of 5).
F4, log beats the cap at eight members. (a) Unit error, visible row: log is below
    clip_200 in at least 4 of 5 tables (5 of 5). (b) Segment, M = 1, clean: log is below
    raw in at least 15 of 20 seeds (20 of 20, -5.8%).
F5, log overshoots the segment. Segment, M = 20, clean, eight members: under log the
    median over seeds of the predicted over the true Q(0.999) in the segment is above
    1.25 (1.96).

Reported without a prediction: the unit error at one member, the 0.999 level, M = 5, the
random row and the time per cell on the GPU.

Before these were written TabPFN-3.5 was fitted only on random data, to read its
configuration. The verdict script analysis/confirm_tabpfn35.py was written with them and
run once on stand-ins, V3's rows of 21.10.2026 renamed, where it returns the bracketed
values; that tests the code and nothing else. The
run script first fits 3.5 at eight members on random data of the run's size on the GPU,
a check of memory and nothing else, and stops if it fails.
```

## `unit_error_confirm.py` and `legit_segment.py` on TabPFN-3.5, outcome, 21.10.2026

```
2520 cells, none failed, 8220 s on the RTX 2060 in one stream; the memory check took 14 s
and 2040 MB at eight members and 5000 test rows.

F1  FAILS. At one member the shift of 20 raises 3.5's expected pinball at 0.99 in 18 of
    20 seeds, but by +2.1% on the median, not 5% (V3: +10.0%).
F2  HOLDS. The rise is smaller at eight members in 19 of 20 seeds; at eight it is gone,
    -0.7% on the median.
F3  HALF. (a) FAILS: at eight members clip_200 is below raw in 6 of 20 seeds (+0.4%),
    since the row no longer costs anything there. (b) HOLDS: under the visible unit error
    it is below raw in 5 of 5 tables, by -0.05% to -1.85%.
F4  HOLDS. (a) Log is below clip_200 in 4 of 5 tables (ausprivauto0405 +0.4%). (b) On a
    clean context log is below raw in 19 of 20 seeds, -2.5%.
F5  HOLDS. Under log 3.5 puts Q(0.999) in the segment at 1.92 times the truth (V3: 1.96).

Not predicted, read after the verdicts.

At one member 3.5 places its clean tail far closer to the truth than V3: Q(0.999) outside
the segment at M = 1 is 0.85 of the truth, against 0.52 for V3. The added row then costs
little at 0.99 and a lot at 0.999: +49% over the whole test set and +153% on the segment
at M = 1. clip_200 takes it back, -33% at 0.999 in 20 of 20 seeds and -2.0% at 0.99.

At eight members the ensemble flattens the clean tail again (Q(0.999) outside the segment
0.55 of the truth) and the added row costs nothing over the whole test set, -0.7% at 0.99
and -1.6% at 0.999, though the segment's 0.999 loss still rises 15% at M = 1. With nothing
to repair the cap only costs: at M = 20 on a clean context +5.4% at 0.99 and +13% at 0.999
over the whole test set, +11% and +27% on the segment.

Under the unit error the visible row raises raw's loss at 0.99 by +0.4% to +3.3% at one
member and +0.2% to +3.7% at eight (largest on ausprivauto0405 and on beMTPL97), against
+0.2% to +8.1% for V3 at eight members. At one member clip_200 lowers it in 3 of 5
tables (-0.5% to -2.1%), leaving AutoClaims and norauto unchanged on the median.

Log costs 3.5 at 0.999 as it cost V3. On the segment at M = 20 its expected pinball at
0.999 rises over raw by +29% (one member) and +20% (eight) on a clean context, +97% and
+164% under the shift. On beMTPL97 the clean loss at 0.999 rises by 31% and 35%. By mean
loss at 0.99 log costs on a clean context on ausprivauto0405 and beMTPL97 at both sizes
and on AutoClaims at eight, barely; on ausprivauto0405 at both sizes and AutoClaims at
eight it costs under the random error too. At one member log is below clip_200 under the
visible error in 4 of 5 tables (ausprivauto0405 +1.7%).

What this says. The package default no longer shows the one-row sensitivity on the
generator at 0.99 or 0.999; there the cap has nothing to repair and only costs. The
sensitivity measured in this thesis is a property of a configuration: large for V3 at one
member, smaller at eight, small at 0.99 but large at 0.999 for 3.5 at one member, absent
over the whole test set for 3.5 at eight. Under a unit error on the insurance tables
every configuration still loses up to a few percent and the cap recovers part of it.
There log is below the cap at 0.99 in 4 or 5 of 5 tables for every TabPFN configuration
measured, by small margins for 3.5; it overshoots a legitimate high-scale segment at
0.999 in each. 3.5 ran on the GPU, so its rows are compared with V3's only through the
verdicts and medians above, never row by row.
```

## `experiments/h3_repair/clip_context.py`, the dose curve at the default ensemble, 22.10.2026

```
The headline of part two, +83% in pinball at 0.999 under a shift of 50, is TabPFN-V3 at
one member on 900 drawn test targets. The outcomes of 21.10.2026 show that the sensitivity
depends on the configuration; the drawn loss at 0.999 rests on about one exceedance per
task. This run repeats the dose curve of clip_context.py (xi 0.7 and 0.9, 20 seeds,
shifts 1, 4, 20 and 50, 2000 context and 900 test rows) with raw, clip_200 and log only,
for TabPFN-V3 on CPU in venv-tfm and TabPFN-3.5 on the GPU in venv-tfm9cu, each at one
and eight members. clip_context.py now also writes the expected pinball under the true
conditional GPD in closed form (epb, as in legit_segment.py); the predictions are on it.

    clip_context_dose_v3_nest1.csv          clip_context_dose_tabpfn35_nest1.csv
    clip_context_dose_v3_nest8.csv          clip_context_dose_tabpfn35_nest8.csv

The stored clip_context.csv of 12.9.2026 does not reproduce exactly under the current
code: a smoke test of one task (xi 0.7, seed 7000, shifts 1 and 50) differed from its
stored drawn pinball by 0.1% to 4.6% at 0.99. The new one-member file is therefore the
reference; no comparison crosses into the stored file.

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
A task is one (xi, seed) pair, 40 in all. Expected pinball, paired within a task.

D1, the headline survives the expected loss. V3 at one member, shift 50: raw's expected
    pinball at 0.999 is above raw's on the clean context in at least 32 of 40 tasks, with
    a median rise of at least 20%.
D2, the ensemble damps V3. At shifts 20 and 50 separately, raw's rise at 0.99 over the
    clean context is smaller at eight members than at one in at least 30 of 40 tasks.
D3, 3.5 at its default is flat at 0.99. At eight members the median rise of raw's
    expected pinball at 0.99 over the clean context lies within 3% of zero at shifts 4 and
    20, on each xi separately (four medians).
D4, log beats the cap at the default. At shift 20 log's expected pinball at 0.99 is below
    clip_200's in at least 28 of 40 tasks, for V3 at eight members and for 3.5 at eight.
D5, the cap repairs V3's default where the row hurts. V3 at eight members, shift 50:
    clip_200's expected pinball at 0.999 is below raw's in at least 30 of 40 tasks.

Reported without a prediction: 3.5 at one member, the drawn pinball, borders over the
data, implied xi and every cell not named above. After the dose curve the GPU part runs
the measurements of run/new_models.sh for 3.5 at one member (natural extremes on
freMTPL2sev, the one-row dose on the maximum and on the sd, shape and scale on the
generator, the unit error on seven tables), descriptive as they were for LimiX-2 and
Causilo.

Seen before these were written: the smoke tests above, on one task. V3 at one member
gave an expected-pinball rise at shift 50 of +25% at 0.99 and +34% at 0.999. V3 at eight
members was timed on one cell, raw at shift 50, without reading its loss. 3.5 at eight
members ran raw, clip_200 and log at shifts 1 and 50 for that task; its drawn pinball at
0.999 was 0.13 clean and 0.33 at shift 50, the expected pinball was not read. The verdict
script analysis/dose_ensemble.py was written with these predictions and run once on
stand-ins built from the one-member smoke file, which tests the code and nothing else.
```

## `experiments/h3_repair/clip_context.py`, the dose curve at the default ensemble, outcome, 22.10.2026

```
1920 cells, none failed. V3 on CPU: 2448 s at one member and 15243 s at eight; 3.5 on
the GPU: 1979 s and 2824 s. The one-member V3 rows of the smoke task equal the run's to
the last digit.

D1  FAILS, narrowly. At one member and a shift of 50 raw's expected pinball at 0.999
    rises in 36 of 40 tasks, by +18.7% on the median, not 20%. On the drawn test targets
    of the same run the rise is +83.4%, the headline of part two reproduced; the expected
    loss puts it at about a fifth of that.
D2  HOLDS. The rise at 0.99 is smaller at eight members in 40 of 40 tasks at a shift of
    20 (+6.0% against +0.6%) and of 50 (+6.8% against +0.7%).
D3  HOLDS. 3.5 at eight members moves raw's expected pinball at 0.99 by +1.0% and -0.1%
    at a shift of 4 and by -0.7% and -1.0% at 20 (xi 0.7 and 0.9).
D4  HOLDS. At a shift of 20 log is below clip_200 at 0.99 in 40 of 40 tasks for V3 at
    eight members (-4.2%) and in 33 of 40 for 3.5 at eight (-1.3%).
D5  HOLDS. For V3 at eight members and a shift of 50 clip_200 is below raw at 0.999 in
    35 of 40 tasks, -6.3%.

Not predicted, read after the verdicts.

Raw's median rise over the clean context in %, expected pinball, shift 4 / 20 / 50:

               V3, 1 member     V3, 8          3.5, 1          3.5, 8
    0.99       1.5/ 6.0/ 6.8    0.6/0.6/0.7    0.2/ 1.4/ 1.8   0.2/-0.9/-1.2
    0.999      1.4/ 5.5/18.7    2.4/3.9/6.0    4.6/45 /132     -0.3/3.6/8.2

3.5 at one member is the most sensitive configuration at 0.999 and among the least at
0.99; at eight members it is the least sensitive at both. The median level moves by at
most +7.6% (V3, one member, shift 50).

The cap at 0.99 under a shift of 50: -6.4% for V3 at one member (40 of 40), -1.1% at
eight (27 of 40), -1.8% for 3.5 at one (39 of 40) and +1.0% at eight (3 of 40 lower):
where the row costs nothing the cap costs a little. At 0.999 the cap gains in every
configuration at a shift of 50, from -6.3% (V3, eight) to -59% (3.5, one).

Log at 0.99 is below raw in every configuration and shift, by -0.3% to -7.5% on the
median; on a clean context by -0.3% (3.5, one) to -4.2% (V3, eight). At 0.999 under
a shift log is below clip_200 for V3 (-2.9% to -7.0%) but above it for 3.5 (+3.3% to
+7.6%, lower in 9 to 17 of 40 tasks).

The measurements of run/new_models.sh for 3.5 at one member, read in a copy of
analysis/models_2026_09.ipynb (descriptive, as for LimiX-2 and Causilo): keeping the
largest claim of freMTPL2sev puts Q(0.99) at 0.92 of its value without it (V3: 0.92);
clip_200 lowers the 0.99 loss in 9 of 10 subsamples (-4.3%) and the 0.999 loss in 7 of
10 (-25%); dosed on the sd, its local Q(0.99) moves by at most 11% and its implied xi at
a true 0.3 stays between 0.31 and 0.38 (V3 falls to -0.17); shape against scale gives
93% and 47%, 20 of 20 seeds; the unit error on seven tables costs +2.1% at 0.99, +1.9%
capped.

What this says. The headline of +83% is mostly the noise of one exceedance per task at
0.999: on the expected loss one added row costs V3 at one member +19% at 0.999 and +7% at
0.99. The default ensemble takes most of that away. The sensitivity is real but its
size depends on the configuration, from +132% at 0.999 for 3.5 at one member to nothing
at 0.99 for 3.5 at eight. Log is the better repair at 0.99 everywhere measured here; at
0.999 the cap does better for 3.5.
```

## Correction to the expected pinball of the generator, 22.10.2026

```
The expected pinball (epb*) of clip_context.py, legit_segment.py and recalibration.py is
computed under the ideal GPD. The generator caps the uniform draw at 1 - 1e-9, so its
law puts the mass above the cap point u = Q(1 - 1e-9) on u itself. For q < u, which holds
for every quantile measured, the two laws differ by a constant: the ideal expected
pinball is larger by tau * delta, delta = s * 1e-9^(1 - xi) / (1 - xi). At xi = 0.7
delta is 0.66% of s and moves no reported number by more than 0.6 points. At xi = 0.9 it
is 1.26 s, an unobservable 12.6% of the mean, and it shrinks every relative change on
those tasks.

Recomputed under the generator's own law (no model rerun, scales from the generator
seeds): D1 becomes +23.0% (36 of 40 tasks) instead of +18.7%, so D1 holds under the law
the tasks were drawn from and fails, as recorded, under the ideal law it was written for.
V3 at one member, shift 50: +8.2% at 0.99 and +23.0% at 0.999 (ideal +6.8% and +18.7%);
at eight members +0.8% and +6.9% (ideal +0.7% and +6.0%); 3.5 at one member +154% at
0.999 (ideal +132%). D2 to D5 do not change verdict. The verdicts above stay as written.
From here on the generator's law is the reference for these files.
```

## `experiments/h1_shape_vs_scale/dissociation_real_slope.py` across the Hill share, 22.10.2026

```
The real-data slope of part one (shape share 0.415 for TabPFN-V3 and 0.528 for TabICLv2,
scale near 1) uses a Hill reference at k_share = 0.10, never varied. TODO P1a asks for
0.05, 0.10 and 0.20 before the number is read as more than exploratory. K_SHARES=0.05,
0.1,0.2 refits nothing extra: k only moves the reference, so each (dataset, model, seed)
is fitted once and bootstrapped from the same stream as the default run. Same nine
datasets, three seeds, four members, 100 bootstraps, into
dissociation_real_slope_kshare_tabicl.csv and _tabpfn.csv (two streams).

PREDICTIONS WRITTEN BEFORE THE RUN
----------------------------------
Medians over datasets and seeds of the corrected slopes, per model and k.

K1, the dissociation survives the threshold. For TabPFN-V3 and for TabICLv2, at k_share
    0.05 and at 0.20, the median corrected shape slope is below the median corrected
    scale slope.
K2, the number survives it. For TabPFN-V3 and for TabICLv2 the median corrected shape
    slope at 0.05 and at 0.20 lies within 0.20 of its value at 0.10.

GBM carries no prediction. Seen before these were written: a smoke test of GBM on three
datasets at seed 7000, where its corrected shape slope moved from +1.73 to +0.80 on houses
and from +4.32 to -1.85 on house_16H across k, and the k = 0.10 rows equal the stored ones
to 1e-12. No foundation model was run.
```

## `dissociation_real_slope.py` across the Hill share, outcome, 22.10.2026

```
270 cells on CPU in two streams (3272 s and 5147 s); 27 failed, all CPS1988, which keeps
only two usable features as before. At k = 0.10 the TabICLv2 and GBM rows equal the stored
ones to 1e-4. The TabPFN-V3 rows of the stored file come from the remeasure of 13.9.2026
in another environment (Python 3.12.8); 23 of 27 differ, by up to 0.18 in the corrected
shape slope; the median moves from 0.415 to 0.404. The comparison across k stays inside
the new files.

Median corrected slopes, shape / scale:

                 k = 0.05        k = 0.10        k = 0.20
    TabPFN-V3    0.404 / 1.004   0.404 / 1.004   0.253 / 1.004
    TabICLv2     0.537 / 1.023   0.528 / 1.023   0.304 / 1.023
    GBM          1.813 / 0.954   1.778 / 0.954   0.798 / 0.954

K1  HOLDS. The shape slope is below the scale slope for both foundation models at every
    k. Scale is closer to 1 than shape on 8 of 8 target vectors at every k for TabICLv2,
    on 7 of 8 at 0.05 and 8 of 8 at 0.10 and 0.20 for TabPFN-V3.
K2  HALF. TabPFN-V3 stays within 0.20 of its k = 0.10 value (0.000 and 0.151). TabICLv2
    does at 0.05 (0.009) but not at 0.20, where it falls by 0.224.

What this says. The real-data dissociation does not depend on the Hill share: scale is
followed, shape much less, at every k. Its size does: the shape share ranges from 0.25 to
0.54 across k; a larger k lowers it for every model, GBM included. The real-data
shape share is therefore not a number to set beside the synthetic 42 to 44%; the claim
that carries over is the direction. The scale slope uses no Hill estimate and does not
move with k.
```

## Robustness checks of the leverage result, 27.10.2026

Three checks of whether part two holds outside the conditions it was measured under. They
run after the results they check, so they carry no predictions and confirm nothing; what
each could change is fixed here before the run. Runner: `run/robustness_2710.sh`, four
threads per part.

1. **Position of the leverage row.** Every earlier row sits at the centre of x.
   `influence.py` with DOSE_MODE=sd, DOSES=2,4,10,20,50, XI=0.3,0.7, five seeds, TabICLv2,
   TabPFN-V3 and GBM, at POSITION=centre, edge and random (`influence_sd_pos_*.csv`). The
   centre is measured again at the same thread count, since TabPFN's CPU output depends on
   it. If the global change at edge or random differs from the centre by more than half its
   size, the account through the target's standardisation, which is global, needs a local
   term.
2. **Context size.** The same at N_TRAIN=10000, XI=0.7, DOSES=4,20, POSITION=centre
   (`influence_sd_n10k.csv`). If the global change at 10 000 rows is less than half the
   2000-row value at the same dose, the effect is one of small contexts and the thesis says
   so.
3. **Default ensemble and more tables.** `clip_context_real.py` with
   VARIANTS=without_max,raw,clip_200: freMTPL2sev for TabPFN-V3 at N_EST=8
   (`clip_context_real_nest8.csv`); Brazilian_houses and OnlineNewsPopularity, the two
   tables after freMTPL2sev with the largest natural sd shift in `prevalence_data.csv`, for
   TabPFN-V3, TabICLv2 and GBM at one member (`clip_context_real_wide.csv`) and for
   TabPFN-V3 at eight (`clip_context_real_wide_nest8.csv`). The tables are chosen by where
   leverage occurs, not by how a repair does on them. If the eight-member Q(0.99) ratio on
   freMTPL2sev at 4 sd and above lies within 0.05 of 1, the real-data leverage claim for
   TabPFN-V3 is one of the single member only.

## Context size, outcome, 27.10.2026

`influence_sd_n10k.csv` against `influence_sd_pos_centre.csv` (same threads), XI=0.7, five
seeds, one member, global change of Q(0.99), median (median absolute):

                 dose   2000 rows         10 000 rows
    TabICLv2        4   -1.3% (1.5%)      -18.9% (18.9%)
                   20  -15.1% (15.1%)     -38.7% (38.7%)
    TabPFN-V3       4   -6.1% (10.5%)      +0.1% (50.8%)
                   20  -17.8% (17.8%)     +14.9% (49.1%)
    GBM          4, 20  +1.5% (3.2%)       +0.6% (0.8%)

The rule fires for TabPFN-V3 on its signed median: the fall of Q(0.99) seen at 2000 rows
does not carry to 10 000. It does not fire in the sense it was written for: the row's
influence is larger at 10 000 rows, not smaller, with seeds from -57% to +1454%. The
shape TabPFN-V3 reads off a clean context is itself unstable there (xi_without from -0.12
to 0.56 across five seeds, against 0.48 to 0.69 at 2000), so the leverage cannot be told
apart from the model's own variability at this size with five seeds. For TabICLv2 the
effect grows with the context, the opposite of a small-context effect. GBM stays near
zero at both sizes. What carries: the row moves the foundation models' upper quantile at
both sizes and the trees' at neither; the direction for TabPFN-V3 does not.

Two limits on reading this, added the same day. The run measures the change of the
predicted quantile, not its accuracy, so it says nothing yet about whether a repair helps
or costs at 10 000 rows; that needs raw against the repaired arms on a loss. And the
doses are equal sd shifts, not equal errors: one row reaching a shift of 20 is 28 times
the context's maximum at 2000 rows and 21 times at 10 000 (median `y0_over_max`), so the
two sizes are compared at the same leverage, not at the same error in the target's units.

## Position of the leverage row, outcome, 27.10.2026

`influence_sd_pos_{centre,edge,random}.csv`, five seeds, one member, global change of
Q(0.99), median, at the doses where the centre effect is clear (10, 20 and 50 sd):

                       xi   dose   centre    edge   random
    TabPFN-V3         0.3     10   -15.4%  -16.1%   -15.4%
                              20   -11.6%  -10.7%   -10.9%
                              50   -34.3%  -33.9%   -28.3%
                      0.7     10   -14.7%  -14.8%   -15.1%
                              20   -17.8%  -17.1%   -18.0%
                              50   -29.7%  -29.2%   -28.1%
    TabICLv2          0.3     10    -8.6%   -4.9%    -8.9%
                              20    -7.2%   -5.2%    -8.4%
                              50   -12.8%  -10.3%   -10.3%
                      0.7     10   -14.8%   -3.4%   -16.2%
                              20   -15.1%   -8.1%   -17.6%
                              50   -30.2%  -27.3%   -29.9%

GBM stays within -6% to +5% everywhere. The rule does not fire for TabPFN-V3: the global
change is the same wherever the row sits, within 6 points. That is consistent with an
account through the target's standardisation, which is global, but does not single it
out: the output grid and the joint processing of the context would also act globally,
and this run does not separate them. It fires for TabICLv2 in one cell of twelve (edge,
xi 0.7, dose 10: -3.4% against -14.8%) and comes within half a point of it in another
(dose 20). With five seeds that is evidence of position dependence in some conditions,
weaker at the edge for moderate doses and gone by a shift of 50, not an established
local mechanism; a random position matches the centre for both models.

## The repairs' loss at 10 000 rows, 27.10.2026

The context-size check measured how far one row moves the predicted quantile, not whether
a repair helps. This run asks the second question at 10 000 rows. `clip_context.py`,
TabPFN-V3, XI=0.7, SD_SHIFTS=1,20, ten seeds, N_TEST=900, ARMS=raw,clip,log with
CLIP_C=200, at one member (`clip_context_n10k_nest1.csv`) and eight
(`clip_context_n10k_nest8.csv`), six threads (`run/robustness_2710.sh loss10k`). The
measure is the expected pinball under the generator's law (`analysis.dose_ensemble.
generator_law`), paired against raw within a seed. A shift of 20 is the same leverage as
at 2000 rows, not the same error in the target's units.

The 2000-row reference, same law, XI=0.7, from `clip_context_dose_v3_nest{1,8}.csv` (20
seeds), at 0.99: clip_200 below raw at a shift of 20 in 20 and 19 of 20 seeds (median
-7.7% and -2.2%), on a clean context -0.1% and -0.5%; log below raw at a shift of 20 in
20 of 20 at both sizes, on a clean context -2.4% and -7.1%.

Rules, at each ensemble size separately:

P1. The cap's benefit carries: at a shift of 20, clip_200's expected pinball at 0.99 is
    below raw's in at least 8 of 10 seeds. If it fails, the cap is recommended for
    TabPFN-V3 only at contexts of about 2000 rows.
P2. The cap stays cheap: on a clean context the median of clip_200 over raw, less one, at
    0.99 lies within 2%.
P3. The log carries: at a shift of 20 log is below raw at 0.99 in at least 8 of 10 seeds,
    and on a clean context its median over raw, less one, is at most +2%. This is the
    generator's law; it does not lift the costs log showed on real tables.

Reported without a rule: the same at 0.999, the spread over seeds, and raw's change of
Q(0.99) between shifts 1 and 20.

## Default ensemble and more tables, outcome, 28.10.2026

`clip_context_real_nest8.csv`, `clip_context_real_wide.csv` and
`clip_context_real_wide_nest8.csv`, five subsamples per bin, 2000 context rows, 1000 test
rows. The raw arm reproduces `prevalence_models_wide.csv` on both added tables: GBM
exactly, TabICLv2 to 1e-5, TabPFN-V3 to 0.006 per cell, the environment gap of 11.9.2026.

The rule does not fire. On freMTPL2sev at 4 sd and above the eight-member raw `d_q99` has
a median of -6.1%, a ratio of 0.939, outside the band by 1.1 points; at one member it is
-7.8%. The margin is thin: four of five subsamples fall (-1.4% to -15.9%) while one rises
(+5.3%). At 2 to 4 sd the eight-member fall is -3.3% against -6.8%, inside the band.
The real-data claim for TabPFN-V3 holds at the default ensemble but is smaller there.
clip_200 at eight members: pinball at 0.99 below raw in 9 of 10 subsamples at 2 sd and
above (p = 0.021, median -1.8% against -2.4% at one member), -0.1% to -1.0% in the bins
below.

Brazilian_houses (top bins at a median shift of 3.9 and 5.6 sd) reverses the direction:
the largest row raises Q(0.99) for TabPFN-V3 (+6.5% and +7.6% at one member, +5.9% and
+6.5% at eight) and moves TabICLv2's by about +1%. The fall on freMTPL2sev is not a
property of every table with leverage. The cap wins on the sign count at 2 sd and above
(8 of 10 for TabPFN-V3 at one member and for TabICLv2, 9 of 10 for TabPFN-V3 at eight;
medians -14.7%, -16.5% and -21.4%). The sum over the ten cells disagrees at one member:
+19% for TabPFN-V3 and +52% for TabICLv2, each from one cell where the capped pinball is
14 and 8 times raw's. In both cells the context without its largest row is worse still
(206 and 278 against raw's 13 and 10), so the row carried information the test set
needed. At eight members the sum falls by 11% and no cell rises by more than 4%. The ten
cells hold only four distinct largest rows, so they are not ten independent tests.

OnlineNewsPopularity has no subsample at 4 sd (the top bin's median shift is 2.2). Raw
`d_q99` lies within about 5% for every model; the cap lowers pinball at 0.99 in 3 or 4 of
5 subsamples at 2 sd and above by a median of 1% to 4%, the sum by 0.6% to 3.3%. Too
small to read either way.

What follows. The default ensemble keeps the freMTPL2sev result at a smaller size. The
direction in which the largest row moves the quantile depends on the table. A median
sign count can hide a large loss in the few cells where the largest row is genuine, so a
recommendation of the cap reports the sum or the worst cell next to the sign count and
names the ensemble size.

## The repairs' loss at 10 000 rows, outcome at one member, 28.10.2026

`clip_context_n10k_nest1.csv`, ten seeds, `python -m analysis.loss10k`.

P1 holds: clip_200 below raw at a shift of 20 in 10 of 10 seeds, median -12.8% against
-7.7% at 2000 rows. P2 fails as written, on the side of a gain: on a clean context the
cap lowers the loss by a median of 2.05%, 0.05 points outside the band. The cap costs
nothing there, but it no longer leaves a clean context alone: it binds on 3 to 9 of the
10 000 values (median 7) against 0 to 3 of 2000 (median 1), since the largest of 10 000
draws at xi 0.7 lies further out. That these genuine values cost raw 2% of expected loss
fits the unstable shape TabPFN-V3 reads off a clean context of this size (context-size
outcome). P3 holds: log below raw at a shift of 20 in 10 of 10, median -14.6%; on a
clean context -3.7%.

Without a rule: at a shift of 20 the gains at 0.99 span -2.4% to -41.4% for the cap and
-4.8% to -42.5% for log; both beat raw at 0.999 in 10 of 10 (medians -24.8% and
-28.4%). Raw's expected loss rises between shifts 1 and 20 by 14.2% at 0.99 and by 28.5%
at 0.999. Its Q(0.99) moves by a median of -17.0%, from -59.7% to +1428%, the spread
of the context-size check.

At one member the benefit of both repairs carries to 10 000 rows and grows. The
eight-member file is still running; its verdicts follow in their own section.

## The repairs' loss at 10 000 rows, outcome at eight members, 28.10.2026

`clip_context_n10k_nest8.csv`, ten seeds, all 60 rows, `python -m analysis.loss10k`.

P1 fails: clip_200 is below raw at a shift of 20 in 7 of 10 seeds, median -1.4%, from
-34.1% to +4.5%. By the rule the cap is recommended for TabPFN-V3 at the default ensemble
only at contexts of about 2000 rows. The reason is not a cost of the cap: raw barely
suffers, since its expected loss at 0.99 at a shift of 20 lies 0.4% below the clean
context's (+1.8% at 2000 rows, +14.2% at one member here), so a repair has little to
take back. P2 fails again on the side of a gain, -2.57% on a clean context, about the
cap's whole gain at a shift of 20: at this size the cap acts on the context's own tail
rather than on the added row. P3 holds: log below raw at a shift of 20 in 10 of 10,
median -7.6%; on a clean context -9.2%, so its gain does not come from the row either.
On the generator's law log is the better arm at this size, which says nothing about its
costs on real tables.

Without a rule: at 0.999 both arms beat raw at a shift of 20 in 10 of 10 (medians -6.5%
for the cap, -20.0% for log); raw's expected loss there rises by 4.7%. Raw's Q(0.99)
moves by a median of +11.2% between shifts 1 and 20, from -9.2% to +856%.

What follows for the recommendation. At 10 000 rows one row at a shift of 20 hurts
TabPFN-V3 at one member (+14.2% at 0.99) and both repairs take it back. At the default
ensemble it barely hurts, at either size. The cap's gain there is small at both sizes
(-2.2% at 2000 rows, -1.4% here); what changes at 10 000 rows is its consistency, 19 of
20 seeds against 7 of 10. Judged by its use rather than its harm, the cap on the
generator's law is a repair of the single member. At the default ensemble it costs
little (worst seed +4.5%) but gains too little and too unevenly at 10 000 rows to be
recommended there.

## TabICLv2 at its default ensemble, 28.10.2026

Every leverage and repair result for TabICLv2 was measured at one member; the package
default is eight (`n_estimators=8`, tabicl 2.1.1). Its members differ in their views of
the features only. The target is standardised once before the ensemble
(`tabicl/_sklearn/regressor.py:412`) and every member receives that same target
(`tabicl/_sklearn/preprocessing.py:1177`). TabPFN-V3's default ensemble instead fits a
power transform of the target in four of its eight members; at eight members its
one-row cost on the generator nearly vanishes (outcome above). The prediction that
follows, written before any eight-member TabICLv2 row exists: the effect survives the
ensemble for TabICLv2.

Runner `run/tabicl8_2810.sh`, two parts with their own outputs.

1. `influence.py`, TabICLv2 alone, N_EST=8, DOSE_MODE=sd, DOSES=4,20, XI=0.7, five seeds,
   POSITION=centre, at N_TRAIN=2000 (`influence_sd_tabicl8.csv`) and 10 000
   (`influence_sd_tabicl8_n10k.csv`), four threads. The one-member references are
   `influence_sd_pos_centre.csv` and `influence_sd_n10k.csv`. Then `clip_context_real.py`
   on freMTPL2sev, TabICLv2, N_EST=8, VARIANTS=without_max,raw,clip_200,log, CLIP_C=200
   (`clip_context_real_tabicl8.csv`), against the one-member rows of
   `clip_context_real.csv`.
2. `clip_context.py` with MODEL=TabICLv2, XI=0.7, SD_SHIFTS=1,20, ten seeds, N_TEST=900,
   ARMS=raw,clip,log, CLIP_C=200, at N_TRAIN=2000 and 10 000, at one member and eight
   (`clip_context_tabicl_n{2k,10k}_nest{1,8}.csv`), six threads. The script gained the
   same day a path for a model without a grid; its two border columns stay blank there
   and TabPFN rows are computed as before. Expected pinball under the generator's law,
   paired against raw within a seed. No one-member TabICLv2 row of this design exists, so
   both ensemble sizes are new.

Rules, each at each context size and ensemble size where it applies:

T1. The leverage survives the ensemble on the generator: at a dose of 20 the
    eight-member median global change of Q(0.99) is negative and at least half the
    one-member value (-15.1% at 2000 rows, -38.7% at 10 000), so at most -7.6% and
    -19.4%. If it fails at both sizes, the effect for TabICLv2 is one of the single
    member and the thesis says so for both models. If it holds, the difference between
    the two ensembles is a candidate explanation, not a tested one: this run changes the
    ensemble, not how the target is handled within it.
T2. The leverage survives the ensemble on real data: on freMTPL2sev at 4 sd and above the
    eight-member raw `d_q99` lies at least 0.05 below 0 (the band of the TabPFN-V3 rule
    of 27.10). If it lies within the band, the real-data claim for TabICLv2 is one of the
    single member.
T3. The row costs in loss at eight members: raw's expected pinball at 0.99 at a shift of
    20 over the clean context, median, is at least +2% and at least half the one-member
    rise at the same size.
T4. The cap is worth recommending: at a shift of 20 clip_200's expected pinball at 0.99
    is below raw's in at least 8 of 10 seeds while on a clean context its median over
    raw, less one, is at most +2%. The second half bounds a cost only; P2 of 27.10 was written
    as a band and failed on the side of a gain, which says nothing against the cap.
T5. The log is worth recommending: the same two conditions for log. On the generator's
    law only; it does not lift the costs log showed on real tables.

Reported without a rule: the same at 0.999; the sum over seeds and the worst seed next
to every sign count (outcome of 28.10); on freMTPL2sev the pinball at 0.99 of clip_200
and log against raw at 2 sd and above with its sum and worst cell; the run times.

## TabICLv2 at its default ensemble, outcome, 28.10.2026

`influence_sd_tabicl8.csv` and `influence_sd_tabicl8_n10k.csv` (five seeds each),
`clip_context_real_tabicl8.csv` (five subsamples per bin) and
`clip_context_tabicl_n{2k,10k}_nest{1,8}.csv` (ten seeds, 60 rows each),
`python -m analysis.tabicl8`. Both parts were stopped once from outside at 12:02 and
resumed by their keys; no finished cell was run twice.

T1 holds at both sizes. At a dose of 20 the eight-member median global change of Q(0.99)
is -15.9% at 2000 rows (from -39.9% to -10.3%) against -15.1% at one member; -37.3% at
10 000 rows (from -50.4% to -28.8%) against -38.7%; every seed falls. At a dose of 4 it
is +3.2% and -17.2%. The ensemble leaves TabICLv2's effect whole where TabPFN-V3's nearly
vanishes. As the rule says, the way each model handles the target is a candidate
explanation of the difference, not a tested one.

T2 holds. On freMTPL2sev at 4 sd and above the eight-member raw `d_q99` has a median of
-18.3% against -15.4% at one member; all five subsamples fall (-17.1% to -27.7%). At 2 to
4 sd it is -8.5% against -5.2%. The script's control against `prevalence_models_wide.csv`
prints FAILS (largest gap 0.08 on 20 cells): that file is at one member, so the gap is the
ensemble's, not the instrument's. clip_200 brings `d_q99` at 4 sd and above to +2.1%.
log moves Q(0.99) by -16.3% already below 1.2 sd, where no row carries leverage, so its
`d_q99` is no measure of repair. Pinball at 0.99 at 2 sd and above: clip_200 below raw in
9 of 10 cells (median -3.8%, sum -3.0%, worst cell +12.8%), log in 8 of 10 (median -5.0%,
sum -3.4%, worst cell +11.1%).

T3 fails at 2000 rows and holds at 10 000. Raw's expected pinball at 0.99 at a shift of
20 rises over the clean context by a median of +1.97% at 2000 rows, 0.03 points short of
the bar (+1.85% at one member). At 10 000 rows it rises by +2.31% (+2.48% at one member).
The rise is small at every size: 8 or 9 of 10 seeds rise, single seeds by up to +48%
at 2000 rows and +28% at 10 000. TabPFN-V3 at one member and 10 000 rows rises by
+14.2%. At 0.999 the row lowers raw's expected loss, by 8.4% and 8.6% at 2000 rows
(one and eight members) and by 6.6% and 6.5% at 10 000.

T4 holds at every size and ensemble size. clip_200 below raw at a shift of 20:

    rows    members  below   median   sum     worst seed   clean context
    2000    1        8/10    -1.0%    -5.9%   +1.9%        -0.30%
    2000    8        8/10    -1.3%    -6.5%   +1.9%        -0.31%
    10 000  1        9/10    -2.5%    -5.8%   +0.7%        -0.00%
    10 000  8        10/10   -2.3%    -5.7%   -0.1%        -0.12%

The gain is about the size of the row's cost, so the cap takes back what little there is.
At 0.999 it is worse than raw: below it in 1, 1, 0 and 0 of 10 seeds, medians +6.7%,
+6.8%, +11.2% and +12.0%, since there it undoes the row's gain.

T5 holds at every size and ensemble size: log below raw at a shift of 20 in 10 of 10
everywhere, medians -5.0%, -5.5%, -4.6% and -4.1% (worst seed -1.3% to -2.0%). On a clean
context it is already -3.3%, -3.2%, -2.4% and -2.5%, so about two points of the gain come
from the row and the rest from fitting on log(y). At 0.999 it is below raw in 10, 10, 9
and 9 of 10. On the generator's law only, as for TabPFN-V3 at 10 000 rows.

Run times: a median fit of 4 s and 24 s at 2000 rows (one and eight members), 22 s and
117 s at 10 000; the leverage sweep 487 s at 2000 rows and 3863 s at 10 000; about 34 s
per arm on freMTPL2sev at eight members.

What follows. For TabICLv2 the leverage result is not one of the single member: it holds
at the default ensemble on the generator at both sizes and on freMTPL2sev, at full size.
For TabPFN-V3 it shrinks at the default ensemble. For TabICLv2 the row moves the quantile
far more than the loss: Q(0.99) falls by 16% to 37% while the expected loss at 0.99 rises
by about 2% and the loss at 0.999 falls. Both repairs pass their rules at 0.99, but they
have little to repair there; the cap turns into a loss at 0.999. A test of the candidate
explanation would change how the target is handled within one ensemble, not the ensemble.

## TabPFN-V3 without its target transform, 28.10.2026

The outcome above leaves a candidate explanation. TabICLv2's leverage survives its
default ensemble because every member receives the same target, standardised once.
TabPFN-V3's effect nearly vanishes at eight members because four of them fit a power
transform of the target: REGRESSION_Y_PREPROCESS_TRANSFORMS = (None, "safepower")
(`tabpfn/inference_config.py:181`), crossed with the feature preprocessings and cycled
over the members (`regressor.py:899`, `preprocessing/ensemble.py:1133`). A single member
always gets the first entry, no transform. The 22.10 dose files show the gap at a shift
of 20 and xi 0.7: the median Q(0.99) falls by 22.8% at one member and by 2.2% at eight;
the expected pinball at 0.99 rises by 8.2% and by 1.8%.

The test keeps TabPFN-V3 at eight members and changes only the target. TABPFN_Y=none
(`common/models.py`) sets REGRESSION_Y_PREPROCESS_TRANSFORMS to (None,). A smoke fit
before this section confirmed that all eight members then carry no target transform; the
feature preprocessings (four members each of squashing_scaler_default and quantile_uni)
and the feature shifts stay as at the default. The knob is the column `y_transforms` and
part of the resume key of `clip_context.py` and `influence.py`. The prediction, written
before any row: without the transform the effect comes back.

Runner `run/tabpfn_y_2810.sh`, two parts with their own outputs; verdicts from
`python -m analysis.tabpfn_y`.

1. `clip_context.py`, MODEL=TabPFN-V3, ARMS=raw, XI=0.7, SD_SHIFTS=1,20, twenty seeds,
   N_TRAIN=2000, N_TEST=900, six threads, three arms in turn: one member
   (`clip_context_ytf_nest1.csv`), eight at the default (`clip_context_ytf_nest8.csv`)
   and eight with TABPFN_Y=none (`clip_context_ytf_nest8_none.csv`). The seeds and the
   design are those of `clip_context_dose_v3_nest{1,8}.csv`. Those ran on another
   machine, so the one-member and default arms are measured again here at the same
   thread count.
2. `influence.py`, MODELS=TabPFN-V3, N_EST=8, DOSE_MODE=sd, DOSES=4,20, XI=0.7,
   POSITION=centre, five seeds, 2000 rows, four threads, at the default
   (`influence_sd_tabpfn8.csv`) and with TABPFN_Y=none (`influence_sd_tabpfn8_none.csv`).
   The one-member reference is `influence_sd_pos_centre.csv`, from this machine at four
   threads: -17.8% at a dose of 20.

Three measures, each the median over seeds of a paired change at a shift or dose of 20:
Q, the median predicted Q(0.99) on the test rows with the row over the clean context,
less one (`q99` of `clip_context.py`); R, raw's expected pinball at 0.99 under the
generator's law with the row over the clean context, less one; G, the global change of
Q(0.99) of `influence.py`. A suffix names the arm: 1 for one member, 8 for eight at the
default, 8n for eight without the transform. For each measure the share of the gap the
transform accounts for is s = (X8n - X8) / (X1 - X8).

Y0. The gap reproduces here: Q8 lies above Q1 / 2, R8 below R1 / 2 and G8 above G1 / 2,
    with Q1 and G1 negative and R1 positive. A measure that fails Y0 has no gap to
    explain; it is reported without a verdict.
Y1. Q: s >= 0.5.
Y2. R: s >= 0.5.
Y3. G: s >= 0.5.

The explanation is supported if every measure with a verdict has s >= 0.5. It is
rejected if every one has s <= 0.25: then the ensemble itself removes TabPFN-V3's effect
and the difference from TabICLv2 stays unexplained. Otherwise the transform accounts for
part of the gap and the thesis gives the shares.

Reported without a rule: the same at 0.999; the sign counts with the worst seed; the
dose of 4; the run times.

## TabPFN-V3 without its target transform, outcome, 28.10.2026

`clip_context_ytf_nest1.csv`, `clip_context_ytf_nest8.csv` and
`clip_context_ytf_nest8_none.csv` (twenty seeds, 40 rows each), `influence_sd_tabpfn8.csv`
and `influence_sd_tabpfn8_none.csv` (five seeds), `python -m analysis.tabpfn_y`. The
one-member and default arms reproduce `clip_context_dose_v3_nest{1,8}.csv` to the last
digit on all 40 shared rows (q99, pb99, epb99, q999); the remeasurement was not needed.

Y0 holds for all three measures. At the default eight members Q is -2.2% against -22.8%
at one member, R +1.8% against +8.2%, G -2.8% against -17.8%.

Y1 holds: without the transform Q is -21.4%, s = 0.93. Seed by seed the eight members
without the transform follow the single member, a correlation of 0.98 over the twenty
seeds against 0.67 for the default. Q falls in 12 of 20 seeds in every arm. At one
member the falling seeds fall by 13% to 62% while the others rise by 8% to 38%.

Y2 holds: without the transform R is +6.8%, s = 0.78. Raw's expected loss at 0.99 rises
in 20 of 20 seeds (the smallest rise +2.1%), against 17 of 20 at the default (one seed
-5.8%) and 20 of 20 at one member.

Y3 holds: without the transform G is -15.3%, s = 0.83, a fall in 5 of 5 seeds. At the
default 3 of 5 fall and one seed rises by 14.7%. At a dose of 4 G is -6.1% at one
member, -2.6% at the default and -7.2% without the transform.

The explanation is supported, with shares of 0.93, 0.78 and 0.83.

Without a rule: at 0.999 raw's expected loss rises by about the same in every arm (+5.3%
at one member, +4.8% at the default, +4.9% without the transform), so the transform
shields the level of 0.99 but not that of 0.999. A median fit takes 9 s at one member,
51 s at eight at the default and 32 s without the transform; the leverage sweeps took
1060 s and 1047 s.

What follows. The power transform that half of TabPFN-V3's default members fit to the
target accounts for most of the gap between one member and eight. Averaging over views
of the features does not remove the effect: eight members on the untransformed target
behave like one. The candidate explanation of the TabICLv2 outcome is thereby supported
on the side of TabPFN-V3: the two ensembles differ in how they treat the target. Whether
TabICLv2 would lose its effect under a power transform of the target is not tested here.
The test covers the generator at 2000 rows, xi 0.7 and a shift of 20, not real tables.
TabPFN-V3's default thus carries a partial repair of its own, a non-linear transform of
the target fitted before the model, of the same kind as the log arm.
