# Predictions written before each run, and what happened

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

## Contents

- `experiments/h1_shape_vs_scale/xi_residual.py`
- `experiments/h2_leverage/architecture_tabdpt.py`
- `experiments/h2_leverage/architecture_tabfm.py`
- `experiments/h2_leverage/prevalence_models.py`
- `experiments/h2_leverage/grid_occupancy.py`
- `experiments/side/head_tail_tabicl.py`
- `experiments/h3_repair/clip_context.py`
- `experiments/h3_repair/clip_context_real.py`
- `experiments/h3_repair/unit_error_real.py`
- `experiments/h3_repair/tail_splice.py`

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
