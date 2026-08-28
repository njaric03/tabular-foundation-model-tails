# -*- coding: utf-8 -*-
"""
Koherentnost pod reparametrizacijom targeta.

ZASTO
-----
Vec je izmereno (mejl v15): log-transformacija targeta podize implicirano xi sa 0.47 na
0.65 pri pravom xi = 0.7. To je zabelezeno kao kuriozitet, ali je zapravo tvrdnja o
KOHERENTNOSTI modela.

Monotona transformacija g je bijekcija, pa su kvantili EKVIVARIJANTNI:

    Q_{g(Y)}(alpha | x) = g( Q_Y(alpha | x) )

To vazi za pravu uslovnu raspodelu, bez ikakvih pretpostavki. Dakle koherentan prediktor
mora da zadovolji:

    g^{-1}( Q_model(alpha ; g(y)) )  ~=  Q_model(alpha ; y)

Sve odstupanje je NEKOHERENTNOST: predikcija zavisi od proizvoljnog izbora korisnika koji
na istinu ne utice.

Ishod je upotrebljiv u oba smera:
  - ako je odstupanje veliko i sistematsko u jednom smeru -> besplatan rucak
    (transformisi target pa vrati nazad, dobices bolji rep)
  - ako je veliko a nesistematsko -> crvena zastavica za pouzdanost

MERA
----
Fituje se isti model dvaput: na y i na g(y). Drugi se vraca u originalnu skalu preko
g^{-1}. Poredi se:

  xi_sirovo    implicirano xi iz Q(0.99)/Q(0.9) na sirovom fitu
  xi_transf    isto, iz vracenog fita
  odstupanje   medijana |log(Q_transf / Q_sirovo)| po nivoima -- mera nekoherentnosti

Transformacije: log, kubni koren, Yeo-Johnson-like znak-koren.
Sve su monotone i invertibilne na pozitivnom targetu.

Rezultati u `coherence.csv`.

POKRETANJE
----------
    python -u coherence.py
    XI=0.7 TRANSF=log SEEDOVA=3 python -u coherence.py
"""
import importlib.util
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths

warnings.filterwarnings("ignore")


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.3,0.7,0.9").split(",")]
TRANSF = os.environ.get("TRANSF", "log,koren3,koren").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM").split(",")
N_EST = int(os.environ.get("N_EST", "1"))     # 1, da se izbegne efekat iz findings/h1/vincentization.md
N_TRAIN, N_TEST = 2000, 900
NIVOI = [0.5, 0.9, 0.99]
OUT = os.environ.get("OUTPUT", "coherence.csv")
KOLONE = ["n_est", "generator", "xi", "model", "transform", "seed", "xi_raw", "xi_transformed",
          "deviation", "odst_q50", "odst_q90", "odst_q99", "seconds", "reason"]
KEY = ["xi", "model", "transform", "seed", "n_est", "generator"]

TRANSFORMACIJE = {
    "log":    (np.log,                  np.exp),
    "cbrt": (lambda y: np.cbrt(y),    lambda z: z ** 3),
    "sqrt":  (lambda y: np.sqrt(y),    lambda z: np.maximum(z, 0.0) ** 2),
}


# Struktura generatora je KLJUCNA za ovaj test: kod multiplikativnog (y = s(x) * sum)
# log-transformacija po konstrukciji linearizuje strukturu, pa bi "log pomaze" bio
# artefakt generatora a ne nalaz. ADITIVNI generator je kontrola u kojoj log NIJE
# kanonska skala.
GENERATOR = os.environ.get("GENERATOR", "multiplicative")


def podaci(n, xi, rng):
    X = rng.normal(size=(n, 5))
    lin = (X @ generator.W) / np.linalg.norm(generator.W)
    U = np.clip(rng.random(n), 1e-9, 1 - 1e-9)
    sum_ = ((1 - U) ** (-xi) - 1) / xi
    if GENERATOR == "additive":
        # y = mu(x) + sum;  mu > 0 pa je y > 0, a log(y) NIJE aditivno u x
        return X, 5.0 + 2.0 * lin + sum_
    return X, np.exp(0.6 * lin) * sum_


def kvantili(ime, Xtr, ytr, Xte, seed):
    return models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


def xi_iz(q):
    num, den = q[:, NIVOI.index(0.99)], q[:, NIVOI.index(0.9)]
    ok = den > 1e-9
    return metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))) if ok.any() else np.nan


def jedan(xi, ime, naziv_t, seed):
    g, g_inv = TRANSFORMACIJE[naziv_t]
    rng = np.random.default_rng(seed)
    Xtr, ytr = podaci(N_TRAIN, xi, rng)
    Xte, _ = podaci(N_TEST, xi, rng)
    ytr = np.maximum(ytr, 1e-9)

    q_sir = kvantili(ime, Xtr, ytr, Xte, seed)
    q_tr = g_inv(kvantili(ime, Xtr, g(ytr), Xte, seed))

    # nekoherentnost: koliko se vraceni kvantili razlikuju od sirovih
    ok = (q_sir > 1e-9) & (q_tr > 1e-9)
    odst = {}
    for i, a in enumerate(NIVOI):
        m = ok[:, i]
        odst[a] = float(np.median(np.abs(np.log(q_tr[m, i] / q_sir[m, i])))) if m.any() else np.nan
    return dict(xi_raw=xi_iz(q_sir), xi_transformed=xi_iz(q_tr),
                deviation=float(np.nanmean(list(odst.values()))),
                odst_q50=odst[0.5], odst_q90=odst[0.9], odst_q99=odst[0.99],
                reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for naziv_t in TRANSF:
                for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                    if append.key(dict(xi=xi, model=ime, transform=naziv_t, seed=seed), KEY[:4]) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        r = jedan(xi, ime, naziv_t, seed)
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                    r.update(n_est=N_EST, generator=GENERATOR,
                             xi=xi, model=ime, transform=naziv_t, seed=seed,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    poruka = r.get("reason") or (
                        f"xi sirovo {r['xi_raw']:+.3f} -> transf {r['xi_transformed']:+.3f}  "
                        f"| nekoherentnost {r['deviation']:.3f}")
                    print(f"  xi={xi} {ime:10s} {naziv_t:7s} s={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""] if "reason" in d.columns else d[d.xi_raw.notna()]
    print("\n=== IMPLICIRANO xi: sirov fit naspram transformisanog pa vracenog ===")
    for naziv_t, g in d.groupby("transform"):
        print(f"\n  transformacija: {naziv_t}")
        p = g.pivot_table(index="xi", columns="model", values=["xi_raw", "xi_transformed"],
                          aggfunc="median")
        print("    " + p.round(3).to_string().replace("\n", "\n    "))
    print("\n=== NEKOHERENTNOST (medijana |log odnosa| po nivoima) ===")
    print(d.pivot_table(index=["transform", "xi"], columns="model",
                        values="deviation", aggfunc="median").round(3).to_string())
    print("\n=== nekoherentnost po nivou kvantila (prosek preko svega) ===")
    print(d.groupby("model")[["odst_q50", "odst_q90", "odst_q99"]].median().round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
