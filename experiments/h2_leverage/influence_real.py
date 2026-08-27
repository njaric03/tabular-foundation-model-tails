# -*- coding: utf-8 -*-
"""
Eksperiment 1: uticaj JEDNE STVARNE tacke, bez ijednog ubacivanja.

ZASTO JE OVO ODLUCUJUCE
-----------------------
`influence.py` ubacuje tacku sa y = 100 * max. Merenje ucestalosti pokazuje da
takva tacka u stvarnim podacima NE POSTOJI (0% u sest skupova). Dakle ta intervencija
je vestacka i najlaksa meta za prigovor.

Ovde se nista ne ubacuje. Uzme se uzorak iz stvarnog skupa, nadje se NJEGOVA najveca
vrednost -- tacka koja je vec tu -- i model se fituje DVA puta: sa njom i bez nje.
Razlika je uticaj jedne stvarne opservacije.

Ako se predvidjeni ekstremni kvantil pomeri za desetine procenata zbog jednog reda koji
u podacima legitimno stoji, onda pitanje nije "sta ako neko ubaci outlier" nego
"sta vec jeste slucaj".

KONTROLA
--------
Uz izbacivanje NAJVECE tacke meri se i izbacivanje SLUCAJNE tacke, na istom uzorku i
istom seedu. Ako je efekat isti, merimo samo sum od promene velicine skupa.

MERI SE
-------
  promena Q(0.99), Q(0.9), Q(0.5) na izdvojenom test skupu
  leverage: max / drugi_po_velicini u tom uzorku

Rezultati u `influence_real.csv`.

POKRETANJE
----------
    python -u influence_real.py
    SKUPOVI=freMTPL2sev SEEDOVA=3 MODELI=TabICLv2 python -u influence_real.py
"""
import importlib.util
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import append, datasets, models, paths

warnings.filterwarnings("ignore")


PODRAZUMEVANI = ["freMTPL2sev", "OnlineNewsPopularity", "Buzzinsocialmedia_Twitter",
                 "Allstate_Claims_Severity"]
SKUPOVI = os.environ.get("DATASETS", ",".join(PODRAZUMEVANI)).split(",")
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM").split(",")
SEEDOVA = int(os.environ.get("SEEDS", "10"))
N_EST = int(os.environ.get("N_EST", "1"))
N_FIT, N_TEST = 2000, 1500
NIVOI = [0.5, 0.9, 0.99]
OUT = os.environ.get("OUTPUT", "influence_real.csv")
KOLONE = ["n_est", "dataset", "model", "seed", "leverage", "max_y",
          "d_q50", "d_q90", "d_q99", "k_q50", "k_q90", "k_q99", "seconds", "reason"]
KEY = ["dataset", "model", "seed", "n_est"]


def kvantili(ime, Xtr, ytr, Xte, seed):
    return models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


def promena(q_bez, q_sa):
    """Relativna promena po nivou: sa cim se poredi je fit BEZ te tacke."""
    out = []
    for i in range(len(NIVOI)):
        a, b = q_bez[:, i], q_sa[:, i]
        ok = np.abs(a) > 1e-9
        out.append(float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan)
    return out


def jedan(skup, ime, seed, kes):
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"])
    X, y = kes[skup]
    ok = np.isfinite(y) & (y > 0)
    X, y = X[ok], y[ok]
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    Xf, yf, Xte = X[fi], y[fi], X[ti]

    j_max = int(np.argmax(yf))
    poredak = np.sort(yf)
    leverage = float(poredak[-1] / max(poredak[-2], 1e-12))

    # bez najvece tacke
    maska = np.ones(len(yf), bool); maska[j_max] = False
    q_bez = kvantili(ime, Xf[maska], yf[maska], Xte, seed)
    # sa njom (pun uzorak)
    q_sa = kvantili(ime, Xf, yf, Xte, seed)
    d = promena(q_bez, q_sa)

    # KONTROLA: izbaci slucajnu tacku umesto najvece
    j_slu = int(rng.integers(len(yf)))
    while j_slu == j_max:
        j_slu = int(rng.integers(len(yf)))
    maska2 = np.ones(len(yf), bool); maska2[j_slu] = False
    q_bez2 = kvantili(ime, Xf[maska2], yf[maska2], Xte, seed)
    k = promena(q_bez2, q_sa)

    return dict(leverage=leverage, max_y=float(poredak[-1]),
                d_q50=d[0], d_q90=d[1], d_q99=d[2],
                k_q50=k[0], k_q90=k[1], k_q99=k[2], reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                if append.key(dict(dataset=skup, model=ime, seed=seed), KEY[:3]) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(skup, ime, seed, kes)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                r.update(dataset=skup, model=ime, seed=seed,
                         seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                poruka = r.get("reason") or (
                    f"leverage {r['leverage']:>5.1f}x | najveca: q99 {r['d_q99']:+6.1%} "
                    f"q50 {r['d_q50']:+5.1%} | kontrola q99 {r['k_q99']:+5.1%}")
                print(f"  {skup[:24]:<25} {ime:10s} s={seed} {poruka} [{r['seconds']}s]",
                      flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.d_q99.notna()]
    print("\n=== UTICAJ JEDNE STVARNE TACKE na Q(0.99), medijana ===")
    print(d.pivot_table(index="dataset", columns="model", values="d_q99", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== KONTROLA: izbacena SLUCAJNA tacka ===")
    print(d.pivot_table(index="dataset", columns="model", values="k_q99", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== leverage (max / drugi po velicini) po skupu ===")
    print(d.groupby("dataset").leverage.median().round(2).to_string())
    print("\n=== efekat kod uzoraka sa velikom polugom (>= 3x) ===")
    v = d[d.leverage >= 3]
    if len(v):
        print(f"  n={len(v)}")
        print(v.pivot_table(index="model", values=["d_q99", "k_q99"], aggfunc="median")
              .map(lambda x: f"{x:+.1%}").to_string())
    else:
        print("  nema takvih uzoraka")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
