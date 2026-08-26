# -*- coding: utf-8 -*-
"""
Objasnjava li ODSECEN NOSAC deficit uslovne sredine?

HIPOTEZA
--------
TabPFN-ova "bar distribution" ima nosac izveden iz OPSEGA trening targeta. Pod teskim
repom prava uslovna sredina zavisi od mase IZNAD max(y_train), koju model strukturno
nema. Ako je tako, model nije "u zabludi" -- on je tacan Bayesovac pod odsecenim
nosacem, a deficit je posledica odsecanja, ne loseg ucenja.

To bi objasnilo i zasto korekcija konstantom nista ne menja (gama devijansa +0.0% kod
TFM-ova u `mean_correction.csv`): nema sta da se skalira, fali masa.

MERA
----
Za GPD(xi, s) sa poznatim xi i s racuna se, po test tacki:

    prava sredina        E[Y|x]            = s / (1 - xi)                  (xi < 1)
    odsecena sredina     E[Y|x, Y <= T]    T = max(y_train)
    modelova sredina     iz predikcije

pa se gleda koji od dva odnosa je blizi jedinici:

    ratio_true = model / prava
    ratio_truncated = model / odsecena

  odnos_odsec ~ 1, odnos_prava < 1   -> hipoteza potvrdjena, mehanizam je odsecanje
  oba < 1                            -> odsecanje objasnjava samo deo
  odnos_odsec > 1                    -> model je iznad i odsecene sredine, hipoteza pada

Formule za standardni GPD(xi, 1), C = (U^(-xi) - 1)/xi:
    S(t)         = (1 + xi t)^(-1/xi)
    E[min(C,t)]  = (1 - (1 + xi t)^(1 - 1/xi)) / (1 - xi)
    E[C | C<=t]  = (E[min(C,t)] - t S(t)) / (1 - S(t))
Skaliranje: Y = s C, T/s je prag u standardnoj skali.

Rezultati u `truncated_support.csv`.

POKRETANJE
----------
    python -u truncated_support.py
    XI=0.7,0.9 SEEDOVA=3 python -u truncated_support.py
"""
import importlib.util
import os
import time
import warnings

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths

warnings.filterwarnings("ignore")


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.5,0.7,0.9").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = os.environ.get("MODELS", "TabICLv2,TabPFN-V3,GBM").split(",")
N_TRAIN, N_TEST = 2000, 900
# `findings/h1/vincentization.md`: velicina ansambla menja rep kod TabPFN-a, pa je podesiva.
N_EST = int(os.environ.get("N_EST", "4"))
OUT = os.environ.get("OUTPUT", "truncated_support.csv")
KOLONE = ["n_est", "xi", "model", "seed", "ratio_true", "ratio_truncated", "threshold_std",
          "share_above_threshold", "seconds", "reason"]
KEY = ["xi", "model", "seed", "n_est"]


def sredina_modela(ime, Xtr, ytr, Xte, seed):
    return models.mean(ime, Xtr, ytr, Xte, seed=seed, n_est=N_EST)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)

    # brza provera formule Monte Carlom
    rng = np.random.default_rng(0)
    for xi, t in [(0.7, 3.0), (0.9, 10.0)]:
        C = ((1 - rng.random(2_000_000)) ** (-xi) - 1) / xi
        mc = C[C <= t].mean()
        an = float(metrics.truncated_mean_std(t, xi))
        print(f"  provera formule xi={xi} t={t}: MC {mc:.4f} vs analiticki {an:.4f}"
              f"  (razlika {abs(mc-an)/an:.2%})", flush=True)

    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                if append.key(dict(xi=xi, model=ime, seed=seed), KEY[:3]) in gotovi:
                    continue
                t1 = time.time()
                try:
                    rng = np.random.default_rng(seed)
                    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
                    Xtr, ytr, _ = _p.X, _p.y, _p.s
                    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
                    Xte, yte, s_te = _p.X, _p.y, _p.s
                    T = float(ytr.max())
                    prava = s_te / (1 - xi)
                    odsec = s_te * metrics.truncated_mean_std(T / s_te, xi)
                    mod = sredina_modela(ime, Xtr, ytr, Xte, seed)
                    r = dict(ratio_true=float(np.mean(mod) / np.mean(prava)),
                             ratio_truncated=float(np.mean(mod) / np.mean(odsec)),
                             threshold_std=float(np.median(T / s_te)),
                             share_above_threshold=float((yte > T).mean()),
                             reason="")
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                r.update(xi=xi, model=ime, seed=seed, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                poruka = r.get("reason") or (f"model/prava {r['ratio_true']:.3f}  "
                                             f"model/odsecena {r['ratio_truncated']:.3f}")
                print(f"  xi={xi} {ime:10s} seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); d = d[d.ratio_true.notna()]
    print("\n=== ODNOS PREMA PRAVOJ SREDINI (1.00 = tacno) ===")
    print(d.pivot_table(index="xi", columns="model", values="ratio_true").round(3).to_string())
    print("\n=== ODNOS PREMA ODSECENOJ SREDINI E[Y | Y <= max(y_train)] ===")
    print(d.pivot_table(index="xi", columns="model", values="ratio_truncated").round(3).to_string())
    print("\n=== koliko odsecanje objasnjava ===")
    for (xi, m), g in d.groupby(["xi", "model"]):
        a, b = g.ratio_true.mean(), g.ratio_truncated.mean()
        obj = (b - a) / (1 - a) if abs(1 - a) > 1e-6 else np.nan
        print(f"  xi={xi} {m:<12} prava {a:.3f} -> odsecena {b:.3f}   "
              f"objasnjeno {obj:.0%}" if np.isfinite(obj) else
              f"  xi={xi} {m:<12} prava {a:.3f} -> odsecena {b:.3f}")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
