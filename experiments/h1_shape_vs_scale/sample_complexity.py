# -*- coding: utf-8 -*-
"""
Doza-odgovor po velicini konteksta: popravlja li se OBLIK repa sa n_train?

PITANJE
-------
`findings/h1/dissociation.md`: pri n_train = 2000 modeli hvataju 94% promene SKALE i 36%
promene OBLIKA repa, dok EVT orakl hvata 98% oblika. Ostaje jedno objasnjenje koje
merenje jos nije iskljucilo: mozda je 2000 redova prosto premalo ZA MODEL, iako je
dovoljno za orakl.

Ako je tako, udeo za oblik raste sa n_train i deficit je stvar kolicine podataka.
Ako saturira dok skala stoji na ~95%, onda ogranicenje nije u podacima nego u onome
sto je model doneo sa sobom -- dakle u prioru. To je empirijski dokaz za mehanizam (a)
iz `thesis/notes/pfn_theory.md`, odeljak 5, bez ijednog pretreniranja.

KAKO CITATI
-----------
  oblik raste ka 90%+        -> deficit je kolicina podataka, nalaz je blazi
  oblik saturira ispod ~50%  -> deficit je prior; to je jak, kauzalno-blizak nalaz
  oblik raste ali sporo      -> izmeriti nagib i reci koliko bi n trebalo

Skala se meri uporedo, kao kontrola: ona bi trebalo da ostane visoka na svakom n.

NASTAVLJIVOST
-------------
Pise red po red u `sample_complexity.csv` i preskace vec uradjene (n, model, seed).

POKRETANJE
----------
    python -u sample_complexity.py
    N_GRID=500,1000,2000 SEEDS=3 python -u sample_complexity.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()

QS = [0.5, 0.9, 0.99]
HI, LO = 0.99, 0.9
N_EST = int(os.environ.get("N_EST", "4"))
N_TRAIN, N_TEST = 2000, 900


# `findings/h1/vincentization.md`: ansambl spljosti xi kod TabPFN-a, pa se velicina
# ansambla prosledjuje kroz N_EST koji cita `shape_of_x.py`.
N_GRID = [int(v) for v in os.environ.get("N_GRID", "500,1000,2000,4000").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "GBM,TabICLv2,TabPFN-V3"))
SEEDS = [7000 + 1000 * i for i in range(SEEDOVA)]
N_TERCILA = 3
OUT = os.environ.get("OUTPUT", "sample_complexity.csv")
KLJUC = ["n_train", "model", "seed", "n_est"]
KOLONE = ["n_train", "model", "seed", "n_est", "shape_share", "scale_share",
          "xi_t1", "xi_t3", "xi_true_t1", "xi_true_t3", "seconds", "reason"]


def jedan(n_train, ime, seed):
    # Separate streams for train and test. Drawing the test set after the
    # training set from ONE stream made its position depend on n_train, so the
    # test set differed at every grid point and the extra variance was charged
    # to the n_train axis. The published CSV was measured the old way; the
    # notebook says so where it reports scatter across n_train.
    rng = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1_000_000)
    Xtr, ytr, _, _ = generator.gpd(n_train, rng, xi=generator.XI_OF_X)
    Xte, _, s_te, xi_te = generator.gpd(N_TEST, rng_test, xi=generator.XI_OF_X)
    q = models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=QS, n_est=N_EST)

    # --- OBLIK: tercili po x4 (tezina u skali je nula) ---
    iv = np.quantile(Xte[:, 4], np.linspace(0, 1, N_TERCILA + 1))
    t_ob = np.clip(np.digitize(Xte[:, 4], iv[1:-1]), 0, N_TERCILA - 1)
    xi_hat = []
    for t in range(N_TERCILA):
        sel = t_ob == t
        num, den = q[sel, QS.index(HI)], q[sel, QS.index(LO)]
        ok = den > 1e-9
        r = float(np.median(num[ok] / den[ok])) if ok.any() else np.nan
        xi_hat.append(metrics.xi_from_ratio(r) if np.isfinite(r) else np.nan)
    xi_pr = [float(xi_te[t_ob == t].mean()) for t in range(N_TERCILA)]
    udeo_ob = (xi_hat[2] - xi_hat[0]) / (xi_pr[2] - xi_pr[0])

    # --- SKALA: tercili po linearnom prediktoru ---
    lin = Xte @ generator.W / np.linalg.norm(generator.W)
    iv = np.quantile(lin, np.linspace(0, 1, N_TERCILA + 1))
    t_sk = np.clip(np.digitize(lin, iv[1:-1]), 0, N_TERCILA - 1)
    med_pr = s_te * metrics.gpd_quantile(0.5, xi_te)
    med_mo = q[:, QS.index(0.5)]
    a_mo, b_mo = (np.median(med_mo[t_sk == 0]), np.median(med_mo[t_sk == N_TERCILA - 1]))
    a_pr, b_pr = (np.median(med_pr[t_sk == 0]), np.median(med_pr[t_sk == N_TERCILA - 1]))
    udeo_sk = np.log(b_mo / a_mo) / np.log(b_pr / a_pr) if a_mo > 0 and b_mo > 0 else np.nan

    return dict(shape_share=udeo_ob, scale_share=udeo_sk,
                xi_t1=xi_hat[0], xi_t3=xi_hat[2],
                xi_true_t1=xi_pr[0], xi_true_t3=xi_pr[2], reason="")


def main():
    gotovi = append.done(OUT, KLJUC)
    if gotovi:
        print(f"nastavljam, vec uradjeno {len(gotovi)}", flush=True)

    t0 = time.time()
    for n in N_GRID:
        for ime in MODELI:
            for seed in SEEDS:
                k = dict(n_train=n, model=ime, seed=seed, n_est=N_EST)
                if append.key(k, KLJUC) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(n, ime, seed)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                r.update(k, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                if r.get("reason"):
                    print(f"  n={n:>5} {ime:10s} seed={seed}  -- {r['reason']}", flush=True)
                else:
                    print(f"  n={n:>5} {ime:10s} seed={seed}  oblik {r['shape_share']:>6.0%}"
                          f"  skala {r['scale_share']:>6.0%}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT))
    # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""]
    print("\n=== UDEO PRAVE PROMENE, po velicini trening skupa ===")
    for kol, lab in [("shape_share", "OBLIK xi(x)"), ("scale_share", "SKALA s(x)")]:
        print(f"\n  {lab}")
        p = d.pivot_table(index="n_train", columns="model", values=kol, aggfunc="mean")
        print("    " + p.map(lambda v: f"{v:.0%}").to_string().replace("\n", "\n    "))
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
