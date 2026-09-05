# -*- coding: utf-8 -*-
"""
Je li deo "kompresije repa" artefakt AGREGACIJE ansambla, a ne mreze?

ZASTO
-----
Clanovi ansambla se kod TFM-ova usrednjavaju PO KVANTILIMA u skali targeta
(vincentizacija), a ne kao mesavina raspodela. Za dva izvora nesigurnosti to nije
isto: mesavina zadrzava rep najteze komponente, dok vincentizacija usrednjava
kvantile i time sistematski daje LAKSI rep.

Ako implicirano xi opada sa brojem clanova ansambla, onda deo dosadasnjeg nalaza
nije o prioru nego o tome kako se ansambl sklapa -- a to menja tumacenje.

MERA
----
Generator sa KONSTANTNIM xi (bez zavisnosti od x), pa se meri implicirano xi iz
Q(0.99)/Q(0.9) za n_estimators iz {1, 2, 4, 8}. Sve ostalo fiksno.

  xi opada sa brojem clanova   -> agregacija kompresuje; nalaz treba oslabiti
  xi ravno                     -> agregacija nije kriva; nalaz stoji kakav jeste
  xi raste                     -> agregacija pomaze, nalaz je bio konzervativan

Rezultati u `vincentization.csv`.

POKRETANJE
----------
    python -u vincentization.py
    XI=0.7 SEEDOVA=5 python -u vincentization.py
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


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.3,0.7,0.9").split(",")]
N_EST = [int(v) for v in os.environ.get("N_EST", "1,2,4,8").split(",")]
SEEDOVA = int(os.environ.get("SEEDS", "5"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3"))
N_TRAIN, N_TEST = 2000, 900
OUT = os.environ.get("OUTPUT", "vincentization.csv")
KOLONE = ["xi_true", "model", "n_estimators", "seed", "xi_implied", "seconds", "reason"]
KEY = ["xi_true", "model", "n_estimators", "seed"]


def kvantili(ime, n_est, Xtr, ytr, Xte, seed):
    if ime == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=n_est, device="cpu", random_state=seed)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=QS))
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=n_est, device="cpu", random_state=seed,
                        ignore_pretraining_limits=True)
    m.fit(Xtr, ytr)
    return np.stack([np.asarray(a) for a in
                     m.predict(Xte, output_type="quantiles", quantiles=QS)], axis=1)


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for ne in N_EST:
                for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                    if append.key(dict(xi_true=xi, model=ime, n_estimators=ne, seed=seed), KEY[:4]) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        rng = np.random.default_rng(seed)
                        _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
                        Xtr, ytr = _p.X, _p.y
                        _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
                        Xte, _ = _p.X, _p.y
                        q = kvantili(ime, ne, Xtr, ytr, Xte, seed)
                        num = q[:, QS.index(HI)]
                        den = q[:, QS.index(LO)]
                        ok = den > 1e-9
                        r = dict(xi_implied=metrics.xi_from_ratio(float(np.median(num[ok] / den[ok]))),
                                 reason="")
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                    r.update(xi_true=xi, model=ime, n_estimators=ne, seed=seed,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    poruka = r.get("reason") or f"xi_impl {r['xi_implied']:+.3f}"
                    print(f"  xi={xi} {ime:10s} n_est={ne} seed={seed}  {poruka}"
                          f"  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""] if "reason" in d.columns else d[d.xi_implied.notna()]
    print("\n=== IMPLICIRANO xi PO BROJU CLANOVA ANSAMBLA ===")
    for ime, g in d.groupby("model"):
        print(f"\n  {ime}")
        p = g.pivot_table(index="xi_true", columns="n_estimators", values="xi_implied")
        print("    " + p.round(3).to_string().replace("\n", "\n    "))
    print("\n=== nagib: promena xi od n_est=1 do n_est=max ===")
    for (ime, xi), g in d.groupby(["model", "xi_true"]):
        a = g[g.n_estimators == min(N_EST)].xi_implied.mean()
        b = g[g.n_estimators == max(N_EST)].xi_implied.mean()
        print(f"  {ime:<12} xi={xi}:  {a:+.3f} -> {b:+.3f}   ({b-a:+.3f})")
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
