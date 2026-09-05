# -*- coding: utf-8 -*-
"""
Popravlja li MESAVINA ono sto vincentizacija kvari? I zasto TabICL nije pogodjen?

ZASTO
-----
`findings/h1/vincentization.md`: usrednjavanje kvantila (vincentizacija) spusta implicirano xi
kod TabPFN-a za 0.17 pri xi=0.9, dok TabICL nije pogodjen. Mehanizam je poznat
(Lichtendahl i dr., Management Science 2013: usrednjavanje kvantila je UVEK ostrije),
ali dva pitanja ostaju:

  1. Da li MESAVINA (linearni pul: usrednji CDF-ove, pa invertuj) cuva rep?
     Ako da, to nije samo dijagnoza nego GOTOVA ISPRAVKA.
  2. Zasto TabICL nije pogodjen? Ostrenje zavisi od NESLAGANJA medju clanovima;
     ako se TabICL-ovi clanovi manje razilaze, mehanizam je zatvoren.

DIZAJN
------
Umesto ansambla iz biblioteke, clanovi se prave rucno: M modela sa `n_estimators=1` i
razlicitim `random_state`. Time se dobijaju POJEDINACNE clanske raspodele, sto biblioteka
ne izlaze.

Za svaku test tacku se onda racunaju tri stvari iz ISTIH clanova:

  JEDAN         prosek impliciranog xi po clanu (referenca bez agregacije)
  VINCENT       usrednji kvantile po nivou, pa izracunaj xi   <- ono sto biblioteka radi
  MESAVINA      usrednji CDF-ove, invertuj, pa izracunaj xi   <- predlozena ispravka

Uz to se meri NESLAGANJE clanova: koeficijent varijacije Q(0.99) preko clanova.

KAKO CITATI
-----------
  MESAVINA blize JEDNOM nego VINCENT  -> ispravka radi, i to je preporuka
  neslaganje kod TabPFN >> TabICL     -> mehanizam zatvoren
  MESAVINA = VINCENT                  -> agregacija nije uzrok, traziti dalje

Rezultati u `mixture.csv`.

POKRETANJE
----------
    python -u mixture.py
    XI=0.9 CLANOVA=4 SEEDOVA=3 python -u mixture.py
"""
import importlib.util
import os
import time

import numpy as np
import pandas as pd

from common import append, generator, metrics, models, paths, quiet

quiet.silence()


XI_LISTA = [float(v) for v in os.environ.get("XI", "0.7,0.9").split(",")]
CLANOVA = int(os.environ.get("MEMBERS", "4"))
SEEDOVA = int(os.environ.get("SEEDS", "3"))
MODELI = models.parse_list(os.environ.get("MODELS", "TabICLv2,TabPFN-V3"))
N_TRAIN, N_TEST = 2000, 900
# Gusta mreza za CDF, ali sa TACNO 0.90 i 0.99 unutra -- inace brojke nisu uporedive
# sa ostalim skriptama koje mere bas na tim nivoima.
NIVOI = np.unique(np.round(np.concatenate([np.linspace(0.02, 0.995, 80),
                                           [0.90, 0.99]]), 4))
I_LO = int(np.where(NIVOI == 0.90)[0][0])
I_HI = int(np.where(NIVOI == 0.99)[0][0])
OUT = os.environ.get("OUTPUT", "mixture.csv")
KOLONE = ["members", "xi", "model", "seed", "xi_single", "xi_vincentized", "xi_mixture",
          "disagreement_cv", "seconds", "reason"]
KEY = ["xi", "model", "seed", "members"]


def clan_kvantili(ime, Xtr, ytr, Xte, rs):
    """Jedan clan: n_estimators=1, dat random_state. Vraca (n_test, n_nivoa)."""
    if ime == "TabICLv2":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=1, device="cpu", random_state=rs)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict(Xte, output_type="quantiles", alphas=list(NIVOI)))
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=1, device="cpu", random_state=rs,
                        ignore_pretraining_limits=True)
    m.fit(Xtr, ytr)
    return np.stack([np.asarray(a) for a in
                     m.predict(Xte, output_type="quantiles", quantiles=list(NIVOI))], axis=1)


def kvantil_mesavine(Q, nivoi, ciljevi):
    """
    Kvantili jednako-tezinske MESAVINE clanova.

    Q: (M, n_test, A) kvantili po clanu. Za svaku tacku se napravi zajednicka y-mreza
    od svih clanskih kvantila, svaki clanski CDF se interpolira na nju, CDF-ovi se
    usrednje, pa se invertuje na ciljnim nivoima.
    """
    M, n, A = Q.shape
    out = np.empty((n, len(ciljevi)))
    for i in range(n):
        y = np.sort(Q[:, i, :].ravel())
        F = np.zeros_like(y)
        for m in range(M):
            F += np.interp(y, Q[m, i, :], nivoi, left=nivoi[0], right=nivoi[-1])
        F /= M
        # F je neopadajuce po konstrukciji; invertuj linearnom interpolacijom
        out[i] = np.interp(ciljevi, F, y)
    return out


def jedan(xi, ime, seed):
    rng = np.random.default_rng(seed)
    _p = generator.gpd(N_TRAIN, rng, xi=xi, clip=True)
    Xtr, ytr = _p.X, _p.y
    _p = generator.gpd(N_TEST, rng, xi=xi, clip=True)
    Xte, _ = _p.X, _p.y
    Q = np.stack([clan_kvantili(ime, Xtr, ytr, Xte, seed + 137 * k)
                  for k in range(CLANOVA)], axis=0)          # (M, n, A)

    def xi_iz(num, den):
        ok = den > 1e-9
        return metrics.xi_from_ratio(float(np.median(num[ok] / den[ok])))

    # JEDAN clan: prosek impliciranog xi preko clanova
    poj = [xi_iz(Q[m, :, I_HI], Q[m, :, I_LO]) for m in range(CLANOVA)]
    xi_single = float(np.nanmean(poj))

    # VINCENT: usrednji kvantile po nivou
    V = Q.mean(axis=0)
    xi_vin = xi_iz(V[:, I_HI], V[:, I_LO])

    # MESAVINA: usrednji CDF-ove, invertuj
    Mx = kvantil_mesavine(Q, NIVOI, np.array([NIVOI[I_LO], NIVOI[I_HI]]))
    xi_mes = xi_iz(Mx[:, 1], Mx[:, 0])

    # neslaganje clanova na Q(0.99)
    q99 = Q[:, :, I_HI]
    cv = float(np.median(q99.std(axis=0, ddof=1) / np.maximum(np.abs(q99.mean(axis=0)), 1e-9)))
    return dict(xi_single=xi_single, xi_vincentized=xi_vin, xi_mixture=xi_mes,
                disagreement_cv=cv, reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    t0 = time.time()
    for xi in XI_LISTA:
        for ime in MODELI:
            for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
                if append.key(dict(xi=xi, model=ime, seed=seed), KEY[:3]) in gotovi:
                    continue
                t1 = time.time()
                try:
                    r = jedan(xi, ime, seed)
                except Exception as e:
                    r = dict(reason=f"{type(e).__name__}: {e}"[:100])
                r.update(members=CLANOVA, xi=xi, model=ime, seed=seed, seconds=round(time.time() - t1, 1))
                append.write(OUT, r, KOLONE)
                poruka = r.get("reason") or (
                    f"jedan {r['xi_single']:+.3f} | vincent {r['xi_vincentized']:+.3f} | "
                    f"mesavina {r['xi_mixture']:+.3f} | neslaganje {r['disagreement_cv']:.3f}")
                print(f"  xi={xi} {ime:10s} seed={seed}  {poruka}  [{r['seconds']}s]", flush=True)

    d = pd.read_csv(paths.result(OUT)); # Filter on whether the MEASUREMENT succeeded, not on notna() of the estimate:
    # xi_from_ratio returns NaN exactly on the heaviest tails (outside the brentq
    # bracket), so filtering on the estimate drops heavy tails preferentially.
    d = d[d.reason.fillna("") == ""] if "reason" in d.columns else d[d.xi_single.notna()]
    print("\n=== IMPLICIRANO xi: jedan clan / vincentizacija / mesavina ===")
    for kol in ["xi_single", "xi_vincentized", "xi_mixture"]:
        print(f"\n  {kol}")
        print("    " + d.pivot_table(index="xi", columns="model", values=kol)
              .round(3).to_string().replace("\n", "\n    "))
    print("\n=== koliko agregacija kosta, po nacinu ===")
    for (xi, m), g in d.groupby(["xi", "model"]):
        j, v, s = g.xi_single.mean(), g.xi_vincentized.mean(), g.xi_mixture.mean()
        print(f"  xi={xi} {m:<12} jedan {j:+.3f} | vincent {v:+.3f} ({v-j:+.3f}) | "
              f"mesavina {s:+.3f} ({s-j:+.3f})")
    print("\n=== neslaganje clanova (CV od Q(0.99)) ===")
    print(d.pivot_table(index="xi", columns="model", values="disagreement_cv")
          .round(3).to_string())
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
