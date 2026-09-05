# -*- coding: utf-8 -*-
"""
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
    SKUPOVI=freMTPL2sev MODELI=GBM KVOTA=5 python -u prevalence_models.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, models, paths, quiet

quiet.silence()


N_FIT = int(os.environ.get("N_FIT", "2000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
N_EST = int(os.environ.get("N_EST", "1"))
KVOTA = int(os.environ.get("KVOTA", "8"))
MAKS_POKUSAJA = int(os.environ.get("MAKS_POKUSAJA", "4000"))
MODELI = models.parse_list(os.environ.get("MODELS", "GBM,XGBoost,TabICLv2,TabPFN-V3"))
NIVOI = [0.5, 0.9, 0.99]
KORPE = [(0.0, 1.2), (1.2, 2.0), (2.0, 4.0), (4.0, 1e9)]
OUT = os.environ.get("OUTPUT", "prevalence_models.csv")

if os.environ.get("DATASETS"):
    SKUPOVI = os.environ["DATASETS"].split(",")
elif paths.result("prevalence_data.csv").exists():
    _d = pd.read_csv(paths.result("prevalence_data.csv"))
    _d = _d[_d.reason.isna() & (_d.sd_shift_max >= 2.0)]
    SKUPOVI = list(_d.sort_values("sd_shift_max", ascending=False).dataset)
else:
    SKUPOVI = ["freMTPL2sev"]

KOLONE = ["n_est", "dataset", "model", "bin", "repeat", "sd_shift", "leverage",
          "d_q50", "d_q90", "d_q99", "seconds", "reason"]
KEY = ["dataset", "model", "bin", "repeat", "n_est"]


def kvantili(ime, Xtr, ytr, Xte, seed):
    return models.quantiles(ime, Xtr, ytr, Xte, seed=seed, levels=NIVOI, n_est=N_EST)


def promena(q_bez, q_sa):
    out = []
    for i in range(len(NIVOI)):
        a, b = q_bez[:, i], q_sa[:, i]
        ok = np.abs(a) > 1e-9
        out.append(float(np.median((b[ok] - a[ok]) / a[ok])) if ok.any() else np.nan)
    return out


def nadji_poduzorke(y, rng):
    """Odbacivacko uzorkovanje: po KVOTA poduzoraka u svakoj korpi pomeraja sd."""
    n = min(N_FIT + N_TEST, len(y))
    nadjeno = {k: [] for k in range(len(KORPE))}
    for _ in range(MAKS_POKUSAJA):
        if all(len(v) >= KVOTA for v in nadjeno.values()):
            break
        idx = rng.permutation(len(y))[:n]
        yf = y[idx[:N_FIT]]
        p = np.sort(yf)
        pom = np.std(yf) / max(np.std(p[:-1]), 1e-12)
        pol = abs(p[-1]) / max(abs(p[-2]), 1e-12)
        for ki, (lo, hi) in enumerate(KORPE):
            if lo <= pom < hi and len(nadjeno[ki]) < KVOTA:
                nadjeno[ki].append((idx, pom, pol))
                break
    return nadjeno


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    ids = paths.load_json("sb_openml_ids.json")
    t0 = time.time()

    for dataset in SKUPOVI:
        try:
            X, y = datasets.load(dataset, ids)
        except Exception as e:
            print(f"{dataset}: ne ucitava se ({e})", flush=True)
            continue
        ok = np.isfinite(y) & (y > 0)
        X, y = X[ok], y[ok]
        if len(y) < N_FIT + N_TEST:
            print(f"{dataset}: premalo redova", flush=True)
            continue
        rng = np.random.default_rng(31337)
        poduzorci = nadji_poduzorke(y, rng)
        print(f"\n{dataset}: nadjeno po korpi " +
              ", ".join(f"{KORPE[k][0]}-{KORPE[k][1] if KORPE[k][1]<1e8 else 'inf'}:"
                        f"{len(v)}" for k, v in poduzorci.items()), flush=True)

        for ki, lista in poduzorci.items():
            ime_korpe = f"{KORPE[ki][0]}-{KORPE[ki][1] if KORPE[ki][1] < 1e8 else 'inf'}"
            for j, (idx, pom, pol) in enumerate(lista):
                fi, ti = idx[:N_FIT], idx[N_FIT:]
                Xf, yf, Xte = X[fi], y[fi], X[ti]
                jmax = int(np.argmax(yf))
                maska = np.ones(len(yf), bool); maska[jmax] = False
                for model in MODELI:
                    if append.key(dict(dataset=dataset, model=model, bin=ime_korpe, repeat=j), KEY[:4]) in gotovi:
                        continue
                    t1 = time.time()
                    try:
                        q_bez = kvantili(model, Xf[maska], yf[maska], Xte, 7000 + j)
                        q_sa = kvantili(model, Xf, yf, Xte, 7000 + j)
                        d = promena(q_bez, q_sa)
                        r = dict(d_q50=d[0], d_q90=d[1], d_q99=d[2], reason="")
                    except Exception as e:
                        r = dict(reason=f"{type(e).__name__}: {e}"[:110])
                    r.update(dataset=dataset, model=model, bin=ime_korpe, repeat=j,
                             sd_shift=pom, leverage=pol,
                             seconds=round(time.time() - t1, 1))
                    append.write(OUT, r, KOLONE)
                    por = r["reason"] or f"q99 {r['d_q99']:+7.1%} q50 {r['d_q50']:+6.1%}"
                    print(f"  {dataset[:20]:<21} {model:10s} korpa {ime_korpe:>8} #{j} "
                          f"sd={pom:5.2f} {por} [{r['seconds']}s]", flush=True)

    izvestaj()
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


def izvestaj():
    if not os.path.exists(paths.result(OUT)):
        return
    d = pd.read_csv(paths.result(OUT))
    d = d[d.d_q99.notna()]
    if not len(d):
        print("nema rezultata"); return
    print("\n=== FAZA 2: promena Q(0.99), medijana, po korpi pomeraja sd ===")
    print(d.pivot_table(index="bin", columns="model", values="d_q99", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== kontrola: promena medijane Q(0.5) ===")
    print(d.pivot_table(index="bin", columns="model", values="d_q50", aggfunc="median")
          .map(lambda v: f"{v:+.1%}").to_string())
    print("\n=== rasipanje u najvisoj korpi (bimodalnost?) ===")
    v = d[d.bin.str.startswith("4.0")]
    if len(v):
        print(v.groupby("model").d_q99.describe()[["count", "min", "50%", "max"]].round(3)
              .to_string())


if __name__ == "__main__":
    main()
