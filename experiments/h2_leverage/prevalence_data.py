# -*- coding: utf-8 -*-
"""
FAZA 1 studije prevalencije: koliko cesto u stvarnim tabelama UOPSTE postoji
tacka sa dovoljnom polugom.

ZASTO
-----
Najlaksi prigovor na ceo nalaz o nerobusnoj standardizaciji glasi: "intervencija
od 100 x max je vestacka". `influence_real.py` je to i potvrdio -- na
sest skupova poluga >= 100x se javlja u 0% uzoraka.

Odgovor nije braniti intervenciju nego IZMERITI baznu stopu. Pitanje koje ovaj
skript resava je cisto pitanje o PODACIMA i ne trazi nijedan model:

    U kom udelu stvarnih regresionih tabela postoji podskup u kome jedna tacka
    dominira empirijskom standardnom devijacijom?

Uzrocna velicina nije odnos max/drugi (to je pokazano u findings/NALAZI.md 4.2) nego
POMERAJ sd. Zato se meri i jedno i drugo, a prag se postavlja na sd.

DIZAJN
------
Za svaki skup iz `sb_datasets.json` (102 skupa iz ScoringBench-a):
  * izvuci UZORAKA podskupova velicine N_FIT (isto kao u eksperimentima sa modelima)
  * u svakom: leverage = max/drugi, pomeraj_sd = sd(sa max) / sd(bez max)
  * agregiraj: medijana i P(pomeraj_sd >= prag) za pragove 1.5 / 2 / 4 / 10

Faza 2 (`prevalence_models.py`) pusta modele SAMO na skupovima koje
ova faza oznaci kao rizicne. Time se sto skupova x tri modela svede na izvodljivo.

IZLAZ
-----
`prevalence_data.csv`, jedan red po skupu. Nastavljiv -- vec uradjeni skupovi
se preskacu.

POKRETANJE
----------
    python -u prevalence_data.py
    N_SUBSAMPLES=50 DATASETS=freMTPL2sev,Allstate_Claims_Severity python -u prevalence_data.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, metrics, models, paths, quiet

quiet.silence()


UZORAKA = int(os.environ.get("N_SUBSAMPLES", "40"))
N_FIT = int(os.environ.get("N_FIT", "2000"))
PRAGOVI = [1.5, 2.0, 4.0, 10.0]
OUT = os.environ.get("OUTPUT", "prevalence_data.csv")

_svi = paths.load_json("sb_datasets.json")
SKUPOVI = os.environ.get("DATASETS", "").split(",") if os.environ.get("DATASETS") else _svi

KOLONE = (["dataset", "n_rows", "n_subsamples", "leverage_median", "leverage_p90",
           "sd_shift_median", "sd_shift_max"]
          + [f"p_sd_{p}" for p in PRAGOVI]
          + [f"p_leverage_{p}" for p in (3, 5, 10, 100)]
          + ["hill_xi", "n_fit", "seconds", "reason"])
# n_fit is a knob that changes the result, so it is a column and part of the key
# (rule 2 in RULES.md); UZORAKA is already recorded as n_subsamples.
KEY = ["dataset", "n_fit"]


def jedan(skup, ids):
    X, y = datasets.load(skup, ids)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    y = y[ok]
    if len(y) < 300:
        raise RuntimeError(f"premalo redova ({len(y)})")

    n = min(N_FIT, len(y) // 2)
    rng = np.random.default_rng(20260824)
    poluge, pomeraji = [], []
    for _ in range(UZORAKA):
        u = y[rng.permutation(len(y))[:n]]
        p = np.sort(u)
        drugi = max(abs(p[-2]), 1e-12)
        poluge.append(abs(p[-1]) / drugi)
        sd_sa = np.std(u)
        sd_bez = np.std(p[:-1])
        pomeraji.append(sd_sa / max(sd_bez, 1e-12))

    poluge = np.asarray(poluge)
    pomeraji = np.asarray(pomeraji)
    r = dict(dataset=skup, n_rows=len(y), n_subsamples=UZORAKA, n_fit=N_FIT,
             leverage_median=float(np.median(poluge)),
             leverage_p90=float(np.quantile(poluge, 0.9)),
             sd_shift_median=float(np.median(pomeraji)),
             sd_shift_max=float(pomeraji.max()),
             hill_xi=metrics.hill(y), reason="")
    for p in PRAGOVI:
        r[f"p_sd_{p}"] = float((pomeraji >= p).mean())
    for p in (3, 5, 10, 100):
        r[f"p_leverage_{p}"] = float((poluge >= p).mean())
    return r


def izvestaj(d):
    d = d[d.reason.isna() | (d.reason == "")]
    if not len(d):
        print("nema uspesnih skupova")
        return
    print(f"\n=== PREVALENCIJA na {len(d)} skupova, {UZORAKA} podskupova po skupu ===")
    for p in PRAGOVI:
        kol = f"p_sd_{p}"
        ima = (d[kol] > 0).sum()
        cest = (d[kol] >= 0.05).sum()
        print(f"  pomeraj sd >= {p:>4}x : postoji u {ima:>3}/{len(d)} skupova "
              f"({ima/len(d):5.1%}) | u >=5% uzoraka: {cest:>3} skupova")
    print("\n  --- za poredjenje, stara mera (poluga max/drugi) ---")
    for p in (3, 5, 10, 100):
        kol = f"p_leverage_{p}"
        ima = (d[kol] > 0).sum()
        print(f"  poluga >= {p:>4}x     : postoji u {ima:>3}/{len(d)} skupova ({ima/len(d):5.1%})")

    riz = d[d[f"p_sd_{PRAGOVI[2]}"] > 0].sort_values(f"p_sd_{PRAGOVI[2]}", ascending=False)
    print(f"\n=== RIZICNI SKUPOVI (pomeraj sd >= {PRAGOVI[2]}x bar jednom), ulaz u fazu 2 ===")
    if len(riz):
        print(riz[["dataset", "n_rows", "sd_shift_median", "sd_shift_max",
                   f"p_sd_{PRAGOVI[2]}", "hill_xi"]].to_string(index=False))
        # paths.data, ne golo ime: bez toga fajl zavrsi u direktorijumu iz kog je
        # skripta pokrenuta, pored `data/prevalence_at_risk.json` koji je vec
        # tamo, pa dve kopije iste liste zive na dva mesta i razilaze se.
        izlaz = paths.data("prevalence_at_risk.json")
        izlaz.parent.mkdir(parents=True, exist_ok=True)
        with open(izlaz, "w", encoding="utf-8") as f:
            json.dump(list(riz.dataset), f, ensure_ascii=False, indent=1)
        print(f"\n  -> {len(riz)} skupova upisano u {izlaz.relative_to(paths.ROOT)}")
    else:
        print("  nijedan, to je samo po sebi nalaz, i to jak")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} datasets already measured", flush=True)
    ids = paths.load_json("sb_openml_ids.json")
    t0 = time.time()
    for i, skup in enumerate(SKUPOVI, 1):
        if append.key(dict(dataset=skup, n_fit=N_FIT), KEY) in gotovi:
            continue
        t1 = time.time()
        try:
            r = jedan(skup, ids)
        except Exception as e:
            r = dict(dataset=skup, reason=f"{type(e).__name__}: {e}"[:110])
        r["seconds"] = round(time.time() - t1, 1)
        append.write(OUT, r, KOLONE)
        if r.get("reason"):
            print(f"[{i:>3}/{len(SKUPOVI)}] {skup[:34]:<35} PAO: {r['reason'][:60]}", flush=True)
        else:
            print(f"[{i:>3}/{len(SKUPOVI)}] {skup[:34]:<35} n={r['n_rows']:>7} "
                  f"sd_pomeraj med {r['sd_shift_median']:.2f} max {r['sd_shift_max']:6.2f} "
                  f"| P(sd>=4x) {r['p_sd_4.0']:.0%} | xi_hill {r['hill_xi']:.2f} "
                  f"[{r['seconds']}s]", flush=True)

    izvestaj(pd.read_csv(paths.result(OUT)))
    print(f"\nukupno {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
