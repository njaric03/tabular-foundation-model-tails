# -*- coding: utf-8 -*-
"""
Koliko je ispravka sejanja pomerila TabPFN brojke.

Pusta se posle `run/run_tabpfn_reseed.sh`. Poredi stari CSV (TabPFN sa `random_state=0`,
dakle sa zamrznutim permutacijama ansambla kroz sve seedove) i novi (`random_state=seed`).

Gleda se dvoje, i drugo je bar jednako vazno kao prvo:

  UDEO  tacka -- koliki deo prave promene model uhvati
  SD    rasipanje po seedovima -- ono sto je zamrzavanje najdirektnije potiskivalo

Ako se UDEO jedva pomeri a SD skoci, nalaz o TabPFN-u prezivljava, ali svaka ograda tipa
"+-" uz njega mora da se preracuna. Ako se pomeri i UDEO, menja se §3.1 u `findings/NALAZI.md`.

CETVRTI RED NIJE POREDJENJE SEJANJA. `scale_of_x.csv` nema kolonu `n_est`, pa je
dugo bio neupotrebljiv kao osnova. Utvrdjeno 27.8. posredno: kolone `med_prava` i
`xi_prosek` zavise samo od podataka i identicne su 60/60 u oba nova fajla, sto redove
poravnava; a `med_model` se od starog fajla razlikuje za medijanih 0,21% pri n_est=4 i
4,16% pri n_est=1. Stara osnova je dakle merena na n_est=4. Zato cetvrti red poredi
n_est=4 sa n_est=1 -- to je efekat velicine ansambla (v. `findings/h1/vincentization.md`), ne
sejanja, i tako je i obelezen.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common import paths

# (opis, stari fajl, novi fajl, vrsta, napomena)
# Napomena != "" znaci da red NIJE poredjenje sejanja i da se tako ne sme citati.
PAROVI = [
    ("oblik  n_est=4, 20 seedova", "shape_of_x_20.csv",
     "shape_of_x_tabpfn_reseed_20.csv", "oblik", ""),
    ("oblik  n_est=1, 20 seedova", "shape_of_x_nest1.csv",
     "shape_of_x_tabpfn_reseed_nest1.csv", "oblik", ""),
    ("skala  n_est=4, 20 seedova", "scale_of_x.csv",
     "scale_of_x_tabpfn_reseed_20.csv", "skala", ""),
    ("skala  n_est=1, 20 seedova", "scale_of_x.csv",
     "scale_of_x_tabpfn_reseed_nest1.csv", "skala",
     "osnova je n_est=4, pa ovo meri velicinu ansambla, ne sejanje"),
]

MODEL = "TabPFN-V3"


def udeli(putanja: str, vrsta: str) -> pd.Series | None:
    """Udeo prave promene koji model uhvati, po seedu."""
    put = paths.result(putanja)
    if not put.exists():
        return None
    d = pd.read_csv(put)
    d = d[d.model == MODEL]
    if d.empty:
        return None
    if vrsta == "oblik":
        # nagib impliciranog xi kroz tercile, deljen pravim nagibom
        p = d.pivot_table(index="seed", columns="tercile",
                          values=["xi_implied", "xi_true"])
        nm = p[("xi_implied", 3)] - p[("xi_implied", 1)]
        npr = p[("xi_true", 3)] - p[("xi_true", 1)]
    else:
        # log-odnos medijane izmedju 3. i 1. tercila, deljen pravim
        p = d.pivot_table(index="seed", columns="tercile",
                          values=["median_model", "median_true"])
        nm = np.log(p[("median_model", 3)] / p[("median_model", 1)])
        npr = np.log(p[("median_true", 3)] / p[("median_true", 1)])
    return (nm / npr).replace([np.inf, -np.inf], np.nan).dropna()


def main() -> int:
    print(f"=== {MODEL}: random_state=0 (staro) naspram random_state=seed (novo) ===\n")
    print(f"{'merenje':30s}{'udeo staro':>12s}{'udeo novo':>11s}{'difference':>10s}"
          f"{'sd staro':>10s}{'sd novo':>9s}")
    print("-" * 82)

    ima_novih = False
    napomene = []
    for opis, stari, novi, vrsta, napomena in PAROVI:
        s, n = udeli(stari, vrsta), udeli(novi, vrsta)
        if s is None:
            print(f"{opis:30s}{'(nema starog)':>12s}")
            continue
        if n is None:
            print(f"{opis:30s}{s.mean():>11.0%}{'--':>11s}{'--':>10s}"
                  f"{s.std():>10.2f}{'--':>9s}   merenje jos nije pusteno")
            continue
        ima_novih = True
        zvezda = ""
        if napomena:
            napomene.append(napomena)
            zvezda = f"  ({len(napomene)})"
        print(f"{opis:30s}{s.mean():>11.0%}{n.mean():>11.0%}"
              f"{n.mean()-s.mean():>+10.0%}{s.std():>10.2f}{n.std():>9.2f}{zvezda}")

    for i, napomena in enumerate(napomene, 1):
        print(f"\n  ({i}) NE CITATI KAO SEJANJE: {napomena}.")

    if not ima_novih:
        print(f"\nNijedno ponovljeno merenje jos nije zavrseno. Pusti `run/run_tabpfn_reseed.sh`.")
        return 0

    print("\nKAKO CITATI")
    print("  udeo se jedva pomerio, sd skocio  -> nalaz stoji, ali se ograde")
    print("                                       preracunavaju (rasipanje je bilo lazno usko)")
    print("  pomerio se i udeo                 -> findings/NALAZI.md §3.1 i disocijacija se menjaju")
    print("  nista se nije pomerilo            -> zamrzavanje nije bilo bitno za ovu meru;")
    print("                                       zapisati kao proverenu ogradu, ne cutati")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
