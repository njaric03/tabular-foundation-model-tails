# -*- coding: utf-8 -*-
"""
Glavni grafik teze: disocijacija nivoa i oblika.

STA POKAZUJE
------------
Za svaki model dve brojke, obe kao UDEO PRAVE PROMENE koji model uhvati:

    x  SKALA   prati li model s(x)   -- log-odnos predvidjene medijane kroz tercile po X@W
    y  OBLIK   prati li model xi(x)  -- nagib impliciranog xi kroz tercile po x4

Generator je konstruisan tako da su te dve stvari razdvojene: `x4` ima tezinu NULA u
skali, pa sve sto model uhvati o repu preko `x4` jeste prilagodjavanje oblika, a ne
posledica toga sto je negde skala veca.

Dijagonala y = x je "nema disocijacije": model koji oblik prati jednako dobro kao nivo
lezi na njoj. Sve sto je ISPOD dijagonale je disocijacija, a vertikalno rastojanje do nje
je njena velicina.

Tvrdnja teze je da tabelarni fundacioni modeli sede desno-dole: skala skoro savrseno,
oblik zamrznut. Grafik postoji da se to vidi bez citanja tabele.

STRELICE
--------
Za tri modela koja imaju i merenje na `n_est=1` crta se strelica od vrednosti pri
n_est=4 do vrednosti pri n_est=1, po osi OBLIKA. Vincentizacija (usrednjavanje clanova
ansambla) spusta implicirano xi, pa ansambl sam po sebi pojacava prividnu disocijaciju.
Kod TabPFN-a je to 42% -> 19%. Bez toga bi grafik tvrdio vise nego sto podaci nose.

NEIZVESNOST
-----------
Trake su STANDARDNA GRESKA SREDINE (sd / sqrt(broj seedova)), ne sd. Pitanje na koje
grafik odgovara je "koliko pouzdano znamo taj udeo", a ne "koliko rasipa po seedovima".
Oba broja stoje u tabeli koju skripta ispise.

POKRETANJE
----------
    python -u plot_dissociation.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import paths

# --- paleta ---------------------------------------------------------------
# Forma je dumbbell (dve mere po stavci), za koju je pravilo "jedna nijansa, dva
# stepena". Stepeni 250 i 550 plave rampe, provereni validatorom:
#   node validate_palette.js "#86b6ef,#1c5cab" --mode light --ordinal -> ALL PASS
#   (monotono po svetlini, DL >= 0.06, svetli kraj 2,06:1 prema povrsini)
# Identitet modela nosi red na osi, ne boja, pa ogranicenje od tri slota ne vazi.
BOJA_SKALA = "#86b6ef"     # mera koja je vec dobra -- povlaci se
BOJA_OBLIK = "#1c5cab"     # nalaz -- nosi naglasak
POVRSINA = "#fcfcfb"
MASTILO = "#0b0b0b"
MASTILO_2 = "#52514e"
MASTILO_3 = "#898781"      # muted (ose, oznake)
MREZA = "#e1e0d9"
OSNOVA = "#c3c2b7"

ETALONI = {"GBM", "XGB"}
REDOSLED = ["GBM", "TabICLv2", "EXAONE", "TabPFN-V3", "XGB"]

OBLIK_N4 = ["shape_of_x_20.csv", "shape_of_x_exaone_20.csv"]
OBLIK_N1 = ["shape_of_x_nest1.csv", "shape_of_x_exaone_nest1.csv"]
SKALA_N4 = ["scale_of_x.csv", "scale_of_x_exaone_20.csv"]


def _ucitaj(fajlovi: list[str]) -> pd.DataFrame:
    putanje = [paths.result(f) for f in fajlovi]
    delovi = [pd.read_csv(p) for p in putanje if p.exists()]
    if len(delovi) < len(putanje):
        fale = [p.name for p in putanje if not p.exists()]
        raise FileNotFoundError(f"nedostaju rezultati: {fale}")
    return pd.concat(delovi, ignore_index=True) if delovi else pd.DataFrame()


def udeo_oblika(d: pd.DataFrame) -> pd.Series:
    """Nagib impliciranog xi kroz tercile, deljen pravim nagibom. Po (model, seed)."""
    if d.empty:
        return pd.Series(dtype=float)
    p = d.pivot_table(index=["model", "seed"], columns="tercile",
                      values=["xi_implied", "xi_true"])
    return ((p[("xi_implied", 3)] - p[("xi_implied", 1)]) /
            (p[("xi_true", 3)] - p[("xi_true", 1)]))


def udeo_skale(d: pd.DataFrame) -> pd.Series:
    """Log-odnos medijane izmedju 3. i 1. tercila, deljen pravim. Po (model, seed)."""
    if d.empty:
        return pd.Series(dtype=float)
    p = d.pivot_table(index=["model", "seed"], columns="tercile",
                      values=["median_model", "median_true"])
    return (np.log(p[("median_model", 3)] / p[("median_model", 1)]) /
            np.log(p[("median_true", 3)] / p[("median_true", 1)]))


def saberi(s: pd.Series) -> pd.DataFrame:
    """Po modelu: sredina, sd po seedovima, standardna greska sredine, broj seedova."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    g = s.groupby(level=0)
    return pd.DataFrame({"sred": g.mean(), "sd": g.std(),
                         "se": g.std() / np.sqrt(g.count()), "n": g.count()})


def main() -> int:
    o4, o1, s4 = (saberi(f(_ucitaj(fs))) for f, fs in
                  ((udeo_oblika, OBLIK_N4), (udeo_oblika, OBLIK_N1), (udeo_skale, SKALA_N4)))
    modeli = [m for m in REDOSLED if m in o4.index and m in s4.index]
    if not modeli:
        print("nema podataka -- treba i xi-od-x-* i skala-od-x-*")
        return 1

    print("=== udeo prave promene koji model uhvati (n_est=4, 20 seedova) ===\n")
    print(f"{'model':<12}{'SKALA':>8}{'+-se':>7}{'OBLIK':>9}{'+-se':>7}{'sd oblik':>10}"
          f"{'oblik n_est=1':>15}")
    for m in modeli:
        n1 = f"{o1.loc[m, 'sred']:.0%}" if m in o1.index else "--"
        print(f"{m:<12}{s4.loc[m,'sred']:>8.0%}{s4.loc[m,'se']:>7.2f}"
              f"{o4.loc[m,'sred']:>9.0%}{o4.loc[m,'se']:>7.2f}{o4.loc[m,'sd']:>10.2f}{n1:>15}")

    # ------------------------------------------------------------------ crtanje
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": OSNOVA, "axes.linewidth": 0.8,
        "xtick.color": MASTILO_3, "ytick.color": MASTILO,
        "figure.facecolor": POVRSINA, "axes.facecolor": POVRSINA,
    })
    fig, ax = plt.subplots(figsize=(8.0, 0.82 * len(modeli) + 2.5))
    yy = list(range(len(modeli)))[::-1]          # prvi model gore

    for y, m in zip(yy, modeli):
        xs, xo = s4.loc[m, "sred"], o4.loc[m, "sred"]
        # spona = disocijacija; njena duzina JE nalaz
        ax.plot([xo, xs], [y, y], color=BOJA_SKALA, lw=3.2, solid_capstyle="round",
                zorder=2, alpha=0.55)
        # neizvesnost sredine
        for x, boja in ((xs, BOJA_SKALA), (xo, BOJA_OBLIK)):
            se = (s4 if boja == BOJA_SKALA else o4).loc[m, "se"]
            ax.plot([x - se, x + se], [y, y], color=boja, lw=1.2, alpha=0.9, zorder=3)
        # tacka pri n_est=1: koliko je od jaza posledica ansambla, a ne modela
        if m in o1.index and abs(o1.loc[m, "sred"] - xo) > 0.02:
            x1 = o1.loc[m, "sred"]
            ax.annotate("", xy=(x1, y), xytext=(xo, y),
                        arrowprops=dict(arrowstyle="-|>", color=MASTILO_3, lw=1.0,
                                        alpha=0.55, shrinkA=8, shrinkB=1))
            ax.plot(x1, y, marker="o", ms=7, mfc=POVRSINA, mec=MASTILO_3, mew=1.3, zorder=4)
            ax.annotate(f"{x1:.0%} pri n_est=1", (x1, y), textcoords="offset points",
                        xytext=(0, 13), fontsize=8.2, color=MASTILO_3, ha="center")
        # krajevi: 2px prsten u boji povrsine
        ax.plot(xo, y, marker="o", ms=11, color=BOJA_OBLIK, mec=POVRSINA, mew=2, zorder=5)
        ax.plot(xs, y, marker="o", ms=11, color=BOJA_SKALA, mec=POVRSINA, mew=2, zorder=5)
        ax.annotate(f"{xo:.0%}", (xo, y), textcoords="offset points", xytext=(-11, -3.5),
                    fontsize=9.5, color=BOJA_OBLIK, ha="right", weight="bold")
        ax.annotate(f"{xs:.0%}", (xs, y), textcoords="offset points", xytext=(11, -3.5),
                    fontsize=9.5, color=MASTILO_2, ha="left")

    ax.set_yticks(yy)
    ax.set_yticklabels([m + ("  (etalon)" if m in ETALONI else "") for m in modeli],
                       fontsize=10.5)
    ax.set_ylim(-0.7, len(modeli) - 0.3)
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.axvline(1.0, color=OSNOVA, lw=1.0, ls="--", zorder=1)
    ax.annotate("savršeno praćenje", (1.0, len(modeli) - 0.45), fontsize=8.5,
                color=MASTILO_3, ha="center", va="bottom")
    ax.grid(True, axis="x", color=MREZA, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel("udeo prave promene koji model uhvati", color=MASTILO_2, labelpad=8)
    fig.suptitle("Modeli prate nivo, oblik repa im je zamrznut",
                 x=0.012, y=0.985, ha="left", fontsize=13, color=MASTILO)
    fig.text(0.012, 0.925,
             "svaka spona je jaz između praćenja SKALE s(x) i praćenja OBLIKA repa ξ(x); "
             "20 seedova, n_est=4, crte = standardna greška sredine",
             ha="left", fontsize=8.8, color=MASTILO_2)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BOJA_OBLIK, mec=POVRSINA, mew=1.5,
               ms=9, label="OBLIK repa  ξ(x)"),
        Line2D([], [], marker="o", ls="", color=BOJA_SKALA, mec=POVRSINA, mew=1.5,
               ms=9, label="SKALA  s(x)"),
        Line2D([], [], marker="o", ls="", mfc=POVRSINA, mec=MASTILO_3, mew=1.3,
               ms=7, label="oblik pri n_est=1 (bez vincentizacije)"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False,
        fontsize=9, handletextpad=0.5, columnspacing=1.8)

    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    for ext in ("pdf", "png"):
        put = paths.figure(f"dissociation.{ext}")
        fig.savefig(put, dpi=200, bbox_inches="tight", facecolor=POVRSINA)
        print(f"  -> {put}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
