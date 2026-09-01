# -*- coding: utf-8 -*-
"""
Referenca za oblik na stvarnim podacima: isti instrument kao kod modela, i koliko
od izabranog gradijenta pravi sam izbor.

DVE RUPE KOJE SE OVDE ZATVARAJU
-------------------------------
1. INSTRUMENT NIJE ISTI SA OBE STRANE. U `dissociation_real.py` referentno xi po
   tercilu dolazi iz Hillovog estimatora nad rezidualima y/s_hat, a modelovo xi
   iz inverzije odnosa Q(0,99)/Q(0,9). To su dva razlicita estimatora sa
   razlicitim pristrasnostima, a udeo se racuna kao njihov kolicnik. Na
   sintetickim podacima ta razlika je proverena naspram poznatog xi
   (`findings/h1/xi_residual.md`); na stvarnim podacima prave vrednosti nema, pa
   nije proverena nikako. Ovde se referenca racuna i INVERZIJOM ISTOG ODNOSA, iz
   empirijskih kvantila u tercilu, pa je poredjenje instrumentom upareno.

2. IZBOR OBELEZJA JE MAKSIMUM PREKO SVIH KOLONA. Obelezje se bira kao ono uz koje
   je |xi(t3) - xi(t1)| na fit delu najvece. Maksimum mnogo suma je i sam veliki:
   kada prave zavisnosti nema, izbor je vodjen greskom procene, pa referentni
   gradijent na test delu bude blizu nule i kolicnik eksplodira. To je izmereni
   raspon udela -2,80 do 6,16, koji se do sada objasnjavao samo kao "slab
   referentni gradijent", bez imenovanog uzroka. Ovde se meri i NULTA RASPODELA
   tog izbora: ista procedura nad permutovanim y, gde po konstrukciji nema
   nikakve zavisnosti repa od atributa.

Nijedan model se ne poziva. Sve je nad podacima, pa je cena minuti, ne sati.

KAKO CITATI
-----------
  ref_range_hill iznad null_p95              gradijent je jaci od izbora, skup broji
  ref_range_hill unutar nulte raspodele      za taj skup rangovna tvrdnja ne stoji ni
                                             po cemu osim po izboru
  hill i odnos istog znaka i reda velicine   razlika oblika nije artefakt instrumenta

Rezultati u `reference_instrument.csv`.

POKRETANJE
----------
    python -u experiments/h1_shape_vs_scale/reference_instrument.py
    DATASETS=diamonds,houses SEEDS=1 python -u reference_instrument.py
"""
import os
import time

import numpy as np
import pandas as pd

from common import append, datasets, metrics, paths, quiet

quiet.silence()

N_FIT, N_TEST = 3000, 6000
N_TERCILA = 3
MIN_PO_TERCILU = 400
HI, LO, MID = 0.99, 0.9, 0.5
N_PERM = int(os.environ.get("N_PERM", "50"))     # nulta raspodela izbora
N_BOOT = int(os.environ.get("N_BOOT", "200"))    # interval za referentni gradijent
SEEDOVA = int(os.environ.get("SEEDS", "3"))
PODRAZUMEVANI = ["OnlineNewsPopularity", "diamonds", "particulate-matter-ukair-2017",
                 "Buzzinsocialmedia_Twitter", "CPS1988", "218_house_8L",
                 "superconduct", "houses", "Allstate_Claims_Severity", "house_16H"]
SKUPOVI = os.environ.get("DATASETS", ",".join(PODRAZUMEVANI)).split(",")
OUT = os.environ.get("OUTPUT", "reference_instrument.csv")

KOLONE = ["dataset", "seed", "feature", "n_features",
          # referentni gradijent, tri instrumenta nad ISTIM tercilima
          "ref_range_hill", "ref_range_ratio", "ref_range_ratio_residual",
          # interval za Hillov, i nulta raspodela koju sam izbor obelezja pravi
          "boot_lo", "boot_hi", "null_p50", "null_p95", "above_null",
          "seconds", "reason"]
KEY = ["dataset", "seed"]


def skala_regresijom(Xf, yf):
    ok = yf > 0
    D = np.c_[np.ones(ok.sum()), Xf[ok]]
    b, *_ = np.linalg.lstsq(D, np.log(yf[ok]), rcond=None)
    return lambda X: np.clip(np.exp(np.c_[np.ones(len(X)), X] @ b), 1e-9, None)


def tercili(v, ivice):
    return np.clip(np.digitize(v, ivice[1:-1]), 0, N_TERCILA - 1)


def xi_hill(y, s_hat, terc):
    """Referenca kakva je i do sada: Hill nad rezidualima, po tercilu."""
    out = []
    for t in range(N_TERCILA):
        z = y[terc == t] / s_hat[terc == t]
        z = z[np.isfinite(z) & (z > 0)]
        out.append(metrics.hill(z) if len(z) >= MIN_PO_TERCILU else np.nan)
    return out


def xi_ratio(y, terc, residual=False):
    """Referenca ISTIM instrumentom koji se trazi od modela: inverzija odnosa.

    Kvantili su empirijski kvantili y u tercilu, pa je jedina razlika prema
    modelovoj strani to sto ih daje uzorak umesto modela.
    """
    out = []
    for t in range(N_TERCILA):
        z = y[terc == t]
        z = z[np.isfinite(z)]
        if len(z) < MIN_PO_TERCILU:
            out.append(np.nan)
            continue
        qh, ql, qm = np.quantile(z, [HI, LO, MID])
        if residual:
            out.append(metrics.xi_from_residual_ratio((qh - qm) / (ql - qm))
                       if ql - qm > 1e-12 else np.nan)
        else:
            out.append(metrics.xi_from_ratio(qh / ql) if ql > 1e-12 else np.nan)
    return out


def izaberi_obelezje(Xf, yf, s_hat_f):
    """Isti izbor kao u `dissociation_real.py`: maksimum |xi(t3) - xi(t1)|."""
    best, best_range, best_edges, scanned = None, -np.inf, None, 0
    for j in range(Xf.shape[1]):
        v = Xf[:, j]
        if len(np.unique(v)) < N_TERCILA * 3:
            continue
        iv = np.quantile(v, np.linspace(0, 1, N_TERCILA + 1))
        if len(np.unique(iv)) < N_TERCILA + 1:
            continue
        xs = xi_hill(yf, s_hat_f, tercili(v, iv))
        if any(not np.isfinite(x) for x in xs):
            continue
        scanned += 1
        r = abs(xs[2] - xs[0])
        if r > best_range:
            best, best_range, best_edges = j, r, iv
    return best, best_edges, scanned


def jedan(skup, seed, kes):
    if skup not in kes:
        kes[skup] = datasets.load(skup, kes["_ids"], positive_only=True)
    X, y = kes[skup]
    if len(y) < N_FIT + N_TEST:
        raise RuntimeError(f"premalo redova ({len(y)})")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: N_FIT + N_TEST]
    fi, ti = idx[:N_FIT], idx[N_FIT:]
    Xf, yf, Xte, yte = X[fi], y[fi], X[ti], y[ti]

    s_hat = skala_regresijom(Xf, yf)
    j, ivice, scanned = izaberi_obelezje(Xf, yf, s_hat(Xf))
    if j is None:
        raise RuntimeError("nijedno obelezje nema tri upotrebljiva tercila")

    terc = tercili(Xte[:, j], ivice)
    s_te = s_hat(Xte)

    h = xi_hill(yte, s_te, terc)
    a = xi_ratio(yte, terc)
    b = xi_ratio(yte, terc, residual=True)

    # Interval za referentni gradijent: bootstrap nad test redovima.
    boot = []
    for _ in range(N_BOOT):
        k = rng.integers(0, len(yte), len(yte))
        hb = xi_hill(yte[k], s_te[k], terc[k])
        boot.append(hb[2] - hb[0])
    boot = np.array([v for v in boot if np.isfinite(v)])

    # Nulta raspodela IZBORA: ista procedura nad permutovanim y, gde zavisnosti
    # repa od atributa nema po konstrukciji. Sve sto se tu dobije je posledica
    # uzimanja maksimuma preko kolona.
    null = []
    for _ in range(N_PERM):
        yp = rng.permutation(yf)
        s_p = skala_regresijom(Xf, yp)
        jp, ivp, _ = izaberi_obelezje(Xf, yp, s_p(Xf))
        if jp is None:
            continue
        hp = xi_hill(yte, s_te, tercili(Xte[:, jp], ivp))
        if np.isfinite(hp[2]) and np.isfinite(hp[0]):
            null.append(abs(hp[2] - hp[0]))
    null = np.array(null)

    ref = h[2] - h[0]
    return dict(
        feature=j, n_features=scanned,
        ref_range_hill=ref,
        ref_range_ratio=a[2] - a[0],
        ref_range_ratio_residual=b[2] - b[0],
        boot_lo=float(np.quantile(boot, 0.025)) if len(boot) else np.nan,
        boot_hi=float(np.quantile(boot, 0.975)) if len(boot) else np.nan,
        null_p50=float(np.median(null)) if len(null) else np.nan,
        null_p95=float(np.quantile(null, 0.95)) if len(null) else np.nan,
        above_null=(bool(abs(ref) > np.quantile(null, 0.95)) if len(null) else None),
        reason="")


def main():
    gotovi = append.done(OUT, KEY)
    if gotovi:
        print(f"resuming, {len(gotovi)} cells already measured", flush=True)
    kes = {"_ids": paths.load_json("sb_openml_ids.json")}
    t0 = time.time()
    for skup in SKUPOVI:
        for seed in [7000 + 1000 * i for i in range(SEEDOVA)]:
            k = dict(dataset=skup, seed=seed)
            if append.key(k, KEY) in gotovi:
                continue
            t1 = time.time()
            try:
                r = jedan(skup, seed, kes)
            except Exception as e:
                r = dict(reason=f"{type(e).__name__}: {e}"[:110])
            r.update(k, seconds=round(time.time() - t1, 1))
            append.write(OUT, r, KOLONE)
            poruka = r.get("reason") or (
                f"hill {r['ref_range_hill']:+.3f} "
                f"[{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}] | "
                f"odnos {r['ref_range_ratio']:+.3f} | "
                f"rezid {r['ref_range_ratio_residual']:+.3f} | "
                f"nulta p95 {r['null_p95']:.3f} -> "
                f"{'jaci' if r['above_null'] else 'U SUMU'}")
            print(f"  {skup:32s} s={seed}  {poruka}  [{r['seconds']}s]", flush=True)
    report(t0)


def report(t0=None):
    d = pd.read_csv(paths.result(OUT))
    d = d[d.reason.isna() | (d.reason.astype(str).str.strip() == "")]
    if d.empty:
        return
    print("\n=== referentni gradijent po instrumentu (medijana po skupu) ===")
    print(d.groupby("dataset")[["ref_range_hill", "ref_range_ratio",
                                "ref_range_ratio_residual", "null_p95"]]
          .median().round(3).to_string())
    print("\n=== skupovi ciji je gradijent jaci od nulte raspodele izbora ===")
    print(d.groupby("dataset").above_null.mean().round(2).to_string())
    print("\nSkupovi ispod 0,5 ovde ne nose rangovnu tvrdnju: kod njih je izabrani "
          "gradijent u opsegu koji sam izbor proizvodi iz suma.")
    if t0 is not None:
        print(f"\ntotal {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
