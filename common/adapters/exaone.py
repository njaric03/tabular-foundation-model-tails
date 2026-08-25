# -*- coding: utf-8 -*-
"""
Adapter za EXAONE-Tabular: izvlacenje kvantila iz glave sa 999 kvantila.

ZASTO
-----
EXAONE-Tabular (LG AI Research, avgust 2026) je na dan 17.8.2026. DRUGI na TabArena
rang-listi (Elo 1755 ukupno, 1883 na regresiji), sa ~21,1M parametara. Nema objavljen
rad -- samo model karta i kod. Njegov `regressor.py:341-349` radi istu stvar kao
ostala cetiri paketa:

    center_tensor = selected_y.mean()
    scale_tensor  = selected_y.std(correction=1)
    transformed   = (selected_y - center_tensor) / (scale_tensor + 1e-8)

Nerobusna standardizacija targeta, dok atributi idu kroz QuantileTransformer
(rangovna transformacija, po konstrukciji imuna na outliere). Model karta to kaze i
recima: "Targets are standardized against the fitted support set".

Ovaj fajl daje pristup kvantilima da bi se model mogao meriti istim alatom kao
TabICLv2, TabPFN i TabDPT.

DVA PROBLEMA KOJA SE OVDE RESAVAJU
----------------------------------
1. `predict()` vraca samo tacku (odsecena sredina po centralnih 99,8% kvantilne
   funkcije). Kvantilna banka od 999 nivoa se odbacuje u `_collapse_members`.
   Resenje: privremena zamena `_collapse_members` tako da vrati izabrani nivo.
   Ostatak putanje (`* scale + center`, tezine clanova ansambla) ostaje netaknut,
   pa je izlaz de-normalizovan tacno kako to model radi.

2. `from_pretrained()` PADA na torch 2.13:

       ValueError: checkpoint quantile levels do not match the model

   Uzrok nije neslaganje modela nego float32 zaokruzivanje: checkpoint nosi
   `quantile_levels`, a glava ih gradi sa `torch.linspace(1/1000, 999/1000, 999)`.
   240 od 999 nivoa se razlikuje za **jedan ULP** (max |razlika| = 5,96e-08), a
   `checkpoint.py:295` poredi sa `torch.equal`, koje je bitovski tacno.
   Resenje: `torch.equal` se privremeno zamenjuje verzijom koja, kad tacno poredjenje
   padne, dozvoljava odstupanje do 1e-6 uz isti oblik. Nista drugo se ne menja.

NIVOI
-----
Checkpoint nosi tau_i = (i+1)/1000 za i = 0..998, provereno citanjem safetensors fajla.
Dakle Q(0,5) -> 499, Q(0,9) -> 899, Q(0,99) -> 989, Q(0,999) -> 998.

UPOTREBA
--------
    from common.adapters import exaone

    m = exaone.napravi(seed=0)
    m.fit(Xtr, ytr)
    Q = exaone.kvantili(m, Xte, [0.5, 0.9, 0.99])   # (n_test, 3)
"""
import contextlib
import os

import numpy as np
import torch


QUANT_COUNT = 999


def indeks_nivoa(tau: float) -> int:
    """tau_i = (i+1)/1000 -> indeks u banci od 999 kvantila."""
    i = int(round(tau * (QUANT_COUNT + 1))) - 1
    if not 0 <= i < QUANT_COUNT:
        raise ValueError(f"tau={tau} van mreze od {QUANT_COUNT} nivoa")
    return i


@contextlib.contextmanager
def _blaga_jednakost(atol: float = 1e-6):
    """`torch.equal` uz toleranciju od jednog ULP-a, samo za trajanje ucitavanja.

    Vidi docstring: bez ovoga `from_pretrained` pada na torch 2.13.
    Tacno poredjenje se i dalje pokusava prvo; tolerancija se koristi tek kad ono
    padne, i to samo za tenzore istog oblika i tipa.
    """
    orig = torch.equal

    def blago(a, b):
        if orig(a, b):
            return True
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return bool((a - b).abs().max() <= atol)

    torch.equal = blago
    try:
        yield
    finally:
        torch.equal = orig


def napravi(seed: int = 0, n_est: int | None = None, device: str = "cpu",
            dtype: str | None = None):
    """Ucitan EXAONETabularRegressor. Seed ne ulazi u tezine -- model je pretreniran;
    on odredjuje permutacije clanova ansambla, isto kao `random_state` kod ostalih.

    DTYPE. Manifest pina `compute_dtype="float16"`, jer je model pisan za CUDA
    ("a CUDA GPU is strongly recommended -- the model uses fused attention kernels
    and half precision"). Na CPU torch nema brze fp16 kernele nego ih emulira, pa je
    inferencija reda velicine sporija. Podrazumevano se zato na CPU trazi
    **float32**: to je preciznije, ne manje precizno, ali JESTE odstupanje od
    isporucene konfiguracije i tako mora biti zapisano u radu.
    Za merenje u isporucenoj konfiguraciji: `DTYPE=float16`.

    N_EST. Manifest pina `ensemble_count=8`. Ostala merenja u repozitorijumu koriste
    1 ili 4 (`findings/h1/vincentization.md`), pa je knob izlozen radi uporedivosti.
    """
    from exaonetabular import EXAONETabularRegressor
    if dtype is None:
        dtype = os.environ.get("DTYPE") or ("float32" if str(device) == "cpu" else None)
    with _blaga_jednakost():
        return EXAONETabularRegressor.from_pretrained(
            device=device, seed=seed, ensemble_count=n_est, compute_dtype=dtype)


def kvantili(model, X, tau_lista):
    """(n_test, len(tau_lista)) predvidjeni kvantili, na originalnoj skali targeta.

    Radi tako sto `_collapse_members` privremeno vraca izabrane nivoe umesto odsecene
    sredine. Banka se pre indeksiranja sortira -- isto sto model radi u svojoj
    `trimmed` grani, kao zastita od ukrstanja kvantila. Ostatak `predict` putanje
    (`* scale + center`, spajanje clanova) ostaje netaknut.

    BRZI PUT. Kad su tezine clanova `None` -- a jesu ispod ~12.000 redova konteksta,
    jer NNLS trazi 2.000 izdvojenih redova -- `predict` clanove spaja sa `.mean(dim=0)`,
    sto radi i na tenzoru oblika (clanovi, redovi, K). Tada svi nivoi izlaze iz JEDNOG
    prolaza. Sa fitovanim tezinama blend se ne emituje preko trece ose, pa se pada
    nazad na jedan prolaz po nivou.
    """
    import exaonetabular.regressor as R

    X = np.asarray(X, dtype=np.float64)
    idx = [indeks_nivoa(t) for t in tau_lista]
    brzo = model._state().get("member_weights") is None and not os.environ.get("POJEDINACNO")
    orig = R.EXAONETabularRegressor._collapse_members

    def napravi_kolaps(izbor):
        def kolaps(self, output, query_count):
            ocek = (self.manifest.runtime.ensemble_count, query_count,
                    self.manifest.output_width)
            if tuple(output.shape) != ocek or not bool(torch.isfinite(output).all()):
                raise RuntimeError("model vratio nevalidne kvantile")
            return torch.sort(output.float(), dim=-1).values[..., izbor]
        return kolaps

    try:
        if brzo:
            R.EXAONETabularRegressor._collapse_members = napravi_kolaps(idx)
            Q = np.asarray(model.predict(X), dtype=np.float64)
            return Q.reshape(len(X), len(idx))
        izlaz = []
        for i in idx:
            R.EXAONETabularRegressor._collapse_members = napravi_kolaps(i)
            izlaz.append(np.asarray(model.predict(X), dtype=np.float64))
        return np.column_stack(izlaz)
    finally:
        R.EXAONETabularRegressor._collapse_members = orig


def parametara(model) -> int:
    return int(sum(p.numel() for p in model.model.parameters()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = int(os.environ.get("N", "800"))
    X = rng.normal(size=(n, 5))
    y = np.exp(0.6 * X[:, 0]) * ((1 - rng.random(n)) ** (-0.7) - 1) / 0.7
    Xte = rng.normal(size=(200, 5))

    m = napravi()
    print(f"parametara: {parametara(m):,}")
    print(f"ensemble_count={m.manifest.runtime.ensemble_count}  "
          f"quantile_count={m.manifest.regression.quantile_count}  "
          f"point_estimate={m.manifest.regression.point_estimate}")
    m.fit(X, y)
    Q = kvantili(m, Xte, [0.5, 0.9, 0.99, 0.999])
    tacka = m.predict(Xte)
    print("monotonost Q50<=Q90<=Q99<=Q999:",
          bool(np.all(np.diff(Q, axis=1) >= -1e-9)))
    print("medijane:", np.round(np.median(Q, axis=0), 3),
          " tacka:", round(float(np.median(tacka)), 3))
    print("empirijski y:", np.round(np.quantile(y, [0.5, 0.9, 0.99, 0.999]), 3))
