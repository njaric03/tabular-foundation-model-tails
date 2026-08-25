# -*- coding: utf-8 -*-
"""Serbian to English names for CSV columns, environment knobs and files.

Used once, by the migration that renamed everything, and kept afterwards so the
old names in the archive and in the Serbian findings can still be matched to the
current ones. Nothing in the experiments imports this at run time.
"""

COLUMNS = {
    "tercil": "tercile", "xi_pravo": "xi_true", "udeo_oblik": "shape_share",
    "udeo_skala": "scale_share", "sekundi": "seconds", "razlog": "reason",
    "skup": "dataset", "doza": "dose", "pozicija": "position",
    "varijanta": "variant", "ponavljanje": "repeat", "ponovak": "repeat",
    "korpa": "bin", "poluga": "leverage", "sd_pomeraj": "sd_shift",
    "med_prava": "median_true", "med_model": "median_model",
    "xi_prosek": "xi_mean", "familija": "family", "obelezje": "feature",
    "raspon_ref": "ref_range", "neslaganje_cv": "disagreement_cv",
    "prag_std": "threshold_std", "udeo_iznad_praga": "share_above_threshold",
    "granica_pokrivenost": "bound_coverage", "uticaj_sredina": "mean_influence",
    "n_redova": "n_rows", "uzoraka": "n_subsamples", "pomeren": "shifted",
    "raspon_procena": "estimate_range", "raspon_prag": "range_threshold",
    "udeo_vezanih": "tied_share", "prolazi": "passes", "c_odnos": "c_ratio",
    "udeo_nepoz": "unknown_share", "n_po_grupi": "n_per_group",
    "transformacija": "transform", "odnos_prava": "ratio_true",
    "odnos_odsec": "ratio_truncated", "odnos_skala": "scale_ratio",
    "xi_sirovo": "xi_raw", "xi_transf": "xi_transformed",
    "odstupanje": "deviation", "razlika": "difference",
    "xi_implied": "xi_implied", "ratio_model": "ratio_model",
    "n_estimators": "n_estimators", "n_train": "n_train", "n_est": "n_est",
    "xi_jedan": "xi_single", "xi_vincent": "xi_vincentized",
    "xi_mesavina": "xi_mixture", "max_y": "max_y",
    "empir": "empirical", "xi_evt_prost": "xi_evt_simple",
    "xi_kvantil": "xi_quantile",
    "A_kalem": "A_grafted", "B_kalem": "B_grafted",
    "A_sirov": "A_raw", "B_sirov": "B_raw",
    "gdev_izotona": "gdev_isotonic", "gdev_konstanta": "gdev_constant",
    "gdev_odnos_proseka": "gdev_mean_ratio", "gdev_sirov": "gdev_raw",
    "rmse_izotona": "rmse_isotonic", "rmse_konstanta": "rmse_constant",
    "rmse_odnos_proseka": "rmse_mean_ratio", "rmse_sirov": "rmse_raw",
    "mu_odnos": "mu_ratio", "n_prekoracenja": "n_exceedances",
    "opis": "description", "kvantil": "quantile", "sredina": "mean",
    "poluga_med": "leverage_median", "poluga_p90": "leverage_p90",
    "sd_pomeraj_med": "sd_shift_median", "sd_pomeraj_max": "sd_shift_max",
    "slaganje_raspon": "agreement_range", "xi_ref_raspon": "xi_ref_range",
    "udeo_ispod_060": "share_below_060", "udeo_jedinstvenih": "unique_share",
    "udeo_med_nepoz": "median_unknown_share",
    "udeo_oblik_kontrola": "shape_share_control",
    "udeo_oblik_rezid": "shape_share_residual",
    "xi_bez": "xi_without", "xi_sa": "xi_with", "d_pravo": "d_true",
    "r_orakl_med": "r_oracle_median", "r_uzorak_med": "r_sample_median",
    "r_uzorak_q10": "r_sample_q10", "r_uzorak_q90": "r_sample_q90",
    "val_kaze_kalemi": "val_says_graft",
}

# Prefixes, applied after the exact names above.
PREFIXES = {
    "odgovor_": "response_", "odnos_": "ratio_", "gr_": "grad_",
    "xi_rezid": "xi_residual", "xi_sirovi": "xi_raw", "xi_pravo": "xi_true",
    "d_q": "dq", "k_q": "kq", "oraklS_": "oracle_s_", "oraklR_": "oracle_r_",
    "empir_": "empirical_", "odst_": "deviation_",
    "lok_q": "local_q", "glob_q": "global_q",
    "p_poluga_": "p_leverage_", "p_sd_": "p_sd_", "pravo_": "true_",
    "test_kalem_": "test_grafted_", "test_sirov_": "test_raw_",
    "val_kalem_": "val_grafted_", "val_sirov_": "val_raw_",
}

ENV = {
    "MODELI": "MODELS", "SEEDOVA": "SEEDS", "BROJ_SEEDOVA": "SEEDS",
    "IZLAZ": "OUTPUT", "IZLAZ1": "OUTPUT1", "IZLAZ2": "OUTPUT2",
    "DOZE": "DOSES", "SKUPOVI": "DATASETS", "CLANOVA": "MEMBERS",
    "BROJ_GRUPA": "N_GROUPS", "POZICIJA": "POSITION", "VARIJANTE": "VARIANTS",
    "NIVOI": "LEVELS", "MODEL": "MODEL",
}


def rename_column(name: str) -> str:
    """Map one column name, exact match first, then a known prefix."""
    if name in COLUMNS:
        return COLUMNS[name]
    for pre, rep in PREFIXES.items():
        if name.startswith(pre):
            return rep + name[len(pre):]
    return name
