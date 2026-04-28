"""
BBC ses siniflandirma icin LightGBM model tanimlari ve egitim yardimcilari.

Disari aktarilan semboller:
  VARSAYILAN_PARAMS       — baslangic/fallback hiperparametreleri
  OZELLIK_GRUPLARI_467    — 467-dim icin ozellik grubu sinirlari
  OZELLIK_GRUPLARI_250    — 250-dim eski format icin
  lgbm_olustur            — model fabrikasi (olustur + fit)
  optuna_hedef            — Optuna trial hedef fonksiyonu
  shap_grup_onemi         — ham SHAP degerlerinden grup katkilari
  cv_degerlendir          — dosya duzeyinde stratified k-fold
"""

import numpy as np
import lightgbm as lgb
import shap
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

SEED = 42

# ── Varsayilan hiperparametreler ──────────────────────────────────────────────
VARSAYILAN_PARAMS = dict(
    n_estimators=500,
    num_leaves=127,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

# ── Ozellik grubu sinirlari ───────────────────────────────────────────────────
OZELLIK_GRUPLARI_467 = {
    "MFCC genisletilmis (320)":     (0,   320),
    "Mel filterbank enerjisi (80)":  (320, 400),
    "Spektral doku (28)":            (400, 428),
    "Tonal / harmonik (36)":         (428, 464),
    "Zamansal dinamik (3)":          (464, 467),
}

OZELLIK_GRUPLARI_250 = {
    "MFCC (240)":              (0,   240),
    "Spektral ozellikler (10)": (240, 250),
}


# ── Model fabrikasi ───────────────────────────────────────────────────────────
def lgbm_olustur(
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    early_stopping_rounds: int = 0,
) -> lgb.LGBMClassifier:
    """
    LGBMClassifier olusturur ve egitir.

    early_stopping_rounds > 0 ise X_val/y_val zorunludur.
    """
    model = lgb.LGBMClassifier(**params)

    fit_kwargs: dict = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    if early_stopping_rounds > 0 and X_val is not None:
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "multi_logloss"

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ── Optuna hedef fonksiyonu ───────────────────────────────────────────────────
def optuna_hedef(trial, X_train, y_train, X_val, y_val, ag_map: dict) -> float:
    """Val macro-F1 maksimizasyonu icin Optuna trial hedefi."""
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators",      200,  800),
        num_leaves        = trial.suggest_int("num_leaves",          31,  255),
        max_depth         = trial.suggest_int("max_depth",            4,   12),
        learning_rate     = trial.suggest_float("learning_rate",    5e-3, 0.2, log=True),
        subsample         = trial.suggest_float("subsample",         0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.4,  1.0),
        min_child_samples = trial.suggest_int("min_child_samples",    10,   80),
        reg_alpha         = trial.suggest_float("reg_alpha",          0.0,  3.0),
        reg_lambda        = trial.suggest_float("reg_lambda",         0.0,  3.0),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    sw = np.array([ag_map[y] for y in y_train])
    model = lgbm_olustur(params, X_train, y_train, X_val, y_val,
                         sample_weight=sw, early_stopping_rounds=30)
    return f1_score(y_val, model.predict(X_val), average="macro")


# ── SHAP grup katkilari ───────────────────────────────────────────────────────
def shap_grup_onemi(
    model: lgb.LGBMClassifier,
    X_ornek: np.ndarray,
    ozellik_gruplari: dict,
) -> dict[str, float]:
    """
    SHAP degerlerini hesaplar ve ozellik gruplarina gore yuzde katki doner.

    Returns:
        {grup_adi: yuzde_katki}  — toplam 100'e esit
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ornek)
    sv = np.array(shap_values)

    # shap>=0.40: (n_samples, n_features, n_classes)
    # shap<0.40 : (n_classes, n_samples, n_features)
    if sv.ndim == 3 and sv.shape[1] == X_ornek.shape[1]:
        shap_abs = np.abs(sv).mean(axis=(0, 2))   # (n_features,)
    elif sv.ndim == 3:
        shap_abs = np.abs(sv).mean(axis=(0, 1))   # (n_features,)
    else:
        shap_abs = np.abs(sv).mean(axis=0)

    grup_katkilari = {
        ad: float(shap_abs[bas:son].sum())
        for ad, (bas, son) in ozellik_gruplari.items()
    }
    toplam = sum(grup_katkilari.values()) or 1.0
    return {k: v / toplam * 100 for k, v in grup_katkilari.items()}


# ── Dosya duzeyinde stratified k-fold CV ─────────────────────────────────────
def cv_degerlendir(
    df_trainval: pd.DataFrame,
    scaler,
    params: dict,
    ozellik_sutunlari: list[str],
    n_folds: int = 5,
) -> pd.DataFrame:
    """
    Dosya duzeyinde stratified k-fold CV uygular.

    Args:
        df_trainval       : train + val DataFrame (dosya/sinif_idx sutunlari olmali)
        scaler            : fit edilmis StandardScaler
        params            : LightGBM hiperparametreleri
        ozellik_sutunlari : X sutun isimleri
        n_folds           : fold sayisi

    Returns:
        fold sonuclarini iceren DataFrame (accuracy, balanced_accuracy,
        macro_f1, weighted_f1 sutunlari)
    """
    dosya_sinif = (df_trainval.groupby("dosya")["sinif_idx"]
                   .first().reset_index())
    dosya_arr = dosya_sinif["dosya"].values
    sinif_arr = dosya_sinif["sinif_idx"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    sonuclar = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(dosya_arr, sinif_arr), 1):
        tr_dosyalar = set(dosya_arr[tr_idx])
        te_dosyalar = set(dosya_arr[te_idx])

        df_tr = df_trainval[df_trainval["dosya"].isin(tr_dosyalar)]
        df_te = df_trainval[df_trainval["dosya"].isin(te_dosyalar)]

        X_tr = scaler.transform(df_tr[ozellik_sutunlari].values.astype(np.float32))
        y_tr = df_tr["sinif_idx"].values
        X_te = scaler.transform(df_te[ozellik_sutunlari].values.astype(np.float32))
        y_te = df_te["sinif_idx"].values

        siniflar   = np.unique(y_tr)
        agirliklar = compute_class_weight("balanced", classes=siniflar, y=y_tr)
        ag_map     = dict(zip(siniflar, agirliklar))
        sw         = np.array([ag_map[y] for y in y_tr])

        # CV icin hafif parametreler — buyuk modelin OOM'unu onler
        cv_params = dict(**params)
        cv_params["n_estimators"] = min(cv_params.get("n_estimators", 500), 300)
        cv_params["num_leaves"]   = min(cv_params.get("num_leaves", 127), 63)

        model = lgbm_olustur(cv_params, X_tr, y_tr, sample_weight=sw)
        y_pred = model.predict(X_te)

        sonuclar.append({
            "fold":              fold,
            "accuracy":          accuracy_score(y_te, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "macro_f1":          f1_score(y_te, y_pred, average="macro"),
            "weighted_f1":       f1_score(y_te, y_pred, average="weighted"),
        })
        print(f"    Fold {fold}/{n_folds}  "
              f"acc={sonuclar[-1]['accuracy']:.4f}  "
              f"macro-F1={sonuclar[-1]['macro_f1']:.4f}")

    return pd.DataFrame(sonuclar)
