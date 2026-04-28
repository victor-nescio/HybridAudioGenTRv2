"""
Klasik ML modelleri icin fabrika fonksiyonlari ve Optuna arama uzaylari.

Desteklenen algoritmalar:
  logistic_regression, random_forest, extra_trees,
  gradient_boosting, xgboost, lightgbm

Her algoritma icin disari aktarilan semboller:
  MODEL_KAYITLARI   — {algoritma_adi: ModelKaydi} sozlugu
  model_olustur     — params dict'ten egitilmis model doner
  optuna_uzayi_olustur — trial'dan params dict olusturur
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb

SEED = 42


# ── Model kaydı veri yapısı ───────────────────────────────────────────────────
@dataclass
class ModelKaydi:
    ad: str
    fabrika: Callable          # params -> sklearn-uyumlu model
    uzay_fn: Callable          # trial -> params dict
    varsayilan: dict = field(default_factory=dict)
    sinif_agirlik_destegi: bool = True   # sample_weight destegi


# ── Fabrika fonksiyonlari ─────────────────────────────────────────────────────
def _lgbm_fabrika(params: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        **{**params, "random_state": SEED, "n_jobs": -1, "verbose": -1})


def _xgb_fabrika(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **{**params,
           "random_state": SEED, "n_jobs": -1,
           "eval_metric": "mlogloss", "verbosity": 0})


def _rf_fabrika(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        **{**params, "random_state": SEED, "n_jobs": -1})


def _et_fabrika(params: dict) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        **{**params, "random_state": SEED, "n_jobs": -1})


def _gb_fabrika(params: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(**{**params, "random_state": SEED})


def _lr_fabrika(params: dict) -> LogisticRegression:
    return LogisticRegression(
        **{**params,
           "random_state": SEED, "n_jobs": -1,
           "max_iter": 2000, "multi_class": "multinomial"})


# ── Optuna arama uzaylari ─────────────────────────────────────────────────────
def _lgbm_uzay(trial) -> dict:
    return dict(
        n_estimators      = trial.suggest_int("n_estimators",      200,  800),
        num_leaves        = trial.suggest_int("num_leaves",          31,  255),
        max_depth         = trial.suggest_int("max_depth",            4,   12),
        learning_rate     = trial.suggest_float("learning_rate",    5e-3, 0.2, log=True),
        subsample         = trial.suggest_float("subsample",         0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.4,  1.0),
        min_child_samples = trial.suggest_int("min_child_samples",    10,   80),
        reg_alpha         = trial.suggest_float("reg_alpha",          0.0,  3.0),
        reg_lambda        = trial.suggest_float("reg_lambda",         0.0,  3.0),
        class_weight      = "balanced",
    )


def _xgb_uzay(trial) -> dict:
    return dict(
        n_estimators      = trial.suggest_int("n_estimators",    100,  800),
        max_depth         = trial.suggest_int("max_depth",          3,   10),
        learning_rate     = trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        subsample         = trial.suggest_float("subsample",      0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight",   1,   20),
        reg_alpha         = trial.suggest_float("reg_alpha",       0.0,  3.0),
        reg_lambda        = trial.suggest_float("reg_lambda",      0.0,  3.0),
        gamma             = trial.suggest_float("gamma",           0.0,  1.0),
    )


def _rf_uzay(trial) -> dict:
    return dict(
        n_estimators      = trial.suggest_int("n_estimators",    100,  800),
        max_depth         = trial.suggest_int("max_depth",          5,   30),
        min_samples_split = trial.suggest_int("min_samples_split",  2,   20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf",   1,   10),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight      = "balanced",
    )


def _et_uzay(trial) -> dict:
    return dict(
        n_estimators      = trial.suggest_int("n_estimators",    100,  800),
        max_depth         = trial.suggest_int("max_depth",          5,   30),
        min_samples_split = trial.suggest_int("min_samples_split",  2,   20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf",   1,   10),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight      = "balanced",
    )


def _gb_uzay(trial) -> dict:
    return dict(
        n_estimators  = trial.suggest_int("n_estimators",    100,  500),
        max_depth     = trial.suggest_int("max_depth",          3,    8),
        learning_rate = trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        subsample     = trial.suggest_float("subsample",      0.5,  1.0),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20),
        max_features  = trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    )


def _lr_uzay(trial) -> dict:
    return dict(
        C      = trial.suggest_float("C", 1e-3, 100.0, log=True),
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"]),
        class_weight = "balanced",
    )


# ── Varsayilan parametreler ───────────────────────────────────────────────────
_LGBM_VARSAYILAN = dict(
    n_estimators=500, num_leaves=127, max_depth=8,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
    class_weight="balanced",
)
_XGB_VARSAYILAN = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
)
_RF_VARSAYILAN = dict(
    n_estimators=300, class_weight="balanced",
)
_ET_VARSAYILAN = dict(
    n_estimators=300, class_weight="balanced",
)
_GB_VARSAYILAN = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
)
_LR_VARSAYILAN = dict(
    C=1.0, solver="lbfgs", class_weight="balanced",
)


# ── Model kaydı sozlugu ───────────────────────────────────────────────────────
MODEL_KAYITLARI: dict[str, ModelKaydi] = {
    "lightgbm": ModelKaydi(
        ad="LightGBM",
        fabrika=_lgbm_fabrika,
        uzay_fn=_lgbm_uzay,
        varsayilan=_LGBM_VARSAYILAN,
    ),
    "xgboost": ModelKaydi(
        ad="XGBoost",
        fabrika=_xgb_fabrika,
        uzay_fn=_xgb_uzay,
        varsayilan=_XGB_VARSAYILAN,
    ),
    "random_forest": ModelKaydi(
        ad="Random Forest",
        fabrika=_rf_fabrika,
        uzay_fn=_rf_uzay,
        varsayilan=_RF_VARSAYILAN,
    ),
    "extra_trees": ModelKaydi(
        ad="Extra Trees",
        fabrika=_et_fabrika,
        uzay_fn=_et_uzay,
        varsayilan=_ET_VARSAYILAN,
    ),
    "gradient_boosting": ModelKaydi(
        ad="Gradient Boosting",
        fabrika=_gb_fabrika,
        uzay_fn=_gb_uzay,
        varsayilan=_GB_VARSAYILAN,
        sinif_agirlik_destegi=False,
    ),
    "logistic_regression": ModelKaydi(
        ad="Logistic Regression",
        fabrika=_lr_fabrika,
        uzay_fn=_lr_uzay,
        varsayilan=_LR_VARSAYILAN,
    ),
}

# model_selection.py ciktisindaki gorunen ad -> anahtar eslesmesi
GORUNEN_AD_ESLESTIRME: dict[str, str] = {
    "LightGBM":           "lightgbm",
    "XGBoost":            "xgboost",
    "Random Forest":      "random_forest",
    "Extra Trees":        "extra_trees",
    "Gradient Boosting":  "gradient_boosting",
    "Logistic Regression":"logistic_regression",
    "k-NN (k=5)":         None,   # HPO destegi yok
    "SVM (RBF)":          None,
}


# ── Yardimci fonksiyonlar ─────────────────────────────────────────────────────
def model_olustur(algoritma: str, params: dict,
                  X_train: np.ndarray, y_train: np.ndarray,
                  sample_weight: np.ndarray | None = None):
    """Params dict'ten model olusturur ve egitir."""
    kayit = MODEL_KAYITLARI[algoritma]
    model = kayit.fabrika(params)

    if sample_weight is not None and kayit.sinif_agirlik_destegi:
        try:
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return model
        except TypeError:
            pass
    model.fit(X_train, y_train)
    return model


def optuna_hedef_olustur(algoritma: str,
                          X_train, y_train, sw_train,
                          X_val, y_val) -> Callable:
    """Verilen algoritma icin Optuna hedef fonksiyonu doner."""
    kayit = MODEL_KAYITLARI[algoritma]

    def hedef(trial) -> float:
        params = kayit.uzay_fn(trial)
        model  = model_olustur(algoritma, params, X_train, y_train, sw_train)
        return f1_score(y_val, model.predict(X_val),
                        average="macro", zero_division=0)

    return hedef


def sinif_agirliklari_hesapla(y_train: np.ndarray) -> np.ndarray:
    siniflar   = np.unique(y_train)
    agirliklar = compute_class_weight("balanced", classes=siniflar, y=y_train)
    ag_map     = dict(zip(siniflar, agirliklar))
    return np.array([ag_map[y] for y in y_train])
