"""
Kazanan klasik ML algoritması için Optuna HPO.

Model seçim sonucunu (outputs/model_selection_report.json) otomatik okur
ve kazanan algoritmaya uygun arama uzayıyla HPO çalıştırır.

Kullanım:
  python scripts/02b_optuna_hpo.py                   # kazananı otomatik seç
  python scripts/02b_optuna_hpo.py --model lightgbm  # manuel seçim
  python scripts/02b_optuna_hpo.py --trials 100       # deneme sayısı

Desteklenen modeller:
  lightgbm, xgboost, random_forest, extra_trees,
  gradient_boosting, logistic_regression

Çıktılar:
  checkpoints/<model>_best.pkl      Eğitilmiş final model
  checkpoints/<model>_scaler.pkl    StandardScaler
  outputs/<model>_hpo_results.json  HPO raporu + metrikler
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.classical import (GORUNEN_AD_ESLESTIRME, MODEL_KAYITLARI,
                                   model_olustur, optuna_hedef_olustur,
                                   sinif_agirliklari_hesapla)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ISLENMIS  = ROOT / "data" / "processed"
CHECKPTS  = ROOT / "checkpoints"
OUTPUTS   = ROOT / "outputs"
SEED      = 42


# ── Kazananı tespit et ────────────────────────────────────────────────────────
def kazanani_bul(model_arg: str | None) -> str:
    if model_arg:
        if model_arg not in MODEL_KAYITLARI:
            print(f"HATA: '{model_arg}' desteklenmiyor.")
            print(f"Secenekler: {', '.join(MODEL_KAYITLARI.keys())}")
            sys.exit(1)
        return model_arg

    rapor_yolu = OUTPUTS / "model_selection_report.json"
    if not rapor_yolu.exists():
        print("HATA: outputs/model_selection_report.json bulunamadi.")
        print("Once python scripts/02_model_selection.py calistir.")
        sys.exit(1)

    with open(rapor_yolu, encoding="utf-8") as f:
        rapor = json.load(f)

    gorunen_ad = rapor["kazanan"]
    anahtar    = GORUNEN_AD_ESLESTIRME.get(gorunen_ad)

    if anahtar is None:
        print(f"'{gorunen_ad}' icin Optuna HPO destegi yok (SVM/kNN).")
        print("--model parametresiyle desteklenen bir algoritma sec.")
        sys.exit(1)

    print(f"Model secim raporu okundu → kazanan: {gorunen_ad} ({anahtar})")
    return anahtar


# ── Veri yükleme ──────────────────────────────────────────────────────────────
def veri_yukle():
    yol = ISLENMIS / "lgbm_ozellikler.parquet"
    if not yol.exists():
        print(f"HATA: {yol} bulunamadi.")
        sys.exit(1)

    df = pd.read_parquet(yol)
    ozellik_sutunlari = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )

    dosya_sinif = df.groupby("dosya")["sinif_idx"].first().reset_index()
    d_trainval, d_test = train_test_split(
        dosya_sinif, test_size=0.15,
        stratify=dosya_sinif["sinif_idx"], random_state=SEED)
    d_train, d_val = train_test_split(
        d_trainval, test_size=0.15 / 0.85,
        stratify=d_trainval["sinif_idx"], random_state=SEED)

    def filtrele(d): return df[df["dosya"].isin(d["dosya"])]

    df_train    = filtrele(d_train)
    df_val      = filtrele(d_val)
    df_test     = filtrele(d_test)
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    X_train = df_train[ozellik_sutunlari].values.astype(np.float32)
    y_train = df_train["sinif_idx"].values
    X_val   = df_val[ozellik_sutunlari].values.astype(np.float32)
    y_val   = df_val["sinif_idx"].values
    X_test  = df_test[ozellik_sutunlari].values.astype(np.float32)
    y_test  = df_test["sinif_idx"].values
    X_tv    = df_trainval[ozellik_sutunlari].values.astype(np.float32)
    y_tv    = df_trainval["sinif_idx"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    X_tv    = scaler.transform(X_tv)

    print(f"Veri: Train={len(X_train):,}  Val={len(X_val):,}  "
          f"Test={len(X_test):,}  Boyut={X_train.shape[1]}")

    return (X_train, y_train, X_val, y_val,
            X_test, y_test, X_tv, y_tv, scaler)


# ── Optuna HPO ────────────────────────────────────────────────────────────────
def hpo_calistir(algoritma: str, X_train, y_train,
                 X_val, y_val, n_deneme: int) -> dict:
    sw_train = sinif_agirliklari_hesapla(y_train)
    hedef    = optuna_hedef_olustur(algoritma,
                                    X_train, y_train, sw_train,
                                    X_val, y_val)

    print(f"\nOptuna HPO: {n_deneme} deneme, val macro-F1 maksimizasyonu")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(hedef, n_trials=n_deneme, show_progress_bar=True)

    print(f"En iyi val macro-F1 : {study.best_value:.4f}")
    print(f"En iyi parametreler : {study.best_params}")
    return study.best_params


# ── Ana akış ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Klasik ML icin Optuna HPO")
    parser.add_argument("--model",   default=None,
                        help="Algoritma adi (bos birakilirsa kazanan okunur)")
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--no-hpo",  action="store_true",
                        help="HPO atla, varsayilan parametrelerle egit")
    args = parser.parse_args()

    CHECKPTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  OPTUNA HPO — KLASIK ML ALGORITMA OPTIMIZASYONU")
    print("=" * 65 + "\n")

    # ── Algoritma seç ──────────────────────────────────────────────────────
    algoritma = kazanani_bul(args.model)
    kayit     = MODEL_KAYITLARI[algoritma]
    print(f"Hedef algoritma: {kayit.ad}\n")

    # ── Veri yükle ─────────────────────────────────────────────────────────
    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_tv, y_tv, scaler) = veri_yukle()

    sw_train = sinif_agirliklari_hesapla(y_train)
    sw_tv    = sinif_agirliklari_hesapla(y_tv)

    # ── HPO veya varsayılan ────────────────────────────────────────────────
    if args.no_hpo:
        print("HPO atlandi, varsayilan parametreler kullaniliyor.")
        en_iyi_params = kayit.varsayilan
        en_iyi_val_f1 = None
    else:
        en_iyi_params = hpo_calistir(
            algoritma, X_train, y_train, X_val, y_val, args.trials)
        # Val F1'i raporla
        model_val = model_olustur(algoritma, en_iyi_params,
                                  X_train, y_train, sw_train)
        en_iyi_val_f1 = f1_score(y_val, model_val.predict(X_val),
                                  average="macro", zero_division=0)

    # ── Final model (train + val) ──────────────────────────────────────────
    print("\nFinal model egitiliyor (train + val)...")
    final_model = model_olustur(algoritma, en_iyi_params,
                                X_tv, y_tv, sw_tv)

    # ── Test değerlendirmesi ───────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    test_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    test_acc = accuracy_score(y_test, y_pred)
    test_bal = balanced_accuracy_score(y_test, y_pred)
    rapor    = classification_report(y_test, y_pred,
                                     zero_division=0, output_dict=True)

    print("\n" + "-" * 65)
    print(f"  Test Macro F1         : {test_f1:.4f}")
    print(f"  Test Accuracy         : {test_acc:.4f}")
    print(f"  Test Balanced Acc     : {test_bal:.4f}")
    print("-" * 65)

    # ── Kaydet ────────────────────────────────────────────────────────────
    model_yolu  = CHECKPTS / f"{algoritma}_best.pkl"
    scaler_yolu = CHECKPTS / f"{algoritma}_scaler.pkl"
    joblib.dump(final_model, model_yolu)
    joblib.dump(scaler,      scaler_yolu)

    sonuc = {
        "algoritma":      algoritma,
        "algoritma_ad":   kayit.ad,
        "hpo_yapildi":    not args.no_hpo,
        "n_deneme":       args.trials if not args.no_hpo else 0,
        "en_iyi_val_f1":  round(en_iyi_val_f1, 4) if en_iyi_val_f1 else None,
        "test_macro_f1":  round(test_f1, 4),
        "test_accuracy":  round(test_acc, 4),
        "test_bal_acc":   round(test_bal, 4),
        "en_iyi_params":  en_iyi_params,
        "sinif_raporu":   rapor,
    }
    sonuc_yolu = OUTPUTS / f"{algoritma}_hpo_results.json"
    with open(sonuc_yolu, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)

    print(f"\n  Model kaydedildi  : checkpoints/{algoritma}_best.pkl")
    print(f"  Scaler kaydedildi : checkpoints/{algoritma}_scaler.pkl")
    print(f"  Rapor kaydedildi  : outputs/{algoritma}_hpo_results.json")
    print("\n" + "=" * 65)
    print(f"  TAMAMLANDI — {kayit.ad}")
    print(f"  Test macro-F1 : {test_f1:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
