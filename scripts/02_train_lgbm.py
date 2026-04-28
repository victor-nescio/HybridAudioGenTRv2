"""
LGBM Baseline Siniflandirici — 467-dim ozellik vektoru uzerinde.

Protokol (Doktora Standarti):
  1. DOSYA duzeyinde stratified train/val/test ayrimi (%70/%15/%15)
     Segment sizintisini (data leakage) onler: ayni dosyanin segmentleri
     ayni bolumde kalir.
  2. Optuna HPO: val macro-F1 maksimizasyonu, N_DENEME deneme
  3. 5-katli stratified cross-validation (dosya duzeyinde)
  4. Sinif agirlikli LightGBM (dengesiz veri icin)
  5. SHAP tabanlı ozellik grubu onem analizi

Ciktilar:
  checkpoints/lgbm_model.pkl          Egitilmis model
  checkpoints/lgbm_scaler.pkl         StandardScaler
  outputs/lgbm_metrics.json           Tum metrikler (per-class F1 dahil)
  outputs/lgbm_confusion_matrix.png   Normalize karmasiklik matrisi
  outputs/lgbm_shap_groups.png        SHAP ozellik grubu onemi
  outputs/lgbm_cv_results.csv         CV fold sonuclari

Kullanim:
  python scripts/02_train_lgbm.py
  python scripts/02_train_lgbm.py --trials 100 --cv 10
  python scripts/02_train_lgbm.py --no-optuna   # varsayilan parametreler
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT       = Path(__file__).resolve().parent.parent
ISLENMIS   = ROOT / "data" / "processed"
CHECKPTS   = ROOT / "checkpoints"
OUTPUTS    = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
from src.models.lgbm import (OZELLIK_GRUPLARI_250, OZELLIK_GRUPLARI_467,
                              VARSAYILAN_PARAMS, cv_degerlendir, lgbm_olustur,
                              optuna_hedef, shap_grup_onemi)

# ── Sabitler ──────────────────────────────────────────────────────────────────
SEED       = 42
N_FOLDS    = 5
TEST_ORANI = 0.15
VAL_ORANI  = 0.15

N_DIM: int = 0
OZELLIK_GRUPLARI: dict = {}  # veri_yukle() tarafindan doldurulur


# ── Veri yukleme ve bolme ─────────────────────────────────────────────────────
def veri_yukle() -> pd.DataFrame:
    global N_DIM, OZELLIK_GRUPLARI

    yol = ISLENMIS / "lgbm_ozellikler.parquet"
    if not yol.exists():
        print(f"HATA: {yol} bulunamadi. Once 01_extract_features.py calistir.")
        sys.exit(1)

    df = pd.read_parquet(yol)

    # Ozellik sutunlarini otomatik tespit et
    ozellik_sutunlari = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    N_DIM = len(ozellik_sutunlari)

    OZELLIK_GRUPLARI = (OZELLIK_GRUPLARI_467 if N_DIM == 467
                        else OZELLIK_GRUPLARI_250)

    print(f"Veri yuklendi: {len(df):,} segment, "
          f"{df['sinif'].nunique()} sinif, {N_DIM}-dim ozellik")

    # NaN / Inf kontrolu
    bozuk = df[ozellik_sutunlari].isin([np.inf, -np.inf]).any().sum()
    bozuk += df[ozellik_sutunlari].isna().any().sum()
    if bozuk:
        print(f"  [!] {bozuk} sutunda NaN/Inf var, 0 ile dolduruluyor")
        df[ozellik_sutunlari] = df[ozellik_sutunlari].replace(
            [np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def dosya_duzeyi_bol(df: pd.DataFrame):
    """
    Segment sizintisini onlemek icin ayrimi DOSYA duzeyinde yapar.
    Her dosyanin tum segmentleri ayni bolumde (train/val/test) kalir.
    """
    dosya_sinif = (df.groupby("dosya")["sinif_idx"]
                   .first().reset_index()
                   .rename(columns={"sinif_idx": "sinif"}))

    # 1. test ayir
    d_trainval, d_test = train_test_split(
        dosya_sinif, test_size=TEST_ORANI,
        stratify=dosya_sinif["sinif"], random_state=SEED)

    # 2. val ayir (train+val icinden)
    val_gercek = VAL_ORANI / (1 - TEST_ORANI)
    d_train, d_val = train_test_split(
        d_trainval, test_size=val_gercek,
        stratify=d_trainval["sinif"], random_state=SEED)

    def filtrele(dosyalar):
        return df[df["dosya"].isin(dosyalar["dosya"])].reset_index(drop=True)

    return filtrele(d_train), filtrele(d_val), filtrele(d_test)


def ozellik_ayir(df: pd.DataFrame):
    sutunlar = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    return df[sutunlar].values.astype(np.float32), df["sinif_idx"].values


# ── HPO sarmalayici ───────────────────────────────────────────────────────────
def hpo_calistir(X_train, y_train, X_val, y_val,
                 ag_map: dict, n_deneme: int) -> dict:
    print(f"\n  Optuna HPO: {n_deneme} deneme, val macro-F1 maksimizasyonu")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(
        lambda t: optuna_hedef(t, X_train, y_train, X_val, y_val, ag_map),
        n_trials=n_deneme,
        show_progress_bar=True,
    )
    print(f"  En iyi val macro-F1 : {study.best_value:.4f}")
    print(f"  En iyi parametreler : {study.best_params}")
    return {**study.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1}


# ── Cikti fonksiyonlari ───────────────────────────────────────────────────────
def karmasiklik_matrisi_ciz(y_gercek, y_tahmin, sinif_adlari, cikti_yolu):
    cm = confusion_matrix(y_gercek, y_tahmin, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=sinif_adlari, yticklabels=sinif_adlari,
                linewidths=0.3, ax=ax)
    ax.set_xlabel("Tahmin edilen sinif", fontsize=12)
    ax.set_ylabel("Gercek sinif", fontsize=12)
    ax.set_title("Normallestirilmis Karmasiklik Matrisi — LGBM Baseline", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(cikti_yolu, dpi=150)
    plt.close(fig)
    print(f"  Karmasiklik matrisi: {cikti_yolu.name}")


def shap_grup_onemi_ciz(model, X_ornek, cikti_yolu):
    """SHAP grup katkilerini hesaplar ve gorsellestirir."""
    print("  SHAP hesaplaniyor (ornek)...")
    grup_onemleri = shap_grup_onemi(model, X_ornek, OZELLIK_GRUPLARI)

    fig, ax = plt.subplots(figsize=(9, 5))
    renkler = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    ax.barh(list(grup_onemleri.keys()), list(grup_onemleri.values()),
            color=renkler, edgecolor="white")
    for i, (k, v) in enumerate(grup_onemleri.items()):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Toplam SHAP katki orani (%)", fontsize=11)
    ax.set_title("Ozellik Grubu Onem Analizi (SHAP) — LGBM", fontsize=12)
    ax.set_xlim(0, max(grup_onemleri.values()) * 1.15)
    plt.tight_layout()
    fig.savefig(cikti_yolu, dpi=150)
    plt.close(fig)
    print(f"  SHAP grafigi      : {cikti_yolu.name}")
    return grup_onemleri


# ── Ana akis ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LGBM baseline siniflandirici")
    parser.add_argument("--trials",     type=int, default=50,
                        help="Optuna deneme sayisi (varsayilan: 50)")
    parser.add_argument("--cv",         type=int, default=N_FOLDS,
                        help="CV fold sayisi (varsayilan: 5)")
    parser.add_argument("--no-optuna",  action="store_true",
                        help="HPO atlayip varsayilan parametrelerle egit")
    args = parser.parse_args()

    CHECKPTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from src.data.labels import SINIFLAR
    sinif_adlari = [SINIFLAR[s][0] for s in sorted(SINIFLAR.keys())]

    # ── 1. Veri yukle ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  LGBM BASELINE EGITIMI")
    print("=" * 65)
    df = veri_yukle()

    # ── 2. Dosya duzeyinde bol ─────────────────────────────────────────────
    df_train, df_val, df_test = dosya_duzeyi_bol(df)
    print(f"\n  Train : {len(df_train):>7,} segment  "
          f"({df_train['dosya'].nunique()} dosya)")
    print(f"  Val   : {len(df_val):>7,} segment  "
          f"({df_val['dosya'].nunique()} dosya)")
    print(f"  Test  : {len(df_test):>7,} segment  "
          f"({df_test['dosya'].nunique()} dosya)")

    X_train, y_train = ozellik_ayir(df_train)
    X_val,   y_val   = ozellik_ayir(df_val)
    X_test,  y_test  = ozellik_ayir(df_test)

    # ── 3. Olceklendirme ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, CHECKPTS / "lgbm_scaler.pkl")

    # ── 4. Sinif agirliklari ───────────────────────────────────────────────
    siniflar   = np.unique(y_train)
    agirliklar = compute_class_weight("balanced", classes=siniflar, y=y_train)
    ag_map     = dict(zip(siniflar, agirliklar))
    sw_train   = np.array([ag_map[y] for y in y_train])

    # ── 5. HPO (Optuna) ────────────────────────────────────────────────────
    if args.no_optuna:
        print("\n  [HPO atlandi] Varsayilan parametreler kullaniliyor.")
        en_iyi_params = VARSAYILAN_PARAMS
    else:
        en_iyi_params = hpo_calistir(
            X_train, y_train, X_val, y_val, ag_map, args.trials)

    # ── 6. Son model — train + val verisiyle ──────────────────────────────
    print("\n  Son model egitiliyor (train + val)...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    sw_trainval = np.concatenate([
        sw_train,
        np.array([ag_map.get(y, 1.0) for y in y_val])
    ])

    son_model = lgbm_olustur(en_iyi_params, X_trainval, y_trainval,
                             sample_weight=sw_trainval)
    joblib.dump(son_model, CHECKPTS / "lgbm_model.pkl")
    print(f"  Model kaydedildi : checkpoints/lgbm_model.pkl")

    # ── 7. Cross-validation ────────────────────────────────────────────────
    print(f"\n  {args.cv}-katli stratified CV (dosya duzeyinde)...")
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    ozellik_sutunlari = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:])
    )
    cv_df = cv_degerlendir(df_trainval, scaler, en_iyi_params,
                           ozellik_sutunlari, args.cv)
    cv_df.to_csv(OUTPUTS / "lgbm_cv_results.csv", index=False)

    print(f"\n  CV Ozeti ({args.cv} fold):")
    for metrik in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        m = cv_df[metrik]
        print(f"    {metrik:<22}  {m.mean():.4f} ± {m.std():.4f}")

    # ── 8. Test seti degerlendirmesi ───────────────────────────────────────
    print("\n  Test seti degerlendirmesi...")
    y_pred = son_model.predict(X_test)

    rapor = classification_report(
        y_test, y_pred, target_names=sinif_adlari, output_dict=True)

    print("\n" + "-" * 65)
    print(f"  Test Accuracy         : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test Balanced Acc     : {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test Macro F1         : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  Test Weighted F1      : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("-" * 65)

    # Per-class F1 (ilk 10)
    print("\n  Per-class F1 (en dusuk 8):")
    per_f1 = {sinif_adlari[i]: rapor[sinif_adlari[i]]["f1-score"]
              for i in range(len(sinif_adlari)) if sinif_adlari[i] in rapor}
    for ad, f1 in sorted(per_f1.items(), key=lambda x: x[1])[:8]:
        print(f"    {ad:<30} {f1:.4f}")

    # ── 9. Metrikler JSON ──────────────────────────────────────────────────
    metrikler = {
        "test": {
            "accuracy":          accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_f1":          f1_score(y_test, y_pred, average="macro"),
            "weighted_f1":       f1_score(y_test, y_pred, average="weighted"),
            "per_class":         {k: v for k, v in rapor.items()
                                  if k not in ("accuracy","macro avg","weighted avg")},
        },
        "cv": {
            metrik: {"mean": float(cv_df[metrik].mean()),
                     "std":  float(cv_df[metrik].std())}
            for metrik in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        },
        "cv_folds": cv_df.to_dict(orient="records"),
        "best_params": {k: v for k, v in en_iyi_params.items()
                        if k not in ("random_state", "n_jobs", "verbose")},
        "n_train_segments": len(X_train),
        "n_val_segments":   len(X_val),
        "n_test_segments":  len(X_test),
        "n_features":       N_DIM,
    }
    with open(OUTPUTS / "lgbm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrikler, f, ensure_ascii=False, indent=2)
    print(f"\n  Metrikler kaydedildi : outputs/lgbm_metrics.json")

    # ── 10. Gorseller ─────────────────────────────────────────────────────
    karmasiklik_matrisi_ciz(y_test, y_pred, sinif_adlari,
                            OUTPUTS / "lgbm_confusion_matrix.png")

    # SHAP: kucuk ornek (hiz icin)
    n_shap = min(500, len(X_test))
    idx    = np.random.RandomState(SEED).choice(len(X_test), n_shap, replace=False)
    shap_gruplari = shap_grup_onemi_ciz(
        son_model, X_test[idx], OUTPUTS / "lgbm_shap_groups.png")

    print("\n  SHAP ozellik grubu onem sirasi:")
    for ad, oran in sorted(shap_gruplari.items(), key=lambda x: -x[1]):
        print(f"    {ad:<35} {oran:.1f}%")

    print("\n" + "=" * 65)
    print("  EGITIM TAMAMLANDI")
    print(f"  Test macro-F1  : {metrikler['test']['macro_f1']:.4f}")
    print(f"  CV macro-F1    : {metrikler['cv']['macro_f1']['mean']:.4f}"
          f" +- {metrikler['cv']['macro_f1']['std']:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
