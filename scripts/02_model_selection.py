"""
Klasik makine ogrenmesi algoritma secimi.

467-dim ozellik vektoru uzerinde 8 algoritmayı varsayilan parametrelerle
karsilastirir; val macro-F1 ve egitim suresine gore siralama yapar.

Amac: Hangi algoritmanin BBC ses siniflandirmaya en uygun oldugunu
      gostermek — HPO yapilmadan once algoritma secimini veriye dayandirmak.

Karsilastirilan algoritmalar:
  1. Logistic Regression   (lineer taban cizgisi)
  2. k-Nearest Neighbors   (ornek tabanli, alt orneklem)
  3. Support Vector Machine (RBF, alt orneklem)
  4. Random Forest
  5. Extra Trees
  6. Gradient Boosting (sklearn)
  7. XGBoost
  8. LightGBM              (hipotez: en iyi)

Ciktilar:
  outputs/model_selection_results.csv   — tam karsilastirma tablosu
  outputs/model_selection_report.json   — JSON raporu
  outputs/model_selection_chart.png     — gorsel karsilastirma

Kullanim:
  python scripts/02_model_selection.py
  python scripts/02_model_selection.py --no-slow   # SVM ve kNN atla
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                               HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import xgboost as xgb

ISLENMIS = ROOT / "data" / "processed"
OUTPUTS  = ROOT / "outputs"
SEED     = 42

# Alt orneklem buyuklugu — yavash algoritmalar icin (SVM, kNN)
YAVAS_ALG_MAX_ORNEK = 15_000


# ── Veri yukle ve bol ─────────────────────────────────────────────────────────
def veri_hazirla():
    yol = ISLENMIS / "lgbm_ozellikler.parquet"
    if not yol.exists():
        print(f"HATA: {yol} bulunamadi. Once 01_extract_features.py calistir.")
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

    df_train = filtrele(d_train)
    df_val   = filtrele(d_val)
    df_test  = filtrele(d_test)

    X_train = df_train[ozellik_sutunlari].values.astype(np.float32)
    y_train = df_train["sinif_idx"].values
    X_val   = df_val[ozellik_sutunlari].values.astype(np.float32)
    y_val   = df_val["sinif_idx"].values
    X_test  = df_test[ozellik_sutunlari].values.astype(np.float32)
    y_test  = df_test["sinif_idx"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    siniflar   = np.unique(y_train)
    agirliklar = compute_class_weight("balanced", classes=siniflar, y=y_train)
    ag_map     = dict(zip(siniflar, agirliklar))
    sw_train   = np.array([ag_map[y] for y in y_train])

    print(f"Veri hazir  ->  Train: {len(X_train):,}  Val: {len(X_val):,}  "
          f"Test: {len(X_test):,}  Boyut: {X_train.shape[1]}")

    return (X_train, y_train, sw_train,
            X_val, y_val,
            X_test, y_test,
            ag_map)


# ── Algoritma tanimi ──────────────────────────────────────────────────────────
def algoritmalari_tanimla(ag_map: dict, yavas: bool = True) -> list[dict]:
    sinif_agirliklari = {int(k): float(v) for k, v in ag_map.items()}

    algoritmalar = [
        {
            "ad": "Logistic Regression",
            "model": LogisticRegression(
                max_iter=1000, C=1.0,
                class_weight="balanced",
                solver="lbfgs", multi_class="multinomial",
                random_state=SEED, n_jobs=-1,
            ),
            "alt_orneklem": False,
        },
        {
            "ad": "Random Forest",
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ),
            "alt_orneklem": False,
        },
        {
            "ad": "Extra Trees",
            "model": ExtraTreesClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced",
                random_state=SEED, n_jobs=-1,
            ),
            "alt_orneklem": False,
        },
        {
            "ad": "Hist Gradient Boosting",
            "model": HistGradientBoostingClassifier(
                max_iter=300, max_depth=8,
                learning_rate=0.1,
                random_state=SEED,
            ),
            "alt_orneklem": False,
        },
        {
            "ad": "XGBoost",
            "model": xgb.XGBClassifier(
                n_estimators=300, max_depth=6,
                learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8,
                device="cuda", tree_method="hist",
                eval_metric="mlogloss",
                random_state=SEED,
                verbosity=0,
            ),
            "alt_orneklem": False,
        },
        {
            "ad": "LightGBM",
            "model": lgb.LGBMClassifier(
                n_estimators=500, num_leaves=127,
                max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced",
                device="gpu",
                random_state=SEED, n_jobs=-1, verbose=-1,
            ),
            "alt_orneklem": False,
        },
    ]

    if yavas:
        algoritmalar += [
            {
                "ad": "k-NN (k=5)",
                "model": KNeighborsClassifier(
                    n_neighbors=5, metric="euclidean", n_jobs=-1),
                "alt_orneklem": True,
            },
            {
                "ad": "SVM (RBF)",
                "model": SVC(
                    C=10.0, kernel="rbf", gamma="scale",
                    class_weight="balanced",
                    decision_function_shape="ovr",
                    random_state=SEED,
                ),
                "alt_orneklem": True,
            },
        ]

    return algoritmalar


# ── Tek algoritma degerlendirme ───────────────────────────────────────────────
def degerlendir(alg: dict,
                X_train, y_train, sw_train,
                X_val, y_val,
                X_test, y_test) -> dict:
    ad = alg["ad"]
    model = alg["model"]

    # Alt orneklem gerekiyorsa
    if alg["alt_orneklem"] and len(X_train) > YAVAS_ALG_MAX_ORNEK:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X_train), YAVAS_ALG_MAX_ORNEK, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
        print(f"  [{ad}] Alt orneklem: {YAVAS_ALG_MAX_ORNEK:,}/{len(X_train):,}")
    else:
        X_tr, y_tr = X_train, y_train

    print(f"  [{ad}] Egitiliyor...", end=" ", flush=True)
    t0 = time.time()

    try:
        # sample_weight destekleyen modeller icin
        if hasattr(model, "fit") and not alg["alt_orneklem"]:
            try:
                model.fit(X_tr, y_tr, sample_weight=sw_train[:len(X_tr)])
            except TypeError:
                model.fit(X_tr, y_tr)
        else:
            model.fit(X_tr, y_tr)
    except Exception as e:
        print(f"HATA: {e}")
        return {"ad": ad, "hata": str(e)}

    egitim_sure = time.time() - t0

    val_tahmin  = model.predict(X_val)
    test_tahmin = model.predict(X_test)

    val_f1   = f1_score(y_val,  val_tahmin,  average="macro", zero_division=0)
    test_f1  = f1_score(y_test, test_tahmin, average="macro", zero_division=0)
    val_acc  = accuracy_score(y_val,  val_tahmin)
    test_acc = accuracy_score(y_test, test_tahmin)

    print(f"Val F1={val_f1:.4f}  Test F1={test_f1:.4f}  Sure={egitim_sure:.1f}s")

    return {
        "ad":            ad,
        "val_macro_f1":  round(val_f1,   4),
        "test_macro_f1": round(test_f1,  4),
        "val_accuracy":  round(val_acc,  4),
        "test_accuracy": round(test_acc, 4),
        "egitim_sure_s": round(egitim_sure, 1),
        "alt_orneklem":  alg["alt_orneklem"],
    }


# ── Gorselleştirme ────────────────────────────────────────────────────────────
def grafik_ciz(df_sonuc: pd.DataFrame, cikti_yolu: Path):
    df = df_sonuc.sort_values("val_macro_f1", ascending=True)

    renkler = ["#e74c3c" if r["ad"] == df_sonuc.loc[
        df_sonuc["val_macro_f1"].idxmax(), "ad"] else "#3498db"
               for _, r in df.iterrows()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Val F1
    bars = axes[0].barh(df["ad"], df["val_macro_f1"],
                        color=renkler, edgecolor="white")
    for bar, val in zip(bars, df["val_macro_f1"]):
        axes[0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=9)
    axes[0].set_xlim(0, df["val_macro_f1"].max() * 1.15)
    axes[0].set_title("Validasyon Macro F1", fontsize=12)
    axes[0].set_xlabel("Macro F1")
    axes[0].axvline(df["val_macro_f1"].max(), color="red",
                    linestyle="--", alpha=0.4)

    # Egitim suresi
    axes[1].barh(df["ad"], df["egitim_sure_s"],
                 color="#95a5a6", edgecolor="white")
    for i, (_, satir) in enumerate(df.iterrows()):
        etiket = f"{satir['egitim_sure_s']:.1f}s"
        if satir["alt_orneklem"]:
            etiket += " *"
        axes[1].text(satir["egitim_sure_s"] + 0.5, i,
                     etiket, va="center", fontsize=9)
    axes[1].set_title("Egitim Suresi (saniye)", fontsize=12)
    axes[1].set_xlabel("Saniye")

    kazanan = df_sonuc.loc[df_sonuc["val_macro_f1"].idxmax(), "ad"]
    plt.suptitle(
        f"Klasik ML Algoritma Karsilastirmasi — BBC 26-Sinif\n"
        f"Kazanan: {kazanan}  |  * Alt orneklem kullanildi",
        fontsize=13
    )
    plt.tight_layout()
    fig.savefig(cikti_yolu, dpi=150)
    plt.close(fig)
    print(f"\nGrafik kaydedildi: {cikti_yolu.name}")


# ── Ana akis ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-slow", action="store_true",
                        help="SVM ve kNN'i atla (cok buyuk veri setleri icin)")
    parser.add_argument("--fast", action="store_true",
                        help="Sadece hizli agac modelleri: ET, HistGB, XGB, LightGBM")
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  KLASIK ML ALGORITMA SECIMI — BBC 26-SINIF")
    print("=" * 65 + "\n")

    # ── Veri ──────────────────────────────────────────────────────────────────
    (X_train, y_train, sw_train,
     X_val, y_val,
     X_test, y_test,
     ag_map) = veri_hazirla()

    # ── Algoritmalar ───────────────────────────────────────────────────────────
    algoritmalar = algoritmalari_tanimla(ag_map, yavas=not args.no_slow)
    if args.fast:
        atlanacak = {"Logistic Regression", "Random Forest"}
        algoritmalar = [a for a in algoritmalar if a["ad"] not in atlanacak]
        print("--fast: LR ve RF atlandi (onceki calisma: LR=0.3593, RF=0.4793)\n")
    print(f"\n{len(algoritmalar)} algoritma karsilastirilacak:\n")

    sonuclar = []
    for alg in algoritmalar:
        sonuc = degerlendir(
            alg,
            X_train, y_train, sw_train,
            X_val, y_val,
            X_test, y_test,
        )
        if "hata" not in sonuc:
            sonuclar.append(sonuc)

    # ── Siralama ───────────────────────────────────────────────────────────────
    df_sonuc = pd.DataFrame(sonuclar).sort_values(
        "val_macro_f1", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("  SONUCLAR (val macro-F1'e gore)")
    print("=" * 65)
    print(df_sonuc[["ad", "val_macro_f1", "test_macro_f1",
                     "val_accuracy", "egitim_sure_s"]].to_string(index=False))

    kazanan = df_sonuc.iloc[0]
    print(f"\n  Kazanan : {kazanan['ad']}")
    print(f"  Val  F1 : {kazanan['val_macro_f1']:.4f}")
    print(f"  Test F1 : {kazanan['test_macro_f1']:.4f}")
    print(f"  Sure    : {kazanan['egitim_sure_s']:.1f}s")

    # ── Kaydet ────────────────────────────────────────────────────────────────
    df_sonuc.to_csv(OUTPUTS / "model_selection_results.csv", index=False)

    rapor = {
        "kazanan":  kazanan["ad"],
        "sonuclar": df_sonuc.to_dict(orient="records"),
        "not": "* alt_orneklem=True olan modeller tam veri seti uzerinde "
               "degerlendirilemedi; sonuclari karsilastirmali yorumlanmali.",
    }
    with open(OUTPUTS / "model_selection_report.json", "w",
              encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=2)

    grafik_ciz(df_sonuc, OUTPUTS / "model_selection_chart.png")

    print("\n  Kaydedilen dosyalar:")
    print("    outputs/model_selection_results.csv")
    print("    outputs/model_selection_report.json")
    print("    outputs/model_selection_chart.png")
    print("\n  Sonraki adim: kazanan algoritmayi Optuna ile optimize et.")
    print(f"  -> python scripts/02_train_lgbm.py  (veya kazanana gore revize et)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
