"""
LGBM Tur4 McNemar testi — dogru yeniden egitim.

Sorunlar:
  - lgbm_baseline_iter.pkl, iteratif loop sonuclariyla tutarsiz F1 veriyor
  - sentetik_lgbm_ozellikler.parquet cache'i Tur1-3 verisini donduruyor (Tur4 degil)

Bu script:
  1. lgbm_ozellikler.parquet + veri_bol(seed=42) ile sabit split
  2. LGBM baseline: sadece gercek egitim verisiyle sifirdan egit
  3. LGBM augmented: gercek + synthetic_tur4/ ozelliklerini cikar ve egit
  4. Test kumesi uzerinde McNemar hesapla

Kullanim:
    python scripts/12b_mcnemar_lgbm_tur4.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import veri_bol

PARQUET_LISTE = ROOT / "data" / "processed" / "veri_listesi.parquet"
PARQUET_LGBM  = ROOT / "data" / "processed" / "lgbm_ozellikler.parquet"
SENTETIK_TUR4 = ROOT / "data" / "synthetic_tur4"
RESULT_DIR    = ROOT / "results" / "v1"
SEED          = 42
N_SINIF       = 26

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
SINIF_IDX = {s: i for i, s in enumerate(SINIF_SIRASI)}


def ozellik_sutunlari(df):
    return sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))


def lgbm_egit(df_train, df_val, df_test, etiket: str):
    import lightgbm as lgb

    sut = ozellik_sutunlari(df_train)

    def xy(df):
        return df[sut].fillna(0).values.astype(np.float32), df["sinif_idx"].values

    X_tr, y_tr = xy(df_train)
    X_vl, y_vl = xy(df_val)
    X_te, y_te = xy(df_test)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    params_yolu = ROOT / "checkpoints" / "lgbm_best_params.json"
    if params_yolu.exists():
        with open(params_yolu) as f:
            params = json.load(f)
        print(f"  HPO parametreleri yuklendi")
    else:
        params = dict(n_estimators=500, num_leaves=127, max_depth=8,
                      learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, min_child_samples=20,
                      reg_alpha=0.1, reg_lambda=1.0)
        print("  Varsayilan parametreler kullaniliyor")

    params.update({"device": "gpu", "verbose": -1,
                   "class_weight": "balanced",
                   "objective": "multiclass", "num_class": N_SINIF,
                   "metric": "multi_logloss", "random_state": SEED})

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=50)])

    pred_te = model.predict(X_te)
    f1_te   = f1_score(y_te, pred_te, average="macro", zero_division=0)
    sure_dk = (time.time() - t0) / 60
    print(f"  [{etiket}] test_F1={f1_te:.4f}  ({sure_dk:.1f} dk)")
    return pred_te, y_te, f1_te


def tur4_sentetik_ozellik_cikar():
    """
    synthetic_tur4/ altindaki WAV'lardan ozellik cikar.
    sentetik_lgbm_ozellikler.parquet cache'ini KULLANMAZ.
    """
    from src.data.features import duz_ozellik_cikar, SEGMENT_SN
    from src.data.preprocessing import onisle

    hedef_siniflar = ["acil_servis", "diger_hayvanlar", "dini_mekan", "insaat_sesleri"]
    satirlar = []

    for sinif in hedef_siniflar:
        sinif_idx = SINIF_IDX[sinif]
        klasor = SENTETIK_TUR4 / sinif
        if not klasor.exists():
            print(f"  [!] Klasor bulunamadi: {klasor}")
            continue
        wavlar = sorted([f for f in klasor.glob("*.wav") if f.stat().st_size > 1000])
        print(f"  {sinif}: {len(wavlar)} WAV isleniyor...")
        for i, wav in enumerate(wavlar):
            try:
                ses, sr_16k = onisle(wav)
                seg_uzunluk = int(SEGMENT_SN * sr_16k)
                n_seg = max(1, len(ses) // seg_uzunluk)
                for s_idx in range(n_seg):
                    seg = ses[s_idx * seg_uzunluk: (s_idx + 1) * seg_uzunluk]
                    if len(seg) < seg_uzunluk // 2:
                        continue
                    ozellik = duz_ozellik_cikar(seg, sr_16k)
                    satir = {"sinif": sinif, "sinif_idx": sinif_idx,
                             "dosya": wav.name, "segment_idx": s_idx}
                    satir.update({f"f{j}": v for j, v in enumerate(ozellik)})
                    satirlar.append(satir)
            except Exception as e:
                print(f"  [!] {wav.name}: {e}")

    df = pd.DataFrame(satirlar)
    print(f"  Toplam {len(df)} sentetik segment cikarildi")
    return df


def mcnemar_hesapla(dogru1, dogru2, etiket=""):
    n01 = np.sum(~dogru1 &  dogru2)
    n10 = np.sum( dogru1 & ~dogru2)
    if n01 + n10 == 0:
        return 0.0, 1.0, 0, 0
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    anlam = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"  {etiket:<45} chi2={chi2_stat:.3f}  p={p_val:.4f} {anlam}"
          f"  (n01={n01}, n10={n10})")
    return chi2_stat, p_val, int(n01), int(n10)


def main():
    print("=" * 65)
    print("  LGBM Tur4 McNemar — Yeniden Egitim")
    print("=" * 65)

    # Sabit split
    print("\nVeri bolunuyor (seed=42)...")
    df_tr_meta, df_vl_meta, df_te_meta = veri_bol(PARQUET_LISTE, ROOT)
    print(f"  Train={len(df_tr_meta):,}  Val={len(df_vl_meta):,}  Test={len(df_te_meta):,}")

    # LGBM ozellik tablosu
    print("\nLGBM ozellikleri yukleniyor...")
    df_lgbm = pd.read_parquet(PARQUET_LGBM)
    print(f"  {len(df_lgbm):,} satir, {df_lgbm.shape[1]} sutun")

    tr_dosyalar = set(df_tr_meta["dosya"].values)
    vl_dosyalar = set(df_vl_meta["dosya"].values)
    te_dosyalar = set(df_te_meta["dosya"].values)

    df_tr_lgbm = df_lgbm[df_lgbm["dosya"].isin(tr_dosyalar)].reset_index(drop=True)
    df_vl_lgbm = df_lgbm[df_lgbm["dosya"].isin(vl_dosyalar)].reset_index(drop=True)
    df_te_lgbm = df_lgbm[df_lgbm["dosya"].isin(te_dosyalar)].reset_index(drop=True)
    print(f"  Train={len(df_tr_lgbm):,}  Val={len(df_vl_lgbm):,}  Test={len(df_te_lgbm):,} segment")

    # Baseline egitimi
    print("\n[1/2] LGBM BASELINE egitiliyor (sadece gercek veri)...")
    pred_bl, y_te, f1_bl = lgbm_egit(df_tr_lgbm, df_vl_lgbm, df_te_lgbm, "baseline")

    # Tur4 sentetik ozellikler (cache'siz)
    print("\nTur4 sentetik ozellikler cikariliyor (synthetic_tur4/)...")
    df_synt = tur4_sentetik_ozellik_cikar()

    # Augmented egitim seti
    sut = ozellik_sutunlari(df_tr_lgbm)
    eksik = [c for c in sut if c not in df_synt.columns]
    if eksik:
        print(f"  [!] {len(eksik)} sutun sentetik veride eksik, 0 ile dolduruluyor")
        for c in eksik:
            df_synt[c] = 0.0

    df_synt_hizali = df_synt[["sinif", "sinif_idx", "dosya", "segment_idx"] + sut]
    df_tr_aug = pd.concat([df_tr_lgbm, df_synt_hizali], ignore_index=True)
    print(f"  Augmented train: {len(df_tr_lgbm):,} gercek + {len(df_synt):,} sentetik"
          f" = {len(df_tr_aug):,}")

    # Augmented egitim
    print("\n[2/2] LGBM AUGMENTED egitiliyor (gercek + Tur4 sentetik)...")
    pred_aug, _, f1_aug = lgbm_egit(df_tr_aug, df_vl_lgbm, df_te_lgbm, "augmented_tur4")

    print(f"\n  Baseline  test_F1 = {f1_bl:.4f}")
    print(f"  Augmented test_F1 = {f1_aug:.4f}")
    print(f"  Delta             = {f1_aug - f1_bl:+.4f}")

    # McNemar
    print(f"\n{'='*65}")
    print("  McNemar Test Sonucu  (Yates duzeltmeli, df=1)")
    dogru_bl  = (pred_bl  == y_te)
    dogru_aug = (pred_aug == y_te)
    chi2_stat, p_val, n01, n10 = mcnemar_hesapla(
        dogru_bl, dogru_aug,
        "LGBM Baseline vs LGBM Augmented (Tur4)")
    print("=" * 65)

    sonuc = {
        "lgbm_tur4_bl_vs_aug": {
            "chi2": chi2_stat, "p": p_val,
            "n01": n01, "n10": n10,
            "f1_baseline": f1_bl, "f1_augmented": f1_aug,
            "delta": round(f1_aug - f1_bl, 4),
            "n_test_segs": int(len(y_te))
        }
    }

    cikti = RESULT_DIR / "mcnemar_lgbm_tur4.json"
    RESULT_DIR.mkdir(exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"\nSonuc kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
