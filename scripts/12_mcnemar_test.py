"""
McNemar istatistiksel anlamlilik testleri.

Karsilastirmalar:
  A) CNN Baseline  vs  CNN Augmented (Tur1)
  B) CNN Baseline  vs  CNN Augmented (Tur2)
  C) CNN Baseline  vs  CNN Augmented (Tur3)
  D) LightGBM Baseline  vs  LightGBM Augmented
  E) CNN Baseline  vs  LightGBM Baseline  (model karsilastirma)

McNemar testi: H0 = iki model ayni hatalari yapiyor.
p < 0.05 -> istatistiksel olarak anlamli fark.

Kullanim:
    python scripts/12_mcnemar_test.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import dataloader_olustur, veri_bol
from src.models.cnn import SpectrogramCNN

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PARQUET      = ROOT / "data" / "processed" / "veri_listesi.parquet"
MODEL_DIR    = ROOT / "models"
CKPT_DIR     = ROOT / "checkpoints"
RESULT_DIR   = ROOT / "results" / "v1"
N_SINIF      = 26
SEED         = 42


# ── McNemar hesaplama ─────────────────────────────────────────────────────────

def mcnemar_hesapla(dogru1, dogru2, etiket=""):
    """
    dogru1, dogru2: bool array (her ornek icin dogru mu?)
    Donus: (chi2_stat, p_value, n01, n10)
    """
    n01 = np.sum(~dogru1 &  dogru2)   # 1 yanlis, 2 dogru
    n10 = np.sum( dogru1 & ~dogru2)   # 1 dogru,  2 yanlis

    if n01 + n10 == 0:
        return 0.0, 1.0, 0, 0

    # Yates duzeltmeli McNemar
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)

    anlam = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"  {etiket:<45} chi2={chi2_stat:.3f}  p={p_val:.4f} {anlam}  "
          f"(n01={n01}, n10={n10})")
    return chi2_stat, p_val, int(n01), int(n10)


# ── CNN tahmin ────────────────────────────────────────────────────────────────

def cnn_tahmin_et(ckpt_yolu: Path, dl_test):
    model = SpectrogramCNN(n_sinif=N_SINIF).to(DEVICE)
    ck = torch.load(ckpt_yolu, map_location=DEVICE)
    # checkpoint ya direkt state_dict ya da {'model_state': ...} olabilir
    state = ck.get("model_state", ck)
    model.load_state_dict(state)
    model.eval()

    tum_tahmin, tum_gercek = [], []
    with torch.no_grad():
        for batch in dl_test:
            # Dataset (mel_tensor, sinif_idx) tuple donduruyor
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["mel"], batch["sinif_idx"]
            x = x.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            tum_tahmin.extend(pred)
            tum_gercek.extend(y.numpy() if hasattr(y, "numpy") else y)

    return np.array(tum_tahmin), np.array(tum_gercek)


# ── LightGBM tahmin ───────────────────────────────────────────────────────────

def lgbm_tahmin_et(model_pkl: Path, scaler_pkl: Path, df_test):
    model  = joblib.load(model_pkl)
    scaler = joblib.load(scaler_pkl)

    # egit_lgbm_aug ile ayni sutun secimi: f0, f1, ..., f466
    ozellik_sut = sorted([c for c in df_test.columns
                          if c.startswith("f") and c[1:].isdigit()],
                         key=lambda c: int(c[1:]))

    X = scaler.transform(df_test[ozellik_sut].fillna(0).values.astype(np.float32))
    y = df_test["sinif_idx"].values
    pred = model.predict(X)
    return pred, y


# ── Ana ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  McNemar Istatistiksel Anlamlilik Testleri")
    print("=" * 65)

    # ── Veri bolunmesi ────────────────────────────────────────────────────────
    print("\nVeri bolunuyor (seed=42)...")
    df_tr, df_val, df_test = veri_bol(PARQUET, ROOT)
    print(f"  Test seti: {len(df_test):,} dosya")

    # CNN test dataloader
    dl_test = dataloader_olustur(df_test, ROOT, batch_size=64, shuffle=False)
    n_test_seg = len(dl_test.dataset)
    print(f"  CNN test segmenti: {n_test_seg:,}")

    # LightGBM test: LGBM parquet
    import pandas as pd
    PARQUET_LGBM = ROOT / "data" / "processed" / "lgbm_ozellikler.parquet"
    df_lgbm = pd.read_parquet(PARQUET_LGBM)
    meta = {"dosya", "sinif", "sinif_idx", "bolum", "n_segs"}
    # Ayni test dosyalari ile eslesir
    test_dosyalar = set(df_test["dosya"].values)
    df_lgbm_test  = df_lgbm[df_lgbm["dosya"].isin(test_dosyalar)].reset_index(drop=True)
    print(f"  LGBM test segmenti: {len(df_lgbm_test):,}")

    # ── CNN tahminleri ────────────────────────────────────────────────────────
    print("\nCNN tahminleri uretiliyor...")
    ckpt_bl   = MODEL_DIR / "cnn_baseline_iter_best.pt"
    ckpt_aug1 = MODEL_DIR / "cnn_augmented_iter_best.pt"   # en son aug (Tur3 tarafinda)

    # Tur2 ve Tur3 aug checkpoint: ayni dosya (son egitim)
    # baseline her seferinde ayni (seed=42 sabit)
    pred_cnn_bl,   gercek_cnn = cnn_tahmin_et(ckpt_bl,   dl_test)
    pred_cnn_aug,  _          = cnn_tahmin_et(ckpt_aug1, dl_test)

    dogru_cnn_bl  = (pred_cnn_bl  == gercek_cnn)
    dogru_cnn_aug = (pred_cnn_aug == gercek_cnn)

    f1_bl  = f1_score(gercek_cnn, pred_cnn_bl,  average="macro", zero_division=0)
    f1_aug = f1_score(gercek_cnn, pred_cnn_aug, average="macro", zero_division=0)
    print(f"  CNN Baseline  test_F1={f1_bl:.4f}")
    print(f"  CNN Augmented test_F1={f1_aug:.4f}")

    # ── LightGBM tahminleri ───────────────────────────────────────────────────
    print("\nLightGBM tahminleri uretiliyor...")
    lgbm_bl_pkl  = CKPT_DIR / "lgbm_baseline_iter.pkl"
    lgbm_aug_pkl = CKPT_DIR / "lgbm_augmented_iter.pkl"
    lgbm_bl_sc   = CKPT_DIR / "lgbm_baseline_iter_scaler.pkl"
    lgbm_aug_sc  = CKPT_DIR / "lgbm_augmented_iter_scaler.pkl"

    pred_lgbm_bl,  gercek_lgbm = lgbm_tahmin_et(lgbm_bl_pkl,  lgbm_bl_sc,  df_lgbm_test)
    pred_lgbm_aug, _           = lgbm_tahmin_et(lgbm_aug_pkl, lgbm_aug_sc, df_lgbm_test)

    dogru_lgbm_bl  = (pred_lgbm_bl  == gercek_lgbm)
    dogru_lgbm_aug = (pred_lgbm_aug == gercek_lgbm)

    f1_lgbm_bl  = f1_score(gercek_lgbm, pred_lgbm_bl,  average="macro", zero_division=0)
    f1_lgbm_aug = f1_score(gercek_lgbm, pred_lgbm_aug, average="macro", zero_division=0)
    print(f"  LGBM Baseline  test_F1={f1_lgbm_bl:.4f}")
    print(f"  LGBM Augmented test_F1={f1_lgbm_aug:.4f}")

    # ── McNemar testleri ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  McNemar Test Sonuclari  (Yates duzeltmeli, df=1)")
    print(f"  {'Karsilastirma':<45} chi2     p-deger  anlam")
    print(f"  {'-'*63}")

    sonuclar = {}

    s1_chi2, s1_p, s1_n01, s1_n10 = mcnemar_hesapla(
        dogru_cnn_bl, dogru_cnn_aug,
        "CNN Baseline vs CNN Augmented (Tur2/3)")
    sonuclar["cnn_bl_vs_aug"] = {"chi2": s1_chi2, "p": s1_p,
                                  "n01": s1_n01, "n10": s1_n10,
                                  "f1_a": f1_bl, "f1_b": f1_aug}

    # CNN vs LGBM (segment bazli karsilastirma -- segment sayilari esit olmayabilir)
    # Sadece CNN baseline vs LGBM baseline (ortak dosya bazinda)
    # Segment duzeyinde dogrudan karsilastirma mumkun degil (farkli pencere sayisi)
    # -> Dosya duzeyinde majority vote ile eslestir
    print(f"\n  Not: CNN vs LGBM karsilastirmasi dosya duzeyinde majority vote ile yapilmaktadir.")

    # CNN dosya bazli tahmin
    from collections import Counter
    def dosya_bazli_tahmin(pred_arr, df_test_in):
        # Her segment icin dosya adi al
        dosya_liste = []
        for _, row in df_test_in.iterrows():
            dosya_liste.extend([row["dosya"]] * int(row.get("n_segs", 1)))
        dosya_arr = np.array(dosya_liste[:len(pred_arr)])

        dosya_pred, dosya_gercek = {}, {}
        for i, d in enumerate(dosya_arr):
            dosya_pred.setdefault(d, []).append(pred_arr[i])

        # Gercek sinif: df_test'ten al
        for _, row in df_test_in.iterrows():
            dosya_gercek[row["dosya"]] = row["sinif_idx"]

        dosyalar = sorted(set(dosya_arr) & set(dosya_gercek.keys()))
        preds   = np.array([Counter(dosya_pred[d]).most_common(1)[0][0] for d in dosyalar])
        gerceks = np.array([dosya_gercek[d] for d in dosyalar])
        return preds, gerceks

    pred_cnn_dosya, gercek_dosya = dosya_bazli_tahmin(pred_cnn_bl, df_test)

    # LGBM dosya bazli tahmin
    lgbm_dosya_pred, lgbm_dosya_gercek = {}, {}
    for i, row in df_lgbm_test.iterrows():
        d = row["dosya"]
        lgbm_dosya_pred.setdefault(d, []).append(pred_lgbm_bl[i])
        lgbm_dosya_gercek[d] = row["sinif_idx"]

    dosyalar_ortak = sorted(set(dosya_bazli_tahmin(pred_cnn_bl, df_test)[0].__class__.__name__ and
                                set(df_test["dosya"].values)) &
                            set(lgbm_dosya_gercek.keys()))

    # Dogrudan dosya listesi olustur
    test_dosya_listesi = sorted(df_test["dosya"].unique())
    lgbm_test_dosyalar = sorted(df_lgbm_test["dosya"].unique())
    ortak_dosyalar = sorted(set(test_dosya_listesi) & set(lgbm_test_dosyalar))

    # CNN dosya bazli tahminler (ortak dosyalar icin)
    cnn_dosya_pred_d  = {}
    cnn_dosya_gercek_d = {}
    seg_sayac = 0
    for _, row in df_test.iterrows():
        d = row["dosya"]
        n = int(row.get("n_segs", 1))
        segs = pred_cnn_bl[seg_sayac:seg_sayac+n]
        seg_sayac += n
        if len(segs) > 0:
            cnn_dosya_pred_d[d]   = Counter(segs).most_common(1)[0][0]
            cnn_dosya_gercek_d[d] = int(row["sinif_idx"])

    cnn_preds_ortak  = np.array([cnn_dosya_pred_d.get(d, -1)  for d in ortak_dosyalar])
    lgbm_preds_ortak = np.array([Counter(lgbm_dosya_pred.get(d, [0])).most_common(1)[0][0]
                                  for d in ortak_dosyalar])
    gercek_ortak     = np.array([cnn_dosya_gercek_d.get(d, -1) for d in ortak_dosyalar])

    mask = (cnn_preds_ortak >= 0) & (gercek_ortak >= 0)
    cnn_p = cnn_preds_ortak[mask]
    lgb_p = lgbm_preds_ortak[mask]
    grc   = gercek_ortak[mask]

    dogru_cnn_dosya  = (cnn_p == grc)
    dogru_lgbm_dosya = (lgb_p == grc)

    f1_cnn_d  = f1_score(grc, cnn_p,  average="macro", zero_division=0)
    f1_lgbm_d = f1_score(grc, lgb_p,  average="macro", zero_division=0)
    print(f"  Ortak dosya sayisi: {mask.sum()}  "
          f"CNN_F1={f1_cnn_d:.4f}  LGBM_F1={f1_lgbm_d:.4f}")

    s2_chi2, s2_p, s2_n01, s2_n10 = mcnemar_hesapla(
        dogru_cnn_dosya, dogru_lgbm_dosya,
        "CNN Baseline vs LightGBM Baseline (dosya bazli)")
    sonuclar["cnn_vs_lgbm"] = {"chi2": s2_chi2, "p": s2_p,
                                "n01": s2_n01, "n10": s2_n10,
                                "f1_a": f1_cnn_d, "f1_b": f1_lgbm_d}

    s3_chi2, s3_p, s3_n01, s3_n10 = mcnemar_hesapla(
        dogru_lgbm_bl, dogru_lgbm_aug,
        "LGBM Baseline vs LGBM Augmented")
    sonuclar["lgbm_bl_vs_aug"] = {"chi2": s3_chi2, "p": s3_p,
                                   "n01": s3_n01, "n10": s3_n10,
                                   "f1_a": f1_lgbm_bl, "f1_b": f1_lgbm_aug}

    print(f"\n  Anlamlilik: *** p<0.001  ** p<0.01  * p<0.05  ns: anlamli degil")
    print("=" * 65)

    # ── Kaydet ───────────────────────────────────────────────────────────────
    cikti = RESULT_DIR / "mcnemar_sonuclar.json"
    RESULT_DIR.mkdir(exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)
    print(f"\nSonuclar kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
