"""
CNN Baseline vs LightGBM Baseline — dosya bazli McNemar.

CNN segment tahminleri majority vote ile dosya duzeyine tasir.
LGBM segment tahminleri majority vote ile dosya duzeyine tasir.
Ortak test dosyalari uzerinde McNemar hesaplanir.

Kullanim:
    python scripts/12d_mcnemar_cnn_vs_lgbm.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dataset import dataloader_olustur, veri_bol
from src.models.cnn import SpectrogramCNN

PARQUET_LISTE = ROOT / "data" / "processed" / "veri_listesi.parquet"
PARQUET_LGBM  = ROOT / "data" / "processed" / "lgbm_ozellikler.parquet"
MODEL_DIR     = ROOT / "models"
RESULT_DIR    = ROOT / "results" / "v1"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
N_SINIF       = 26
SEED          = 42

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]


def ozellik_sutunlari(df):
    return sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))


def cnn_tahmin_et(dl_test):
    ckpt = MODEL_DIR / "cnn_baseline_iter_best.pt"
    model = SpectrogramCNN(n_sinif=N_SINIF).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    preds, gercek = [], []
    with torch.no_grad():
        for batch in dl_test:
            x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) \
                   else (batch["mel"], batch["sinif_idx"])
            logits = model(x.to(DEVICE))
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            gercek.extend(y.numpy() if hasattr(y, "numpy") else y)
    f1 = f1_score(gercek, preds, average="macro", zero_division=0)
    print(f"  CNN baseline segment test_F1={f1:.4f}")
    return np.array(preds), np.array(gercek)


def lgbm_egit_tahmin(df_tr, df_vl, df_te, sut):
    import lightgbm as lgb
    def xy(df):
        return df[sut].fillna(0).values.astype(np.float32), df["sinif_idx"].values
    X_tr, y_tr = xy(df_tr); X_vl, y_vl = xy(df_vl); X_te, y_te = xy(df_te)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_te = scaler.transform(X_te)

    params_yolu = ROOT / "checkpoints" / "lgbm_best_params.json"
    params = json.load(open(params_yolu)) if params_yolu.exists() else dict(
        n_estimators=500, num_leaves=127, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0)
    params.update({"device": "gpu", "verbose": -1, "class_weight": "balanced",
                   "objective": "multiclass", "num_class": N_SINIF,
                   "metric": "multi_logloss", "random_state": SEED})

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=100)])
    pred_te = model.predict(X_te)
    f1 = f1_score(y_te, pred_te, average="macro", zero_division=0)
    print(f"  LGBM baseline segment test_F1={f1:.4f}  ({(time.time()-t0)/60:.1f} dk)")
    return pred_te, y_te


def dosya_bazli_majority(pred_seg, df_meta_or_dosya_arr):
    """
    pred_seg: segment tahminleri (np array)
    df_meta_or_dosya_arr: her segment icin dosya adi (pd.Series ya da df ile n_segs)
    Dondurulenler: {dosya: tahmin_sinif}
    """
    if isinstance(df_meta_or_dosya_arr, pd.DataFrame):
        # df_meta: her satir bir dosya, n_segs sutunu var
        dosya_pred = {}
        sayac = 0
        for _, row in df_meta_or_dosya_arr.iterrows():
            d = row["dosya"]
            n = int(row.get("n_segs", 1))
            segs = pred_seg[sayac: sayac + n]
            sayac += n
            if len(segs) > 0:
                dosya_pred[d] = Counter(segs).most_common(1)[0][0]
        return dosya_pred
    else:
        # dosya_arr: her segmente karsilik gelen dosya adi
        dosya_arr = np.array(df_meta_or_dosya_arr)
        dosya_pred = {}
        for i, d in enumerate(dosya_arr):
            dosya_pred.setdefault(d, []).append(pred_seg[i])
        return {d: Counter(v).most_common(1)[0][0] for d, v in dosya_pred.items()}


def mcnemar(dogru1, dogru2, etiket):
    n01 = int(np.sum(~dogru1 &  dogru2))
    n10 = int(np.sum( dogru1 & ~dogru2))
    if n01 + n10 == 0:
        return 0.0, 1.0, 0, 0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p    = 1 - chi2_dist.cdf(stat, df=1)
    anlam = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {etiket:<50} chi2={stat:.1f}  p={p:.6f} {anlam}  (n={n01+n10})")
    return stat, p, n01, n10


def main():
    print("=" * 65)
    print("  CNN vs LGBM — Dosya Bazli McNemar")
    print("=" * 65)

    # Split
    print("\nVeri bolunuyor (seed=42)...")
    df_tr_m, df_vl_m, df_te_m = veri_bol(PARQUET_LISTE, ROOT)
    print(f"  Test: {len(df_te_m):,} dosya")

    # CNN dataloader ve tahmin
    print("\n[1/2] CNN tahminleri uretiliyor...")
    dl_test = dataloader_olustur(df_te_m, ROOT, batch_size=64, shuffle=False)
    print(f"  {len(dl_test.dataset):,} test segmenti")
    pred_cnn_seg, gercek_cnn_seg = cnn_tahmin_et(dl_test)

    # CNN dosya bazli
    cnn_dosya_pred = dosya_bazli_majority(pred_cnn_seg, df_te_m)

    # LGBM egitim ve tahmin
    print("\n[2/2] LGBM baseline egitiliyor ve tahmin uretiliyor...")
    df_lgbm = pd.read_parquet(PARQUET_LGBM)
    tr_d = set(df_tr_m["dosya"]); vl_d = set(df_vl_m["dosya"]); te_d = set(df_te_m["dosya"])
    df_tr = df_lgbm[df_lgbm["dosya"].isin(tr_d)].reset_index(drop=True)
    df_vl = df_lgbm[df_lgbm["dosya"].isin(vl_d)].reset_index(drop=True)
    df_te = df_lgbm[df_lgbm["dosya"].isin(te_d)].reset_index(drop=True)
    print(f"  {len(df_te):,} LGBM test segmenti")
    sut = ozellik_sutunlari(df_tr)
    pred_lgbm_seg, _ = lgbm_egit_tahmin(df_tr, df_vl, df_te, sut)

    # LGBM dosya bazli
    lgbm_dosya_pred = dosya_bazli_majority(pred_lgbm_seg, df_te["dosya"])

    # Ortak dosyalar
    ortak = sorted(set(cnn_dosya_pred.keys()) & set(lgbm_dosya_pred.keys()))
    print(f"\n  Ortak dosya sayisi: {len(ortak)}")

    # Gercek siniflar (df_te_m'den)
    gercek_dosya = {row["dosya"]: int(row["sinif_idx"])
                    for _, row in df_te_m.iterrows()}

    cnn_p  = np.array([cnn_dosya_pred[d]  for d in ortak])
    lgbm_p = np.array([lgbm_dosya_pred[d] for d in ortak])
    grc    = np.array([gercek_dosya[d]    for d in ortak])

    f1_cnn  = f1_score(grc, cnn_p,  average="macro", zero_division=0)
    f1_lgbm = f1_score(grc, lgbm_p, average="macro", zero_division=0)
    print(f"  CNN  dosya-bazli F1 = {f1_cnn:.4f}")
    print(f"  LGBM dosya-bazli F1 = {f1_lgbm:.4f}")

    print(f"\n{'='*65}")
    print("  McNemar Test Sonucu  (Yates duzeltmeli, df=1)")
    chi2_stat, p_val, n01, n10 = mcnemar(
        cnn_p == grc, lgbm_p == grc,
        "CNN Baseline vs LightGBM Baseline (dosya bazli)")
    print("=" * 65)

    sonuc = {
        "cnn_vs_lgbm_dosya": {
            "chi2": chi2_stat, "p": p_val, "n01": n01, "n10": n10,
            "n_dosya": len(ortak),
            "f1_cnn_dosya": f1_cnn, "f1_lgbm_dosya": f1_lgbm
        }
    }
    cikti = RESULT_DIR / "mcnemar_cnn_vs_lgbm.json"
    RESULT_DIR.mkdir(exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"\nSonuc kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
