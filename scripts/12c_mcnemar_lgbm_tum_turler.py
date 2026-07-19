"""
LGBM McNemar testi — Tur1-3 ve Tur4 icin dogru yeniden egitim.

Baseline bir kez egitirilir, her tur icin ayri augmented model olusturulur.
Her iki karsilastirma icin McNemar hesaplanir ve sonuclar kaydedilir.

Kullanim:
    python scripts/12c_mcnemar_lgbm_tum_turler.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import veri_bol

PARQUET_LISTE    = ROOT / "data" / "processed" / "veri_listesi.parquet"
PARQUET_LGBM     = ROOT / "data" / "processed" / "lgbm_ozellikler.parquet"
SENTETIK_CACHE   = ROOT / "data" / "processed" / "sentetik_lgbm_ozellikler.parquet"
SENTETIK_TUR4    = ROOT / "data" / "synthetic_tur4"
RESULT_DIR       = ROOT / "results"
SEED             = 42
N_SINIF          = 26

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


def lgbm_egit_tahmin(df_train, df_val, df_test, sut, scaler_fit, etiket):
    """
    scaler_fit: (X_tr, y_tr) uzerinde fit edilmis scaler. None ise yeni scaler olustur.
    Dondurulenler: (pred_test, y_test, f1_test, scaler)
    """
    import lightgbm as lgb

    def xy(df):
        return df[sut].fillna(0).values.astype(np.float32), df["sinif_idx"].values

    X_tr, y_tr = xy(df_train)
    X_vl, y_vl = xy(df_val)
    X_te, y_te = xy(df_test)

    if scaler_fit is None:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
    else:
        scaler = scaler_fit
        X_tr = scaler.transform(X_tr)
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

    params.update({"device": "gpu", "verbose": -1,
                   "class_weight": "balanced",
                   "objective": "multiclass", "num_class": N_SINIF,
                   "metric": "multi_logloss", "random_state": SEED})

    t0 = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=100)])

    pred_te = model.predict(X_te)
    f1_te   = f1_score(y_te, pred_te, average="macro", zero_division=0)
    sure_dk = (time.time() - t0) / 60
    print(f"  [{etiket}] test_F1={f1_te:.4f}  ({sure_dk:.1f} dk)")
    return pred_te, y_te, f1_te, scaler


def tur4_ozellik_cikar():
    from src.data.features import duz_ozellik_cikar, SEGMENT_SN
    from src.data.preprocessing import onisle

    hedef = ["acil_servis", "diger_hayvanlar", "dini_mekan", "insaat_sesleri"]
    satirlar = []
    for sinif in hedef:
        klasor = SENTETIK_TUR4 / sinif
        if not klasor.exists():
            continue
        wavlar = sorted([f for f in klasor.glob("*.wav") if f.stat().st_size > 1000])
        print(f"  {sinif}: {len(wavlar)} WAV...")
        for wav in wavlar:
            try:
                ses, sr = onisle(wav)
                seg_uzunluk = int(SEGMENT_SN * sr)
                for s_idx in range(max(1, len(ses) // seg_uzunluk)):
                    seg = ses[s_idx * seg_uzunluk: (s_idx + 1) * seg_uzunluk]
                    if len(seg) < seg_uzunluk // 2:
                        continue
                    oz = duz_ozellik_cikar(seg, sr)
                    satir = {"sinif": sinif, "sinif_idx": SINIF_IDX[sinif],
                             "dosya": wav.name, "segment_idx": s_idx}
                    satir.update({f"f{j}": v for j, v in enumerate(oz)})
                    satirlar.append(satir)
            except Exception as e:
                print(f"  [!] {wav.name}: {e}")
    df = pd.DataFrame(satirlar)
    print(f"  {len(df)} segment cikarildi")
    return df


def mcnemar(dogru1, dogru2, etiket):
    n01 = int(np.sum(~dogru1 &  dogru2))
    n10 = int(np.sum( dogru1 & ~dogru2))
    if n01 + n10 == 0:
        print(f"  {etiket}: n01=n10=0, test yapilamadi")
        return 0.0, 1.0, 0, 0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p    = 1 - chi2_dist.cdf(stat, df=1)
    anlam = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {etiket:<45} chi2={stat:.3f}  p={p:.4f} {anlam}  (n01={n01}, n10={n10})")
    return stat, p, n01, n10


def sentetik_hizala(df_synt, sut):
    eksik = [c for c in sut if c not in df_synt.columns]
    if eksik:
        for c in eksik:
            df_synt[c] = 0.0
    meta = ["sinif", "sinif_idx", "dosya", "segment_idx"]
    return df_synt[[c for c in meta if c in df_synt.columns] + sut]


def main():
    print("=" * 65)
    print("  LGBM McNemar — Tur1-3 ve Tur4 Yeniden Egitim")
    print("=" * 65)

    # Split
    print("\nVeri bolunuyor (seed=42)...")
    df_tr_m, df_vl_m, df_te_m = veri_bol(PARQUET_LISTE, ROOT)

    df_lgbm = pd.read_parquet(PARQUET_LGBM)
    tr_d = set(df_tr_m["dosya"]); vl_d = set(df_vl_m["dosya"]); te_d = set(df_te_m["dosya"])

    df_tr = df_lgbm[df_lgbm["dosya"].isin(tr_d)].reset_index(drop=True)
    df_vl = df_lgbm[df_lgbm["dosya"].isin(vl_d)].reset_index(drop=True)
    df_te = df_lgbm[df_lgbm["dosya"].isin(te_d)].reset_index(drop=True)
    sut   = ozellik_sutunlari(df_tr)
    print(f"  Train={len(df_tr):,}  Val={len(df_vl):,}  Test={len(df_te):,} seg  |  {len(sut)} ozellik")

    # ── 1. BASELINE (bir kez) ─────────────────────────────────────────────────
    print("\n[1/3] LGBM BASELINE egitiliyor...")
    pred_bl, y_te, f1_bl, scaler = lgbm_egit_tahmin(df_tr, df_vl, df_te, sut, None, "baseline")

    # ── 2. TUR1-3 AUGMENTED ───────────────────────────────────────────────────
    print("\n[2/3] Tur1-3 sentetik ozellikler yukleniyor (cache)...")
    df_synt13 = pd.read_parquet(SENTETIK_CACHE)
    print(f"  {len(df_synt13)} satir  |  siniflar: {df_synt13['sinif'].value_counts().to_dict()}")
    df_synt13_h = sentetik_hizala(df_synt13, sut)
    df_tr_aug13 = pd.concat([df_tr, df_synt13_h], ignore_index=True)
    print(f"  Augmented: {len(df_tr):,} + {len(df_synt13):,} = {len(df_tr_aug13):,}")

    print("\n  Tur1-3 augmented egitiliyor...")
    pred_aug13, _, f1_aug13, _ = lgbm_egit_tahmin(
        df_tr_aug13, df_vl, df_te, sut, scaler, "augmented_tur13")

    # ── 3. TUR4 AUGMENTED ────────────────────────────────────────────────────
    print("\n[3/3] Tur4 sentetik ozellikler cikariliyor (synthetic_tur4/)...")
    df_synt4 = tur4_ozellik_cikar()
    df_synt4_h = sentetik_hizala(df_synt4, sut)
    df_tr_aug4 = pd.concat([df_tr, df_synt4_h], ignore_index=True)
    print(f"  Augmented: {len(df_tr):,} + {len(df_synt4):,} = {len(df_tr_aug4):,}")

    print("\n  Tur4 augmented egitiliyor...")
    pred_aug4, _, f1_aug4, _ = lgbm_egit_tahmin(
        df_tr_aug4, df_vl, df_te, sut, scaler, "augmented_tur4")

    # ── MCNEMAR ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  McNemar Sonuclari  (Yates duzeltmeli, df=1)")
    print(f"  Baseline test_F1 = {f1_bl:.4f}")
    print(f"  Tur1-3 aug F1    = {f1_aug13:.4f}  (delta={f1_aug13-f1_bl:+.4f})")
    print(f"  Tur4   aug F1    = {f1_aug4:.4f}  (delta={f1_aug4-f1_bl:+.4f})")
    print()

    s13_c, s13_p, s13_n01, s13_n10 = mcnemar(
        pred_bl == y_te, pred_aug13 == y_te, "LGBM Baseline vs Augmented (Tur1-3)")
    s4_c,  s4_p,  s4_n01,  s4_n10  = mcnemar(
        pred_bl == y_te, pred_aug4  == y_te, "LGBM Baseline vs Augmented (Tur4)")
    print("=" * 65)

    sonuc = {
        "lgbm_tur13_bl_vs_aug": {
            "chi2": s13_c, "p": s13_p, "n01": s13_n01, "n10": s13_n10,
            "f1_baseline": f1_bl, "f1_augmented": f1_aug13,
            "delta": round(f1_aug13 - f1_bl, 4), "n_segs": int(len(y_te))
        },
        "lgbm_tur4_bl_vs_aug": {
            "chi2": s4_c, "p": s4_p, "n01": s4_n01, "n10": s4_n10,
            "f1_baseline": f1_bl, "f1_augmented": f1_aug4,
            "delta": round(f1_aug4 - f1_bl, 4), "n_segs": int(len(y_te))
        }
    }

    cikti = RESULT_DIR / "mcnemar_lgbm_tum_turler.json"
    RESULT_DIR.mkdir(exist_ok=True)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    print(f"\nSonuclar kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
