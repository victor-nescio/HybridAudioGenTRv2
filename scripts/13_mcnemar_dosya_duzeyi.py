"""Dosya duzeyinde McNemar — segment bagimliligi duzeltmesi.

Ayni kayittan turetilen segmentler korelasyonlu oldugundan segment duzeyi
McNemar bagimsizlik varsayimini ihlal edebilir. Bu script her test DOSYASI
icin cogunluk oyu tahmini uretir ve McNemar'i dosya duzeyinde tekrarlar
(baseline vs H1/C2/C3/C4). Yeni egitim yok; yalnizca cikarim.

Sanity: segment duzeyi tutarli-3 chi2 degerleri, yayimlanan degerlerle
(25.5 / 20.6 / 23.1 / 66.4) karsilastirilarak hat dogrulanir.

Kullanim: python scripts/13_mcnemar_dosya_duzeyi.py
Cikti   : results/mcnemar_dosya_duzeyi.json
"""
import sys, json
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
import torch
from scipy.stats import chi2 as chi2_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import dataloader_olustur, veri_bol
from src.models.cnn import SpectrogramCNN

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
PARQUET = ROOT / "data" / "processed" / "veri_listesi.parquet"
MODEL_DIR = ROOT / "models"
N_SINIF = 26

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
TUTARLI = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
KONTROL = ["diger_hayvanlar"]
HEDEF = TUTARLI + KONTROL
KONFIG = {
    "H1": "cnn_aug_coherent_H1.pt",
    "C2": "cnn_aug_coherent_coh2.pt",
    "C3": "cnn_aug_coherent_coh3.pt",
    "C4": "cnn_aug_coherent_coh4.pt",
}
BASELINE = "cnn_baseline_iter_best.pt"


def mcnemar(dogru_a, dogru_b):
    n01 = int(np.sum(~dogru_a &  dogru_b))
    n10 = int(np.sum( dogru_a & ~dogru_b))
    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10
    c = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(c), float(1 - chi2_dist.cdf(c, df=1)), n01, n10


def anlam(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def tahmin_et(ckpt, dl):
    model = SpectrogramCNN(n_sinif=N_SINIF).to(DEVICE)
    ck = torch.load(MODEL_DIR / ckpt, map_location=DEVICE)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    pr = []
    with torch.no_grad():
        for batch in dl:
            pr.extend(model(batch[0].to(DEVICE)).argmax(1).cpu().numpy())
    return np.array(pr)


def dosya_coğunluk(pred, dosyalar):
    """Her dosya icin segment tahminlerinin cogunluk oyu (esitlikte kucuk indeks)."""
    oy = {}
    for p, f in zip(pred, dosyalar):
        oy.setdefault(f, []).append(int(p))
    return {f: Counter(v).most_common(1)[0][0] for f, v in oy.items()}


def blok_testler(dogru_b, dogru_a, gt_ad, n_etiket):
    sonuc = {}
    c, p, n01, n10 = mcnemar(dogru_b, dogru_a)
    sonuc["global"] = {"n": int(len(dogru_b)), "chi2": c, "p": p,
                       "anlam": anlam(p), "n01": n01, "n10": n10}
    m3 = np.isin(gt_ad, TUTARLI)
    c, p, n01, n10 = mcnemar(dogru_b[m3], dogru_a[m3])
    sonuc["tutarli_3"] = {"n": int(m3.sum()), "chi2": c, "p": p,
                          "anlam": anlam(p), "n01": n01, "n10": n10}
    sonuc["sinif_basina"] = {}
    for s in HEDEF:
        m = (gt_ad == s)
        c, p, n01, n10 = mcnemar(dogru_b[m], dogru_a[m])
        sonuc["sinif_basina"][s] = {
            "rol": "kontrol" if s in KONTROL else "hedef", "n": int(m.sum()),
            "acc_baseline": float(dogru_b[m].mean()),
            "acc_augmented": float(dogru_a[m].mean()),
            "chi2": c, "p": p, "anlam": anlam(p), "n01": n01, "n10": n10}
    print(f"    [{n_etiket}] global chi2={sonuc['global']['chi2']:.2f} "
          f"p={sonuc['global']['p']:.3g} {sonuc['global']['anlam']} | "
          f"hedef-3 chi2={sonuc['tutarli_3']['chi2']:.2f} "
          f"p={sonuc['tutarli_3']['p']:.3g} {sonuc['tutarli_3']['anlam']}")
    return sonuc


def main():
    print("Veri bolunuyor (seed=42)...")
    _, _, df_test = veri_bol(PARQUET, ROOT)
    dl = dataloader_olustur(df_test, ROOT, batch_size=64, shuffle=False)
    indeks = dl.dataset.indeks
    dosyalar = np.array([w for w, _, _ in indeks])
    gt = np.array([s for _, s, _ in indeks])
    gt_ad = np.array([SINIF_SIRASI[i] for i in gt])
    print(f"Test: {len(gt):,} segment, {len(set(dosyalar)):,} dosya")

    print(f"Baseline cikarimi ({BASELINE})...")
    pred_b = tahmin_et(BASELINE, dl)
    dogru_b_seg = pred_b == gt
    oy_b = dosya_coğunluk(pred_b, dosyalar)

    dosya_sinif = {}
    for f, s in zip(dosyalar, gt):
        dosya_sinif[f] = int(s)
    dosya_liste = sorted(dosya_sinif)
    d_gt = np.array([dosya_sinif[f] for f in dosya_liste])
    d_gt_ad = np.array([SINIF_SIRASI[i] for i in d_gt])
    d_dogru_b = np.array([oy_b[f] == dosya_sinif[f] for f in dosya_liste])

    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"),
             "baseline_ckpt": BASELINE,
             "n_segment": int(len(gt)), "n_dosya": int(len(dosya_liste)),
             "konfigurasyonlar": {}}

    for etiket, ckpt in KONFIG.items():
        print(f"\n{etiket} cikarimi ({ckpt})...")
        pred_a = tahmin_et(ckpt, dl)
        dogru_a_seg = pred_a == gt
        # sanity: segment duzeyi tutarli-3
        m3 = np.isin(gt_ad, TUTARLI)
        c_seg, p_seg, _, _ = mcnemar(dogru_b_seg[m3], dogru_a_seg[m3])
        print(f"  sanity segment hedef-3: chi2={c_seg:.1f} (yayimlanan ile karsilastir)")
        oy_a = dosya_coğunluk(pred_a, dosyalar)
        d_dogru_a = np.array([oy_a[f] == dosya_sinif[f] for f in dosya_liste])
        sonuc["konfigurasyonlar"][etiket] = {
            "ckpt": ckpt,
            "sanity_segment_tutarli3_chi2": c_seg,
            "dosya_duzeyi": blok_testler(d_dogru_b, d_dogru_a, d_gt_ad, f"{etiket} dosya"),
        }

    out = ROOT / "results" / "mcnemar_dosya_duzeyi.json"
    out.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
