"""Kume-direncli (cluster-robust) anlamlilik analizi.

Segment duzeyi McNemar, ayni dosyadan gelen segmentlerin korelasyonunu yok
sayar; dosya duzeyi cogunluk-oyu McNemar ise bilgi kaybi nedeniyle gucsuzdur.
Bu script iki orta-yol test uygular (dosya = kume):

1) Wilcoxon isaretli-sira testi: dosya basina segment dogrulugu farki
   d_i = acc_aug(i) - acc_base(i) uzerinde (sifirlar dusulur, iki yonlu).
2) Kume bootstrap: dosyalar yerine koymali orneklenir (B=10.000), toplam
   segment dogrulugu farkinin dagilimi -> p ve %95 GA.

Kullanim: python scripts/13b_kume_direncli_test.py
Cikti   : results/kume_direncli_test.json (+ results/preds_test_seg.npz)
"""
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import dataloader_olustur, veri_bol
from src.models.cnn import SpectrogramCNN

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
PARQUET = ROOT / "data" / "processed" / "veri_listesi.parquet"
MODEL_DIR = ROOT / "models"
N_SINIF = 26
B_BOOT = 10_000
RNG = np.random.default_rng(42)

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
KONFIG = {
    "H1": "cnn_aug_coherent_H1.pt",
    "C2": "cnn_aug_coherent_coh2.pt",
    "C3": "cnn_aug_coherent_coh3.pt",
    "C4": "cnn_aug_coherent_coh4.pt",
}
BASELINE = "cnn_baseline_iter_best.pt"


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


def kume_testleri(dogru_b, dogru_a, dosya_id):
    """dosya_id: her segment icin dosya indeksi. Dondurur: wilcoxon + bootstrap."""
    n_dosya = dosya_id.max() + 1
    seg_say = np.bincount(dosya_id, minlength=n_dosya)
    top_b = np.bincount(dosya_id, weights=dogru_b, minlength=n_dosya)
    top_a = np.bincount(dosya_id, weights=dogru_a, minlength=n_dosya)
    gecerli = seg_say > 0
    acc_b = top_b[gecerli] / seg_say[gecerli]
    acc_a = top_a[gecerli] / seg_say[gecerli]
    d = acc_a - acc_b
    # 1) Wilcoxon (sifir farklar dusulur)
    d_nz = d[d != 0]
    if len(d_nz) >= 10:
        w, p_w = wilcoxon(d_nz, alternative="two-sided")
        p_w = float(p_w)
    else:
        w, p_w = float("nan"), float("nan")
    # 2) Kume bootstrap: dosyalari yerine koymali ornekle
    nb, na, ns = top_b[gecerli], top_a[gecerli], seg_say[gecerli]
    m = len(ns)
    fark_gozlenen = (na.sum() - nb.sum()) / ns.sum()
    farklar = np.empty(B_BOOT)
    for b in range(B_BOOT):
        idx = RNG.integers(0, m, m)
        farklar[b] = (na[idx].sum() - nb[idx].sum()) / ns[idx].sum()
    # iki yonlu p: isaret degisimi orani
    p_boot = 2 * min((farklar <= 0).mean(), (farklar >= 0).mean())
    p_boot = float(min(max(p_boot, 1.0 / B_BOOT), 1.0))
    ga = np.percentile(farklar, [2.5, 97.5])
    return {
        "n_dosya": int(m),
        "n_segment": int(ns.sum()),
        "fark_segment_acc": float(fark_gozlenen),
        "wilcoxon_n_sifir_disi": int(len(d_nz)),
        "wilcoxon_p": p_w,
        "bootstrap_p": p_boot,
        "bootstrap_ga95": [float(ga[0]), float(ga[1])],
    }


def main():
    print("Veri bolunuyor (seed=42)...")
    _, _, df_test = veri_bol(PARQUET, ROOT)
    dl = dataloader_olustur(df_test, ROOT, batch_size=64, shuffle=False)
    indeks = dl.dataset.indeks
    dosyalar = np.array([w for w, _, _ in indeks])
    gt = np.array([s for _, s, _ in indeks])
    gt_ad = np.array([SINIF_SIRASI[i] for i in gt])
    benzersiz, dosya_id = np.unique(dosyalar, return_inverse=True)
    print(f"Test: {len(gt):,} segment, {len(benzersiz):,} dosya")

    print(f"Baseline cikarimi ({BASELINE})...")
    pred_b = tahmin_et(BASELINE, dl)
    kayit = {"baseline": pred_b, "gt": gt, "dosya_id": dosya_id}

    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"),
             "baseline_ckpt": BASELINE, "B_bootstrap": B_BOOT,
             "konfigurasyonlar": {}}
    m3 = np.isin(gt_ad, TUTARLI)

    for etiket, ckpt in KONFIG.items():
        print(f"\n{etiket} cikarimi ({ckpt})...")
        pred_a = tahmin_et(ckpt, dl)
        kayit[etiket] = pred_a
        dogru_b = (pred_b == gt).astype(float)
        dogru_a = (pred_a == gt).astype(float)
        blok = {"hedef_3": kume_testleri(dogru_b[m3], dogru_a[m3],
                                         np.unique(dosya_id[m3], return_inverse=True)[1])}
        for s in TUTARLI + KONTROL:
            ms = gt_ad == s
            blok[s] = kume_testleri(dogru_b[ms], dogru_a[ms],
                                    np.unique(dosya_id[ms], return_inverse=True)[1])
        sonuc["konfigurasyonlar"][etiket] = blok
        h = blok["hedef_3"]
        print(f"  hedef-3: fark={h['fark_segment_acc']:+.4f}  "
              f"wilcoxon p={h['wilcoxon_p']:.4g}  bootstrap p={h['bootstrap_p']:.4g}  "
              f"GA95=[{h['bootstrap_ga95'][0]:+.4f}, {h['bootstrap_ga95'][1]:+.4f}]")

    np.savez_compressed(ROOT / "results" / "preds_test_seg.npz", **kayit)
    out = ROOT / "results" / "kume_direncli_test.json"
    out.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
