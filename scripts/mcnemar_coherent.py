"""H1 (tutarli-hedef) McNemar: augmentasyon akustik olarak tutarli dusuk-kaynakli
siniflari (kopek/kus/hava) anlamli iyilestiriyor mu? Catch-all diger_hayvanlar kontrol.

Paylasilan baseline (cnn_baseline_iter_best.pt) vs H1 augmented (cnn_aug_coherent_H1.pt).
Sadece hedef siniflarin test segmentlerinde McNemar (duyarli test). Yeni egitim yok.
Sonuc results/mcnemar_coherent.json'a kaydedilir.

Kullanim: python scripts/mcnemar_coherent.py
"""
import sys, json, argparse
from pathlib import Path
from datetime import datetime
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
# Tutarli hedefler (augmente edilenler) + catch-all kontrol
TUTARLI = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
KONTROL = ["diger_hayvanlar"]
HEDEF = TUTARLI + KONTROL
HEDEF_IDX = [SINIF_SIRASI.index(s) for s in HEDEF]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug_ckpt", default="cnn_aug_coherent_H1.pt",
                    help="Augmented checkpoint dosya adi (models/ icinde)")
    ap.add_argument("--baseline_ckpt", default="cnn_baseline_iter_best.pt")
    ap.add_argument("--out", default="results/mcnemar_coherent.json")
    ap.add_argument("--etiket", default="H1", help="Kosu etiketi (rapor icin)")
    return ap.parse_args()


def mcnemar(dogru_a, dogru_b):
    """dogru_a/b: bool array (baseline/augmented dogru mu). Yates duzeltmeli."""
    n01 = int(np.sum(~dogru_a &  dogru_b))   # baseline yanlis, aug dogru
    n10 = int(np.sum( dogru_a & ~dogru_b))   # baseline dogru, aug yanlis
    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10
    c = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = float(1 - chi2_dist.cdf(c, df=1))
    return float(c), p, n01, n10


def anlam(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def tahmin_et(ckpt, dl):
    model = SpectrogramCNN(n_sinif=N_SINIF).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()
    pr, gt = [], []
    with torch.no_grad():
        for batch in dl:
            x, y = batch[0], batch[1]
            pr.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            gt.extend(y.numpy() if hasattr(y, "numpy") else y)
    return np.array(pr), np.array(gt)


def main():
    args = parse_args()
    BASELINE_CKPT = MODEL_DIR / args.baseline_ckpt
    AUG_CKPT      = MODEL_DIR / args.aug_ckpt

    print("Veri bolunuyor (seed=42)...")
    _, _, df_test = veri_bol(PARQUET, ROOT)
    dl = dataloader_olustur(df_test, ROOT, batch_size=64, shuffle=False)
    print(f"Test segmenti: {len(dl.dataset):,}")

    print(f"Baseline tahminleri  ({BASELINE_CKPT.name})...")
    pred_b, gt = tahmin_et(BASELINE_CKPT, dl)
    print(f"Augmented tahminleri ({AUG_CKPT.name})...")
    pred_a, gt2 = tahmin_et(AUG_CKPT, dl)
    assert np.array_equal(gt, gt2), "Gercek etiketler eslesmedi!"

    dogru_b = (pred_b == gt)
    dogru_a = (pred_a == gt)

    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"), "etiket": args.etiket,
             "baseline_ckpt": BASELINE_CKPT.name, "aug_ckpt": AUG_CKPT.name,
             "n_test": int(len(gt)), "testler": {}}

    print("\n" + "=" * 66)
    print(f"  {args.etiket} McNemar — baseline vs tutarli-hedef augmented (Yates, df=1)")
    print("=" * 66)

    # 1) Global saglama
    c, p, n01, n10 = mcnemar(dogru_b, dogru_a)
    print(f"  GLOBAL (26 sinif, n={len(gt):,})            chi2={c:6.3f}  p={p:.4f} {anlam(p)}  (n01={n01}, n10={n10})")
    sonuc["testler"]["global"] = {"n": int(len(gt)), "chi2": c, "p": p, "anlam": anlam(p), "n01": n01, "n10": n10}

    # 2) Sadece TUTARLI hedefler (kopek+kus+hava) — asil hipotez
    idx_tut = [SINIF_SIRASI.index(s) for s in TUTARLI]
    mask = np.isin(gt, idx_tut)
    c, p, n01, n10 = mcnemar(dogru_b[mask], dogru_a[mask])
    print(f"  TUTARLI 3 (kopek+kus+hava, n={int(mask.sum()):,})   chi2={c:6.3f}  p={p:.4f} {anlam(p)}  (n01={n01}, n10={n10})")
    sonuc["testler"]["tutarli_3"] = {"siniflar": TUTARLI, "n": int(mask.sum()), "chi2": c, "p": p, "anlam": anlam(p), "n01": n01, "n10": n10}

    # 3) Sinif basina (tutarli + kontrol)
    print("\n  Sinif basina (true=ilgili sinif segmentleri):")
    sonuc["testler"]["sinif_basina"] = {}
    for s, idx in zip(HEDEF, HEDEF_IDX):
        m = (gt == idx)
        c, p, n01, n10 = mcnemar(dogru_b[m], dogru_a[m])
        acc_b = float(dogru_b[m].mean()); acc_a = float(dogru_a[m].mean())
        rol = "kontrol" if s in KONTROL else "tutarli"
        print(f"    {s:<16}[{rol:>7}] n={int(m.sum()):>5}  acc {acc_b:.3f}->{acc_a:.3f}  "
              f"chi2={c:6.3f} p={p:.4f} {anlam(p)} (n01={n01},n10={n10})")
        sonuc["testler"]["sinif_basina"][s] = {"rol": rol, "n": int(m.sum()),
            "acc_baseline": acc_b, "acc_augmented": acc_a,
            "chi2": c, "p": p, "anlam": anlam(p), "n01": n01, "n10": n10}

    out = ROOT / args.out
    out.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
