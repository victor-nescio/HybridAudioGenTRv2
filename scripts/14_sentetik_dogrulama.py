"""Sentetik kliplerin siniflandirici temelli dogrulanmasi.

Baseline CNN'in her uretim konfigurasyonundaki sentetik klipleri yuzde kacini
hedef sinifa atadigini olcer (classifier-based validity). Algisal dinleme
testinin ucuz ve nesnel vekilidir.

Kullanim: python scripts/14_sentetik_dogrulama.py
Cikti   : results/sentetik_dogrulama.json
"""
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import dataloader_olustur
from src.models.cnn import SpectrogramCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
KONFIG = {
    "H1": "synthetic_turcoh",
    "C2": "synthetic_turcoh2",
    "C3": "synthetic_turcoh3",
    "C4": "synthetic_turcoh4",
}
BASELINE = "cnn_baseline_iter_best.pt"


def main():
    model = SpectrogramCNN(n_sinif=N_SINIF).to(DEVICE)
    ck = torch.load(MODEL_DIR / BASELINE, map_location=DEVICE)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()

    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"),
             "model": BASELINE, "konfigurasyonlar": {}}

    for etiket, klasor in KONFIG.items():
        kok = ROOT / "data" / klasor
        satirlar = []
        for sinif_dir in sorted(p for p in kok.iterdir() if p.is_dir()):
            sinif = sinif_dir.name
            if sinif not in SINIF_SIRASI:
                continue
            for wav in sorted(sinif_dir.glob("*.wav")):
                satirlar.append({"dosya": str(wav.relative_to(ROOT)),
                                 "sinif": sinif,
                                 "sinif_idx": SINIF_SIRASI.index(sinif),
                                 "n_segs": 1})
        if not satirlar:
            continue
        df = pd.DataFrame(satirlar)
        dl = dataloader_olustur(df, ROOT, batch_size=32, shuffle=False)
        pred, gt = [], []
        with torch.no_grad():
            for batch in dl:
                pred.extend(model(batch[0].to(DEVICE)).argmax(1).cpu().numpy())
                gt.extend(batch[1].numpy())
        pred, gt = np.array(pred), np.array(gt)
        blok = {"n_klip": int(len(gt)),
                "hedefe_atama_orani": float((pred == gt).mean()),
                "sinif_basina": {}}
        for s in sorted(set(df["sinif"])):
            m = gt == SINIF_SIRASI.index(s)
            blok["sinif_basina"][s] = {
                "n": int(m.sum()),
                "hedefe_atama": float((pred[m] == gt[m]).mean()),
            }
        sonuc["konfigurasyonlar"][etiket] = blok
        print(f"{etiket}: {len(gt)} klip, hedefe atama {blok['hedefe_atama_orani']:.1%}")

    out = ROOT / "results" / "sentetik_dogrulama.json"
    out.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
