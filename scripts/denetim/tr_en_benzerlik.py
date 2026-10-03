# -*- coding: utf-8 -*-
"""R2.8: Türkçe ve İngilizce promptlar CLAP metin uzayında anlamsal olarak eşdeğer mi?
Her sınıfın 5 prompt çifti (aynı indeks = çeviri çifti) için kosinüs benzerlikleri:
  çift (TR_i–EN_i), sınıf içi EN (EN_i–EN_j), sınıflar arası EN, ve TR→EN eşleştirme doğruluğu
  (her TR promptu için en yakın EN promptunun kendi çevirisi / kendi sınıfı olma oranı).
CLAP metin yolu AudioLDM2 ile aynıdır (RoBERTa pooler_output + text_projection).
Kullanım: python tr_en_benzerlik.py [--ckpt checkpoints/clap_v2_dongu1_best.pt] [--cpu]"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="İnce-ayarlı CLAP (yoksa hazır CLAP)")
ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
if a.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("clapft", ROOT / "scripts" / "08_clap_finetune.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

cfg = M.FinetuneConfig()          # surum v2 -> pooler_output yolu
model = M.CLAPFinetune(cfg).to(M.DEVICE).eval()
etiket = "hazir_CLAP"
if a.ckpt:
    ck = torch.load(ROOT / a.ckpt, map_location=M.DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    etiket = Path(a.ckpt).stem

P = json.loads((ROOT / "data" / "processed" / "turkce_promptlar.json").read_text(encoding="utf-8"))
siniflar = sorted(P)
with torch.no_grad():
    EN = {s: model.metin_embedding(P[s]["ingilizce"]).float().cpu() for s in siniflar}
    TR = {s: model.metin_embedding(P[s]["turkce"]).float().cpu() for s in siniflar}

cift, sinif_ici, siniflar_arasi = [], [], []
dogru_cift = dogru_sinif = toplam = 0
tum_en = torch.cat([EN[s] for s in siniflar])
en_sinif = [s for s in siniflar for _ in range(len(P[s]["ingilizce"]))]
en_idx = [i for s in siniflar for i in range(len(P[s]["ingilizce"]))]
for s in siniflar:
    n = min(len(EN[s]), len(TR[s]))
    C = TR[s][:n] @ EN[s][:n].T
    cift += C.diag().tolist()
    E = EN[s] @ EN[s].T
    sinif_ici += E[~torch.eye(len(E), dtype=bool)].tolist()
    for t in siniflar:
        if t > s:
            siniflar_arasi += (EN[s] @ EN[t].T).flatten().tolist()
    en_yakin = (TR[s][:n] @ tum_en.T).argmax(1)
    for i, j in enumerate(en_yakin.tolist()):
        toplam += 1
        dogru_sinif += en_sinif[j] == s
        dogru_cift += (en_sinif[j] == s and en_idx[j] == i)

ort = lambda x: round(sum(x) / len(x), 4)
sonuc = {"model": etiket, "sinif_sayisi": len(siniflar),
         "cift_TR_EN_ort_kos": ort(cift), "sinif_ici_EN_ort_kos": ort(sinif_ici),
         "siniflar_arasi_EN_ort_kos": ort(siniflar_arasi),
         "TR_en_yakin_EN_kendi_cevirisi": round(dogru_cift / toplam, 4),
         "TR_en_yakin_EN_kendi_sinifi": round(dogru_sinif / toplam, 4),
         "sans_kendi_sinifi": round(5 / len(tum_en), 4)}
print(json.dumps(sonuc, ensure_ascii=False, indent=1))
cikti = ROOT / "results" / "denetim" / f"tr_en_benzerlik_{etiket}.json"
cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi:", cikti)
