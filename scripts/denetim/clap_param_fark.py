# -*- coding: utf-8 -*-
"""v1 ince-ayarlı CLAP checkpoint'leri hazır CLAP'ten ne kadar farklı?
- metin kolu / ses kolu parametre farkları
- NaN içeren tensörler (parametre mi, BatchNorm tamponu mu?)
- AudioLDM2'ye entegre edilmiş checkpoint'te (üretimde kullanılan) metin kolu değişmiş mi?"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(r"F:\Drthesiscode")
from audioldm2 import download_checkpoint

orj = torch.load(download_checkpoint("audioldm2-full"), map_location="cpu", mmap=True)["state_dict"]
orj_clap = {k[len("clap."):]: v for k, v in orj.items() if k.startswith("clap.model.")}


def ozet(ad, sd, onek=""):
    sonuc = {"metin_degisen": 0, "metin_toplam": 0, "metin_maks_fark": 0.0,
             "ses_degisen": 0, "ses_toplam": 0, "nan_tensorler": []}
    for k, v in orj_clap.items():
        k2 = onek + k
        if k2 not in sd:
            continue
        w = sd[k2]
        if not torch.is_tensor(w) or not w.is_floating_point():
            continue
        if torch.isnan(w).any():
            sonuc["nan_tensorler"].append(k)
        kol = "metin" if (".text_" in k or "text_projection" in k or ".text_branch" in k) else (
              "ses" if (".audio_" in k or "audio_projection" in k) else None)
        if kol is None:
            continue
        fark = (w.float() - v.float()).abs().max().item() if not torch.isnan(w).any() else float("nan")
        sonuc[f"{kol}_toplam"] += 1
        if not (fark == 0.0):
            sonuc[f"{kol}_degisen"] += 1
        if kol == "metin" and fark == fark:
            sonuc["metin_maks_fark"] = max(sonuc["metin_maks_fark"], fark)
    n = len(sonuc["nan_tensorler"])
    sonuc["nan_tensor_sayisi"] = n
    sonuc["nan_tensorler"] = sonuc["nan_tensorler"][:8]
    print(ad, json.dumps(sonuc, ensure_ascii=False), flush=True)


for ad in ("clap_bbc_best.pt", "clap_bbc_dongu2_best.pt"):
    ck = torch.load(ROOT / "checkpoints" / ad, map_location="cpu", mmap=True)
    ozet(ad, ck["model_state"])
    del ck

for ad in ("audioldm2_clap_finetuned.pth", "audioldm2_clap_dongu2_finetuned.pth"):
    sd = torch.load(ROOT / "checkpoints" / ad, map_location="cpu", mmap=True)
    sd = sd.get("state_dict", sd)
    ozet(ad, sd, onek="clap.")
    del sd
