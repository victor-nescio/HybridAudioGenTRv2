# -*- coding: utf-8 -*-
"""CLAP tanı testi: hazır vs v1 ince-ayarlı CLAP; fp32 vs fp16 (autocast).
Ölçüler: ham ses gömmesinde NaN oranı, kNN sınıf doğruluğu, şablon promptlarla
sıfır-atış sınıflandırma, metin gömmelerinin çökmesi (ortalama kosinüs)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(r"F:\Drthesiscode")
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("clapft", ROOT / "scripts" / "08_clap_finetune.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

N_SINIF_BASI = int(sys.argv[1]) if len(sys.argv) > 1 else 8
torch.manual_seed(0); np.random.seed(0)

df = pd.read_parquet(M.PARQUET)
_, df_vl, _ = M.veri_bol(df, 0.1, 0.1, 42)          # v1 CLAP doğrulama bölümü
alt = (df_vl.groupby("sinif", group_keys=False)
       .apply(lambda g: g.sample(min(len(g), N_SINIF_BASI), random_state=0))
       .reset_index(drop=True))
print(f"Örnek: {len(alt)} dosya, {alt['sinif'].nunique()} sınıf")

ds = M.BBCSesMetinVeriSeti(alt, ROOT, tr_orani=0.0, augment=False)
wavs = torch.stack([ds[i]["wav"] for i in range(len(ds))])
y = torch.tensor(alt["sinif_idx"].values)

promptlar = json.load(open(ROOT / "data/processed/turkce_promptlar.json", encoding="utf-8"))
siniflar = sorted(alt["sinif"].unique(), key=lambda s: int(alt[alt.sinif == s].sinif_idx.iloc[0]))
sinif_idx_sirasi = torch.tensor([int(alt[alt.sinif == s].sinif_idx.iloc[0]) for s in siniflar])

cfg = M.FinetuneConfig()
model = M.CLAPFinetune(cfg).to(M.DEVICE).eval()
hazir_state = {k: v.clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def ham_ses(wav, fp16):
    import torchaudio
    w = wav.to(M.DEVICE)
    w = torchaudio.functional.resample(w, M.SR, model.audio_cfg["sample_rate"])
    with torch.autocast("cuda", enabled=fp16):
        out = model.model.audio_branch({"waveform": w}, mixup_lambda=None)
        e = model.model.audio_projection(out["embedding"])
    return e.float()


@torch.no_grad()
def degerlendir(etiket, fp16):
    embs = []
    for i in range(0, len(wavs), 16):
        embs.append(ham_ses(wavs[i:i + 16], fp16).cpu())
    E = torch.cat(embs)
    nan_orani = E.isnan().any(dim=1).float().mean().item()
    E = F.normalize(torch.nan_to_num(E), dim=-1)
    knn = M.sinif_knn_dogrulugu(E, y, k=5)
    sonuc = {"nan_orani": round(nan_orani, 3), "knn_acc": round(knn, 3)}
    for dil in ("ingilizce", "turkce"):
        with torch.autocast("cuda", enabled=fp16):
            T = torch.stack([model.metin_embedding(promptlar[s][dil]).float().mean(0) for s in siniflar])
        T = F.normalize(T.cpu(), dim=-1)
        tahmin = sinif_idx_sirasi[(E @ T.T).argmax(1)]
        sonuc[f"sifir_atis_{dil[:2]}"] = round((tahmin == y).float().mean().item(), 3)
        tum = F.normalize(torch.cat([model.metin_embedding(promptlar[s][dil]).float().cpu() for s in siniflar]), dim=-1)
        C = tum @ tum.T
        sonuc[f"metin_ort_kos_{dil[:2]}"] = round(C[~torch.eye(len(C), dtype=bool)].mean().item(), 3)
    print(f"{etiket:28s} fp16={fp16!s:5s} {sonuc}", flush=True)
    return sonuc


rapor = {}
rapor["hazir_fp32"] = degerlendir("hazır CLAP", False)
rapor["hazir_fp16"] = degerlendir("hazır CLAP", True)
for ad in ("clap_bbc_best.pt", "clap_bbc_dongu2_best.pt"):
    ck = torch.load(ROOT / "checkpoints" / ad, map_location=M.DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()
    rapor[f"{ad}_fp32"] = degerlendir(f"v1 {ad}", False)
    model.load_state_dict(hazir_state)
print(json.dumps(rapor, ensure_ascii=False))
