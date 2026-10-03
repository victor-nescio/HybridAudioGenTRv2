# -*- coding: utf-8 -*-
"""R2.1 / F10: BBC arşivinden hangi dosyalar 26 sınıflı veri setine girdi, hangileri dışarıda kaldı ve neden?
data/raw altındaki tüm WAV'lar ile veri_listesi.parquet karşılaştırılır; dışarıda kalanların süre dağılımı ölçülür."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
norm = lambda s: str(s).replace("/", "\\").lower()

v = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")
veri = set(v["dosya"].map(norm))
tum = sorted({str(p.relative_to(ROOT)) for p in (ROOT / "data" / "raw").rglob("*") if p.suffix.lower() == ".wav"})
dis = [p for p in tum if norm(p) not in veri]
print(f"data/raw WAV: {len(tum)}  veri setinde: {len(tum) - len(dis)}  dışarıda: {len(dis)}")

sure, okunamayan = [], 0
for p in dis:
    try:
        i = sf.info(str(ROOT / p))
        sure.append(i.frames / i.samplerate)
    except Exception:
        okunamayan += 1
s = np.array(sure)
sinif_dis = pd.Series([Path(p).parts[2] for p in dis]).value_counts().to_dict()
ozet = {
    "data_raw_wav": len(tum), "veri_setinde": len(tum) - len(dis), "disarida": len(dis),
    "disarida_okunamayan": okunamayan,
    "disarida_5sn_alti": int((s < 5).sum()), "disarida_5sn_ve_ustu": int((s >= 5).sum()),
    "disarida_sure_medyan_sn": round(float(np.median(s)), 1) if len(s) else None,
    "disarida_sinif_basina": sinif_dis,
    "disarida_ornek": [Path(p).name for p in dis[:15]],
}
print(json.dumps({k: ozet[k] for k in ozet if k not in ("disarida_sinif_basina", "disarida_ornek")}, ensure_ascii=False))
print("örnek:", ozet["disarida_ornek"][:6])
cikti = ROOT / "results" / "denetim" / "veri_kapsami.json"
cikti.write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi:", cikti)
