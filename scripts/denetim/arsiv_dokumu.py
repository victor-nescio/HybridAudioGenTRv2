# -*- coding: utf-8 -*-
"""R2.1: BBC arşivinden 26 sınıflı veri setine kadar tam döküm (katalog numarasına göre benzersiz kayıtlar).
  sounds/   : hiçbir anahtar-kelime kuralına uymadığı için data/raw'a taşınmamış kalan dosyalar
  data/raw  : kurallarla 26 sınıfa atanmış dosyalar (aynı kaydın kopyaları dahil)
  veri seti : veri_listesi.parquet (katalog numarasına göre tekil)"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.labels import _KURALLAR, SINIFLAR  # noqa: E402

KAT = re.compile(r"(\d{8})\.wav$", re.I)
anahtar = lambda p: (KAT.search(Path(p).name).group(1) if KAT.search(Path(p).name) else Path(p).name.lower())

kalan = [p for p in (ROOT / "sounds").rglob("*") if p.suffix.lower() == ".wav"]
raw = [p for p in (ROOT / "data" / "raw").rglob("*") if p.suffix.lower() == ".wav"]
veri = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")["dosya"].unique()

k_kalan, k_raw, k_veri = {anahtar(p) for p in kalan}, {anahtar(p) for p in raw}, {anahtar(p) for p in veri}
ozet = {
    "sounds_kalan_dosya": len(kalan), "sounds_kalan_benzersiz_kayit": len(k_kalan),
    "data_raw_dosya": len(raw), "data_raw_benzersiz_kayit": len(k_raw),
    "veri_seti_dosya": int(len(veri)), "veri_seti_benzersiz_kayit": len(k_veri),
    "kalan_ile_raw_ortak_kayit": len(k_kalan & k_raw),
    "toplam_benzersiz_kayit": len(k_kalan | k_raw),
    "kurala_uymayan_ve_disarida_kayit": len(k_kalan - k_raw),
    "kural_sayisi_sinif_basina": {s: len(k) for s, k in _KURALLAR},
    "sinif_sayisi": len(SINIFLAR),
    "kalan_ust_klasorler": sorted({p.relative_to(ROOT / "sounds").parts[0] for p in kalan}),
}
print(json.dumps({k: v for k, v in ozet.items() if k not in ("kural_sayisi_sinif_basina",)}, ensure_ascii=False, indent=1))
(ROOT / "results" / "denetim" / "arsiv_dokumu.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")

# Ek tablo: sınıf başına anahtar kelimeler (ilk eşleşen kural kazanır; sıra önemlidir)
satirlar = ["| Sıra | Sınıf | Anahtar kelimeler (ilk eşleşen kazanır) |", "|---|---|---|"]
for i, (s, k) in enumerate(_KURALLAR, 1):
    satirlar.append(f"| {i} | {s} | {', '.join(x.strip() for x in k)} |")
(ROOT / "outputs" / "makale" / "revizyon_2026-10" / "taksonomi_kurallari.md").write_text(
    "# 26 sınıflı taksonomi: anahtar-kelime kuralları (src/data/labels.py)\n\n" + "\n".join(satirlar) + "\n",
    encoding="utf-8")
print("kaydedildi: results/denetim/arsiv_dokumu.json, outputs/makale/revizyon_2026-10/taksonomi_kurallari.md")
