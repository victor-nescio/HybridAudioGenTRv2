# -*- coding: utf-8 -*-
"""Dışarıda kalan 5.158 dosya, veri setindeki dosyaların kopyası mı? Ayrıca veri setinin İÇİNDE
aynı kaydın farklı yollarda (ör. hem eğitim hem testte) bulunup bulunmadığı denetlenir.
Kimlik: dosya adı (BBC katalog numarası dahil) + bayt boyutu; şüphelilerde MD5."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
norm = lambda s: str(s).replace("/", "\\").lower()


def md5(p, n=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(n), b""):
            h.update(blok)
    return h.hexdigest()


v = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")
veri = {norm(d): d for d in v["dosya"]}
tum = sorted({str(p.relative_to(ROOT)) for p in (ROOT / "data" / "raw").rglob("*") if p.suffix.lower() == ".wav"})
dis = [p for p in tum if norm(p) not in veri]

# 1) Dışarıdakiler: aynı ad + aynı boyutta bir veri seti dosyası var mı?
ad_boyut = defaultdict(list)
for d in veri.values():
    p = ROOT / d
    ad_boyut[(p.name.lower(), p.stat().st_size)].append(d)
kopya, ozgun = [], []
for d in dis:
    p = ROOT / d
    (kopya if (p.name.lower(), p.stat().st_size) in ad_boyut else ozgun).append(d)
print(f"Dışarıda: {len(dis)}  — veri setindeki bir dosyayla aynı ad+boyut: {len(kopya)}  — özgün: {len(ozgun)}")
orn = kopya[:30]
esit = sum(md5(ROOT / d) == md5(ROOT / ad_boyut[((ROOT / d).name.lower(), (ROOT / d).stat().st_size)][0]) for d in orn)
print(f"  MD5 kontrolü (30 örnek kopya): {esit}/30 birebir aynı")

# 2) Veri setinin içinde kopya var mı? (aynı ad+boyut, farklı yol)
ic_kopya = {k: ds for k, ds in ad_boyut.items() if len(ds) > 1}
print(f"Veri seti içinde aynı ad+boyutlu grup: {len(ic_kopya)} (toplam {sum(len(x) for x in ic_kopya.values())} dosya)")

# 3) Bu kopyalar sınıflandırıcının farklı bölümlerine düşmüş mü? (sızıntı)
ds = v.groupby("dosya")["sinif_idx"].first().reset_index()
tv, te = train_test_split(ds["dosya"].values, test_size=0.15, stratify=ds["sinif_idx"].values, random_state=42)
s_tv = ds.set_index("dosya").loc[tv, "sinif_idx"].values
tr, vl = train_test_split(tv, test_size=0.15 / 0.85, stratify=s_tv, random_state=42)
bolum = {**{norm(x): "train" for x in tr}, **{norm(x): "val" for x in vl}, **{norm(x): "test" for x in te}}
capraz = []
for k, grup in ic_kopya.items():
    b = {bolum[norm(d)] for d in grup}
    if len(b) > 1:
        capraz.append({"ad": k[0], "bolumler": sorted(b), "n": len(grup),
                       "md5_ayni": len({md5(ROOT / d) for d in grup}) == 1})
n_md5 = sum(c["md5_ayni"] for c in capraz)
print(f"Farklı bölümlere düşmüş kopya grubu: {len(capraz)} (MD5 birebir aynı: {n_md5})")

ozet = {"disarida": len(dis), "disarida_kopya_ad_boyut": len(kopya), "disarida_ozgun": len(ozgun),
        "md5_ornek_esit_30": esit, "veri_ici_kopya_grubu": len(ic_kopya),
        "veri_ici_kopya_dosya": sum(len(x) for x in ic_kopya.values()),
        "bolumler_arasi_kopya_grubu": len(capraz), "bolumler_arasi_md5_ayni": n_md5,
        "bolumler_arasi_ornek": capraz[:20], "disarida_ozgun_ornek": [Path(d).name for d in ozgun[:20]]}
(ROOT / "results" / "denetim" / "kopya_denetimi.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi: results/denetim/kopya_denetimi.json")
