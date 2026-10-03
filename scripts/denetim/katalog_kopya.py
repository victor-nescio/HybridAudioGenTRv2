# -*- coding: utf-8 -*-
"""Aynı BBC kaydının (aynı katalog numarası / dosya adı) farklı yollarda farklı sürümleri var mı?
(a) Veri setindeki dosyalar arasında aynı ad tekrarı ve bunların sınıflandırıcı bölümlerine dağılımı;
(b) dışarıda kalan dosyaların ne kadarının, veri setindeki bir dosyayla aynı katalog numarasını taşıdığı."""
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
norm = lambda s: str(s).replace("/", "\\").lower()
KATALOG = re.compile(r"(\d{8})\.wav$", re.I)


def anahtar(yol):
    m = KATALOG.search(Path(yol).name)
    return m.group(1) if m else Path(yol).name.lower()


v = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")
dosyalar = v.groupby("dosya")["sinif"].first()
tum = sorted({str(p.relative_to(ROOT)) for p in (ROOT / "data" / "raw").rglob("*") if p.suffix.lower() == ".wav"})
veri_set = {norm(d) for d in dosyalar.index}
dis = [p for p in tum if norm(p) not in veri_set]

ds = v.groupby("dosya")["sinif_idx"].first().reset_index()
tv, te = train_test_split(ds["dosya"].values, test_size=0.15, stratify=ds["sinif_idx"].values, random_state=42)
s_tv = ds.set_index("dosya").loc[tv, "sinif_idx"].values
tr, vl = train_test_split(tv, test_size=0.15 / 0.85, stratify=s_tv, random_state=42)
bolum = {**{x: "train" for x in tr}, **{x: "val" for x in vl}, **{x: "test" for x in te}}

gruplar = defaultdict(list)
for d in dosyalar.index:
    gruplar[anahtar(d)].append(d)
tekrar = {k: g for k, g in gruplar.items() if len(g) > 1}
capraz = {k: g for k, g in tekrar.items() if len({bolum[d] for d in g}) > 1}
capraz_test = {k: g for k, g in capraz.items() if any(bolum[d] == "test" for d in g)}
sinif_farkli = {k: g for k, g in tekrar.items() if len({dosyalar[d] for d in g}) > 1}
katalogsuz = sum(1 for d in dosyalar.index if not KATALOG.search(Path(d).name))

dis_ayni_katalog = [d for d in dis if anahtar(d) in gruplar]
print(f"Veri seti: {len(dosyalar)} dosya, {len(gruplar)} benzersiz katalog anahtarı ({katalogsuz} dosyada katalog no yok)")
print(f"  Aynı anahtarlı (tekrarlı) grup: {len(tekrar)} grup / {sum(len(g) for g in tekrar.values())} dosya")
print(f"  Farklı sınıflandırıcı bölümlerine dağılmış grup: {len(capraz)}  (en az biri testte: {len(capraz_test)})")
print(f"  Grup içinde farklı sınıf etiketi: {len(sinif_farkli)}")
print(f"Dışarıda: {len(dis)} — veri setindeki bir dosyayla aynı katalog anahtarı: {len(dis_ayni_katalog)}")
test_dosya_etkilenen = sum(1 for g in capraz_test.values() for d in g if bolum[d] == "test")
print(f"  Etkilenen test dosyası (aynı kaydın başka sürümü train/val'de): {test_dosya_etkilenen} / {len(te)}")

ozet = {"veri_dosya": len(dosyalar), "benzersiz_anahtar": len(gruplar), "katalogsuz_dosya": katalogsuz,
        "tekrarli_grup": len(tekrar), "tekrarli_dosya": sum(len(g) for g in tekrar.values()),
        "bolumler_arasi_grup": len(capraz), "bolumler_arasi_test_iceren_grup": len(capraz_test),
        "etkilenen_test_dosyasi": test_dosya_etkilenen, "test_dosyasi": len(te),
        "grup_ici_farkli_sinif": len(sinif_farkli),
        "disarida": len(dis), "disarida_ayni_katalog": len(dis_ayni_katalog),
        "ornek_bolumler_arasi": {k: [(Path(d).parent.name, bolum[d]) for d in g] for k, g in list(capraz.items())[:15]}}
(ROOT / "results" / "denetim" / "katalog_kopya.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi: results/denetim/katalog_kopya.json")
