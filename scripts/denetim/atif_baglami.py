# -*- coding: utf-8 -*-
"""Makale metnindeki her [n] atfının geçtiği cümleleri çıkarır (kaynakça denetimi için).
Girdi: v1 metni (docx'ten çıkarılmış düz metin). Çıktı: atif_baglami.md"""
import re
import sys
from collections import defaultdict
from pathlib import Path

girdi = Path(sys.argv[1])
cikti = Path(sys.argv[2])
satirlar = girdi.read_text(encoding="utf-8").splitlines()
govde = "\n".join(satirlar[: next(i for i, s in enumerate(satirlar) if s.strip() == "REFERENCES")])
kaynaklar = satirlar[next(i for i, s in enumerate(satirlar) if s.strip() == "REFERENCES") + 1:]
kaynaklar = [k for k in kaynaklar if k.strip()][:52]


def numaralar(grup):
    out = []
    for parca in grup.split(","):
        parca = parca.strip().strip("[]")
        m = re.match(r"^(\d+)\s*[–-]\s*\[?(\d+)$", parca)
        if m:
            out += list(range(int(m.group(1)), int(m.group(2)) + 1))
        elif parca.isdigit():
            out.append(int(parca))
    return out


baglam = defaultdict(list)
cumleler = re.split(r"(?<=[.!?])\s+(?=[A-Z])", govde.replace("\n", " "))
for c in cumleler:
    for grup in re.findall(r"\[(\d+(?:\s*[–-]\s*\[?\d+\]?)?(?:\],?\s*\[\d+(?:\s*[–-]\s*\[?\d+\]?)?)*)\]", c):
        for n in numaralar(grup.replace("], [", ",").replace("],[", ",")):
            if c.strip() not in baglam[n]:
                baglam[n].append(c.strip())

with open(cikti, "w", encoding="utf-8") as f:
    for i, k in enumerate(kaynaklar, 1):
        f.write(f"## [{i}] {k}\n\n")
        if not baglam[i]:
            f.write("- (METİNDE ATIF BULUNAMADI)\n")
        for c in baglam[i]:
            f.write(f"- {c}\n")
        f.write("\n")
print("atıfsız kaynaklar:", [i for i in range(1, len(kaynaklar) + 1) if not baglam[i]])
print("kaydedildi:", cikti)
