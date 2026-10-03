# -*- coding: utf-8 -*-
"""Kaynakça doğrulaması: her kaynağın DOI'si (Crossref) veya arXiv numarası (arXiv API) sorgulanır,
dönen başlık makaledeki başlıkla karşılaştırılır. DOI/arXiv yoksa Crossref bibliyografik arama yapılır.
Girdi: v1 metni (REFERENCES sonrası 52 satır). Çıktı: JSON + ekrana özet."""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

girdi, cikti = Path(sys.argv[1]), Path(sys.argv[2])
satirlar = girdi.read_text(encoding="utf-8").splitlines()
i0 = next(i for i, s in enumerate(satirlar) if s.strip() == "REFERENCES")
kaynaklar = [k for k in satirlar[i0 + 1:] if k.strip()][:52]
UA = {"User-Agent": "HybridAudioGenTR-kaynak-denetimi/1.0 (mailto:fdcellat@gmail.com)"}


def getir(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower().replace("—", " ").replace("–", " ")).strip()


def benzerlik(a, b):
    return round(difflib.SequenceMatcher(None, norm(a), norm(b)).ratio(), 2)


sonuc = []
for n, k in enumerate(kaynaklar, 1):
    m = re.search(r"[“\"](.+?)[,]?[”\"]", k)
    baslik = m.group(1).rstrip(",") if m else k[:120]
    doi = re.search(r"doi:\s*(\S+?)\.?$", k)
    arx = re.search(r"arXiv[:.]\s*(\d{4}\.\d{4,5})", k)
    kayit = {"no": n, "baslik": baslik, "yontem": None, "bulunan": None, "benzerlik": None, "ek": None}
    try:
        if doi and "arXiv" not in doi.group(1):
            d = json.loads(getir("https://api.crossref.org/works/" + urllib.parse.quote(doi.group(1))))["message"]
            kayit.update(yontem="DOI", bulunan=(d.get("title") or [""])[0],
                         ek=", ".join(a.get("family", "") for a in d.get("author", [])[:4]))
        elif arx or (doi and "arXiv" in doi.group(1)):
            aid = arx.group(1) if arx else doi.group(1).split("arXiv.")[-1]
            x = ET.fromstring(getir("https://export.arxiv.org/api/query?id_list=" + aid))
            ns = {"a": "http://www.w3.org/2005/Atom"}
            e = x.find("a:entry", ns)
            kayit.update(yontem=f"arXiv {aid}", bulunan=" ".join(e.find("a:title", ns).text.split()),
                         ek=", ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)[:4]))
        else:
            q = urllib.parse.quote(baslik)
            it = json.loads(getir(f"https://api.crossref.org/works?query.bibliographic={q}&rows=1"))["message"]["items"]
            if it:
                kayit.update(yontem="arama", bulunan=(it[0].get("title") or [""])[0],
                             ek=", ".join(a.get("family", "") for a in it[0].get("author", [])[:4]) + " | " + it[0].get("DOI", ""))
        if kayit["bulunan"]:
            kayit["benzerlik"] = benzerlik(baslik, kayit["bulunan"])
    except Exception as e:
        kayit["ek"] = f"HATA: {e}"
    durum = "OK" if (kayit["benzerlik"] or 0) >= 0.85 else ("?" if (kayit["benzerlik"] or 0) >= 0.6 else "!!")
    kayit["durum"] = durum
    sonuc.append(kayit)
    print(f"[{n:2d}] {durum:2s} {kayit['benzerlik']}  {kayit['yontem']}  | {baslik[:60]}  ->  {str(kayit['bulunan'])[:60]}", flush=True)
    time.sleep(1.0)

cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi:", cikti)
