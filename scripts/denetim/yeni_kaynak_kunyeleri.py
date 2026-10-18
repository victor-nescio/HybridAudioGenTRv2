# -*- coding: utf-8 -*-
"""Revizyonda eklenen / degisen kaynaklarin kunyeleri Crossref'ten (DOI'li) cekilir; IEEE bicimine yakin
taslak uretilir, elle kontrol icin baslik benzerligi raporlanir. arXiv'li kaynaklar arXiv API'den.

Kullanim: python scripts/denetim/yeni_kaynak_kunyeleri.py -> results/denetim/yeni_kaynaklar.json
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (anahtar, sorgu basligi, ilk yazar soyadi, kaynak turu)
LISTE = [
    ("[3] AudioGen", "AudioGen: Textually Guided Audio Generation", "Kreuk", "arxiv:2209.15352"),
    ("[6] Gui2024", "Adapting Frechet Audio Distance for Generative Music Evaluation", "Gui", "crossref"),
    ("[12] He2023", "Is synthetic data from generative models ready for image recognition?", "He", "arxiv:2210.07574"),
    ("[23] Tokozume2017", "Learning environmental sounds with end-to-end convolutional neural network", "Tokozume", "crossref"),
    ("[27] WavCaps", "WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research", "Mei", "crossref"),
    ("[34] SelfInstruct", "Self-Instruct: Aligning Language Models with Self-Generated Instructions", "Wang", "crossref"),
    ("Tailleur2024", "Correlation of Frechet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependent", "Tailleur", "crossref"),
    ("Azizi2023", "Synthetic Data from Diffusion Models Improves ImageNet Classification", "Azizi", "arxiv:2304.08466"),
    ("Dai2017", "Very deep convolutional neural networks for raw waveforms", "Dai", "crossref"),
    ("Ghosh2025", "Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data", "Ghosh", "arxiv:2410.02056"),
    ("Ronchini2024", "Synthetic training set generation using text-to-audio models for environmental sound classification", "Ronchini", "arxiv:2403.17864"),
    ("MindPrompt2025", "Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification", "Ronchini", "arxiv:2504.03329"),
    ("Messina2026", "Mitigating data replication in text-to-audio generative diffusion models through anti-memorization guidance", "Messina", "arxiv:2509.14934"),
    ("Somepalli2023", "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models", "Somepalli", "crossref"),
    ("Chen2023BEATs", "BEATs: Audio Pre-Training with Acoustic Tokenizers", "Chen", "arxiv:2212.09058"),
    ("US8K", "A Dataset and Taxonomy for Urban Sound Research", "Salamon", "crossref"),
    ("FSD50K", "FSD50K: An Open Dataset of Human-Labeled Sound Events", "Fonseca", "crossref"),
    ("Akiba2019", "Optuna: A Next-generation Hyperparameter Optimization Framework", "Akiba", "crossref"),
    ("Holm1979", "A simple sequentially rejective multiple test procedure", "Holm", "crossref"),
    ("Edwards1948", "Note on the correction for continuity in testing the significance of the difference between correlated proportions", "Edwards", "crossref"),
    ("Field2007", "Bootstrapping clustered data", "Field", "crossref"),
    ("Bates2015", "Fitting Linear Mixed-Effects Models Using lme4", "Bates", "crossref"),
]


def benzer(a, b):
    n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
    return round(difflib.SequenceMatcher(None, n(a), n(b)).ratio(), 2)


def getjson(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kaynak-denetimi/1.0 (mailto:none)"}),
                                timeout=40) as r:
        return r.read()


def crossref(baslik, soyad):
    q = urllib.parse.urlencode({"query.bibliographic": baslik, "query.author": soyad, "rows": 5})
    items = json.loads(getjson("https://api.crossref.org/works?" + q))["message"]["items"]
    en = max(items, key=lambda m: benzer(baslik, (m.get("title") or [""])[0]))
    yazar = [f"{' '.join(p[0] + '.' for p in re.split(r'[ -]', a.get('given', '')) if p)} {a.get('family', '')}".strip()
             for a in en.get("author", [])]
    return {"baslik": (en.get("title") or [""])[0], "yazarlar": yazar, "yer": (en.get("container-title") or [""])[0],
            "yil": (en.get("issued", {}).get("date-parts") or [[None]])[0][0], "cilt": en.get("volume"),
            "sayi": en.get("issue"), "sayfa": en.get("page"), "doi": en.get("DOI"), "tur": en.get("type"),
            "benzerlik": benzer(baslik, (en.get("title") or [""])[0])}


def arxiv(kimlik):
    x = getjson(f"https://export.arxiv.org/api/query?id_list={kimlik}").decode("utf-8")
    baslik = re.sub(r"\s+", " ", re.findall(r"<entry>.*?<title>(.*?)</title>", x, re.S)[0]).strip()
    yazar = re.findall(r"<author>\s*<name>(.*?)</name>", x)
    yorum = re.findall(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", x, re.S)
    yil = re.findall(r"<published>(\d{4})", x)[0]
    return {"baslik": baslik, "yazarlar": yazar, "arxiv": kimlik, "yil_arxiv": int(yil),
            "yorum": re.sub(r"\s+", " ", yorum[0]).strip() if yorum else None}


def main():
    out = {}
    for anahtar, baslik, soyad, tur in LISTE:
        try:
            out[anahtar] = arxiv(tur.split(":")[1]) if tur.startswith("arxiv") else crossref(baslik, soyad)
            out[anahtar]["sorgu"] = baslik
        except Exception as e:
            out[anahtar] = {"hata": str(e)}
        v = out[anahtar]
        print(f"{anahtar:16s} {v.get('benzerlik', '-')!s:5s} {v.get('yil') or v.get('yil_arxiv')} | "
              f"{(v.get('yer') or v.get('yorum') or '')[:60]} | {(v.get('baslik') or v.get('hata'))[:70]}")
        time.sleep(0.5)
    (ROOT / "results" / "denetim" / "yeni_kaynaklar.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
