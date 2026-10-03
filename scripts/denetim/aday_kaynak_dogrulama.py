# -*- coding: utf-8 -*-
"""Revizyonda eklenecek / değiştirilecek aday kaynakların doğrulaması (Crossref DOI veya arXiv kimliği).
Her aday için resmi başlık ve ilk yazarlar çekilir; beklenen başlıkla benzerlik ölçülür.
Çıktı: results/denetim/aday_kaynaklar.json"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "HybridAudioGenTR-kaynak-denetimi/1.0 (mailto:fdcellat@gmail.com)"}

# (anahtar, kimlik, beklenen başlık, kullanım amacı)
ADAYLAR = [
    ("synthio", "arXiv:2410.02056", "Synthio: Augmenting Small-Scale Audio Classification Datasets with Synthetic Data", "R1/R3: T2A augmentasyonu (küçük veri)"),
    ("ronchini2024", "arXiv:2403.17864", "Synthetic training set generation using text-to-audio models for environmental sound classification", "R3: T2A eğitim setleri / ESC"),
    ("mindprompt", "arXiv:2504.03329", "Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification", "R3: prompt tasarımı"),
    ("gui2024fad", "arXiv:2311.01616", "Adapting Frechet Audio Distance for Generative Music Evaluation", "R3/R1.6: FAD sınırları; [6] yerine; fadtk"),
    ("tailleur2024", "arXiv:2403.17508", "Correlation of Fréchet Audio Distance With Human Perception of Environmental Audio Is Embedding Dependant", "R3/R1.6: FAD sınırları; [6] yerine"),
    ("fad_interspeech", "10.21437/Interspeech.2019-2219", "Fréchet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms", "[5]'in yayımlanmış sürümü"),
    ("wavcaps", "arXiv:2303.17395", "WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research", "R3: üretici eğitim verisi (BBC SFX içerir); [27] yerine BBC kullanımı"),
    ("replikasyon_t2a", "arXiv:2509.14934", "Mitigating Data Replication in Text-to-Audio Generative Diffusion", "Ezber analizi gerekçesi"),
    ("somepalli2023", "10.1109/CVPR52729.2023.00586", "Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models", "Ezber analizi yöntemi"),
    ("urbansound8k", "10.1145/2647868.2655045", "A Dataset and Taxonomy for Urban Sound Research", "R3: açık benchmark"),
    ("fsd50k", "10.1109/TASLP.2021.3133208", "FSD50K: An Open Dataset of Human-Labeled Sound Events", "R3: açık benchmark"),
    ("optuna", "10.1145/3292500.3330701", "Optuna: A Next-generation Hyperparameter Optimization Framework", "R3: Optuna"),
    ("holm1979", "search:A simple sequentially rejective multiple test procedure Holm", "A Simple Sequentially Rejective Multiple Test Procedure", "Holm–Bonferroni"),
    ("edwards1948", "10.1007/BF02289261", "Note on the correction for continuity in testing the significance of the difference between correlated proportions", "McNemar Yates/süreklilik düzeltmesi"),
    ("field2007", "10.1111/j.1467-9868.2007.00593.x", "Bootstrapping Clustered Data", "Küme bootstrap"),
    ("lme4", "10.18637/jss.v067.i01", "Fitting Linear Mixed-Effects Models Using lme4", "Karma etkili model (v2 birincil analiz)"),
    ("he2023synthetic", "arXiv:2210.07574", "Is synthetic data from generative models ready for image recognition?", "[12] yerine: görüntüde sentetik veri"),
    ("azizi2023", "arXiv:2304.08466", "Synthetic Data from Diffusion Models Improves ImageNet Classification", "[12] yerine: görüntüde sentetik veri"),
    ("envnet2017", "10.1109/ICASSP.2017.7952651", "Learning environmental sounds with end-to-end convolutional neural network", "[23] yerine: ham dalga vs log-mel"),
    ("dai2017raw", "10.1109/ICASSP.2017.7952190", "Very deep convolutional neural networks for raw waveforms", "[23] yerine: ham dalga vs log-mel"),
    ("selfinstruct", "10.18653/v1/2023.acl-long.754", "Self-Instruct: Aligning Language Models with Self-Generated Instructions", "[34] yerine: kendi ürettiği veriyle iyileşme"),
    ("beats", "arXiv:2212.09058", "BEATs: Audio Pre-Training with Acoustic Tokenizers", "R5: öz-denetimli güçlü sınıflandırıcı"),
    ("mixup", "arXiv:1710.09412", "mixup: Beyond Empirical Risk Minimization", "R4: kıyas (opsiyonel)"),
]


def getir(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode("utf-8", "replace")


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())


sonuc = []
for anahtar, kimlik, beklenen, amac in ADAYLAR:
    k = {"anahtar": anahtar, "kimlik": kimlik, "amac": amac}
    try:
        if kimlik.startswith("arXiv:"):
            x = ET.fromstring(getir("https://export.arxiv.org/api/query?id_list=" + kimlik[6:]))
            ns = {"a": "http://www.w3.org/2005/Atom", "ax": "http://arxiv.org/schemas/atom"}
            e = x.find("a:entry", ns)
            k["baslik"] = " ".join(e.find("a:title", ns).text.split())
            k["yazarlar"] = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
            k["yil"] = e.find("a:published", ns).text[:4]
            yorum, dergi = e.find("ax:comment", ns), e.find("ax:journal_ref", ns)
            k["not"] = " | ".join(t.text.strip() for t in (yorum, dergi) if t is not None)
            d = e.find("ax:doi", ns)
            k["doi"] = d.text if d is not None else None
        else:
            if kimlik.startswith("search:"):
                it = json.loads(getir("https://api.crossref.org/works?rows=1&query.bibliographic="
                                      + urllib.parse.quote(kimlik[7:])))["message"]["items"][0]
            else:
                it = json.loads(getir("https://api.crossref.org/works/" + urllib.parse.quote(kimlik)))["message"]
            k["baslik"] = (it.get("title") or [""])[0]
            k["yazarlar"] = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]
            k["yil"] = str((it.get("issued", {}).get("date-parts") or [[None]])[0][0])
            k["not"] = (it.get("container-title") or [""])[0]
            k["doi"] = it.get("DOI")
        k["benzerlik"] = round(difflib.SequenceMatcher(None, norm(beklenen), norm(k["baslik"])).ratio(), 2)
    except Exception as ex:
        k["hata"] = str(ex)
    durum = "OK" if k.get("benzerlik", 0) >= 0.85 else "!!"
    print(f"{durum} {anahtar:16s} {k.get('benzerlik')}  {k.get('yil')}  {k.get('baslik', k.get('hata', ''))[:70]}"
          f"  | {', '.join((k.get('yazarlar') or [])[:2])}  | {str(k.get('not'))[:60]}", flush=True)
    sonuc.append(k)
    time.sleep(1.0)

(ROOT / "results" / "denetim" / "aday_kaynaklar.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
print("kaydedildi: results/denetim/aday_kaynaklar.json")
