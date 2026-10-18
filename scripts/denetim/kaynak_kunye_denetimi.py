# -*- coding: utf-8 -*-
"""Kaynakca kunye denetimi (18 Ekim): 3 Ekim denetimi basligi eslestirdi ama yayin yeri / yili
karsilastirmadi; [4] (ICASSP yazilmis, aslinda Interspeech) ve [5] bu yuzden kacti. Bu script her
DOI'li kaynak icin Crossref kaydindaki yili ve yayin yerini atif metniyle karsilastirir, arXiv'li
kaynaklarda arXiv kimliginin yilini kontrol eder; uyusmayanlari elle incelemek icin listeler.

Kullanim: python scripts/denetim/kaynak_kunye_denetimi.py  ->  results/denetim/kaynak_kunye.json
"""
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENUE_ANAHTAR = {  # Crossref kapsayici adi -> atifta beklenen anahtar kelimelerden biri
    "interspeech": ["interspeech"], "icassp": ["icassp", "acoust., speech"], "transactions on audio": ["audio, speech",
    "taslp", "trans. audio"], "acm": ["acm"], "neural information": ["neurips", "neural inf"],
    "cvpr": ["cvpr", "comput. vis. pattern"], "iclr": ["iclr", "learn. represent"], "eusipco": ["eusipco"],
    "dcase": ["dcase"], "mmsp": ["mmsp", "multimedia signal"], "kdd": ["kdd", "knowl. discovery"],
}


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kaynak-denetimi/1.0"}), timeout=30) as r:
        m = json.loads(r.read())["message"]
    yil = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"baslik": (m.get("title") or [""])[0], "yer": (m.get("container-title") or [""])[0],
            "yil": yil, "sayfa": m.get("page")}


def main():
    import urllib.parse  # noqa: F401  (crossref icinde)
    denetim = json.loads((ROOT / "results" / "denetim" / "kaynak_dogrulama.json").read_text(encoding="utf-8"))
    metin = (ROOT / "outputs" / "makale" / "revizyon_2026-10" / "v1_metin.txt").read_text(encoding="utf-8")
    ref = [s for s in metin[metin.index("REFERENCES"):].splitlines() if s.strip()][1:]
    sonuc = []
    for x in denetim:
        atif = ref[x["no"] - 1] if x["no"] - 1 < len(ref) else ""
        kayit = {"no": x["no"], "atif": atif, "sorunlar": []}
        doi = re.search(r"10\.\d{4,9}/[^\s|,]+", x.get("ek", "") or "")
        yil_atif = re.findall(r"\b(19|20)(\d{2})\b", atif)
        yil_atif = {int(a + b) for a, b in yil_atif}
        if doi:
            try:
                c = crossref(doi.group(0).rstrip("."))
                kayit["crossref"] = c
                if c["yil"] and yil_atif and c["yil"] not in yil_atif:
                    kayit["sorunlar"].append(f"yil: atif {sorted(yil_atif)} / Crossref {c['yil']}")
                yer = (c["yer"] or "").lower()
                for k, beklenen in VENUE_ANAHTAR.items():
                    if k in yer and not any(b in atif.lower() for b in beklenen):
                        kayit["sorunlar"].append(f"yayin yeri: Crossref '{c['yer']}' atifta yok")
                if x.get("benzerlik", 1) < 0.98:
                    kayit["sorunlar"].append(f"baslik benzerligi {x.get('benzerlik')}: '{c['baslik']}'")
            except Exception as e:
                kayit["sorunlar"].append(f"Crossref hatasi: {e}")
            time.sleep(0.3)
        else:
            ax = re.search(r"arXiv:(\d{2})(\d{2})\.", atif)
            if ax and yil_atif and 2000 + int(ax.group(1)) not in yil_atif:
                kayit["sorunlar"].append(f"arXiv yili 20{ax.group(1)} atifta yok {sorted(yil_atif)}")
        sonuc.append(kayit)
        if kayit["sorunlar"]:
            print(f"[{x['no']}] " + " | ".join(kayit["sorunlar"]))
            print(f"     ATIF: {atif[:220]}")
    cikti = ROOT / "results" / "denetim" / "kaynak_kunye.json"
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{sum(1 for k in sonuc if k['sorunlar'])} / {len(sonuc)} kaynakta elle bakilacak not; kaydedildi: {cikti}")


if __name__ == "__main__":
    main()
