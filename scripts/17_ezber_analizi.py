"""
Ezber / veri replikasyonu analizi (revizyon K2; R1.1, R5.1, R3 "uretici egitim verisi").

Hazir AudioLDM2 ve CLAP'i, BBC Sound Effects'i iceren veriyle (WavCaps, LAION-Audio-630K)
egitilmis; yani arsivimizin TAMAMINI (egitim ve test kayitlari) gormus olabilir. Bu nedenle
"sentetik klip test kayitlarina mi egitim kayitlarina mi yakin" karsilastirmasi ezberi
yakalayamaz. Bunun yerine (Somepalli ve ark., CVPR 2023 yaklasimi):

  sentetik -> gercek : her sentetik klibin, sinifinin gercek segmentleri arasindaki en yakin
                       komsu kosinus benzerligi
  gercek  -> gercek  : her gercek segmentin, AYNI KAYIT DISINDAKI gercek segmentler arasindaki
                       en yakin komsu benzerligi (dogal "farkli ama benzer kayit" duzeyi)

Sentetik klipler gercek kayitlara, gercek kayitlarin birbirine oldugundan belirgin bicimde daha
yakin degilse ve yakin-kopya esigini (gercek->gercek dagiliminin 99. yuzdeligi) asan oran
benzerse, kopyalama izi yoktur. Gomme: VGGish (fadtk), segment basina zaman ortalamasi.

Kullanim:
  python scripts/17_ezber_analizi.py --kaynak data/synthetic_turcoh:kopek_sesleri,kus_sesleri,hava_olaylari,diger_hayvanlar --etiket H1
  python scripts/17_ezber_analizi.py ... --maks_kayit 20      # hizli deneme
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SR, SEG = 16000, 5 * 16000


def vggish_yukle():
    from fadtk.model_loader import VGGishModel
    m = VGGishModel()
    m.load_model()
    return m


def gomme(model, wav: np.ndarray) -> np.ndarray:
    e = model.get_embedding(wav.astype(np.float32))      # (n_kare, 128)
    v = np.asarray(e).mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def ses_oku(yol: Path) -> np.ndarray:
    import resampy
    x, sr = sf.read(str(yol), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resampy.resample(x, sr, SR)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kaynak", action="append", required=True, help="'klasor:sinif1,sinif2'")
    p.add_argument("--etiket", required=True)
    p.add_argument("--maks_kayit", type=int, default=None, help="Sinif basina gercek kayit siniri (deneme)")
    p.add_argument("--maks_segment", type=int, default=6, help="Kayit basina en fazla segment (esit aralikli)")
    a = p.parse_args()

    v = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")
    model = vggish_yukle()
    sonuc = {"etiket": a.etiket, "gomme": "VGGish (fadtk), zaman ortalamasi", "siniflar": {}}

    for k in a.kaynak:
        klasor, siniflar = k.split(":")
        for sinif in siniflar.split(","):
            kayitlar = sorted(v[v.sinif == sinif]["dosya"].unique())
            if a.maks_kayit:
                kayitlar = kayitlar[: a.maks_kayit]
            G, kimlik = [], []
            for i, d in enumerate(kayitlar):
                x = ses_oku(ROOT / d)
                n = max(1, len(x) // SEG)
                for j in np.unique(np.linspace(0, n - 1, min(n, a.maks_segment)).astype(int)):
                    seg = x[j * SEG:(j + 1) * SEG]
                    if len(seg) < SR:
                        continue
                    G.append(gomme(model, seg)); kimlik.append(i)
            G, kimlik = np.stack(G), np.array(kimlik)
            S = np.stack([gomme(model, ses_oku(w)[:SEG]) for w in sorted((ROOT / klasor / sinif).glob("*.wav"))])

            ss = (S @ G.T).max(1)                                   # sentetik -> gercek
            GG = G @ G.T
            GG[kimlik[:, None] == kimlik[None, :]] = -1             # ayni kayit haric
            gg = GG.max(1)                                          # gercek -> gercek
            esik = float(np.percentile(gg, 99))
            sonuc["siniflar"][sinif] = {
                "n_sentetik": int(len(S)), "n_gercek_kayit": int(len(kayitlar)), "n_gercek_segment": int(len(G)),
                "sentetik_gercek_nn_medyan": round(float(np.median(ss)), 4),
                "gercek_gercek_nn_medyan": round(float(np.median(gg)), 4),
                "sentetik_gercek_nn_p95": round(float(np.percentile(ss, 95)), 4),
                "gercek_gercek_nn_p95": round(float(np.percentile(gg, 95)), 4),
                "yakin_kopya_esigi_p99": round(esik, 4),
                "esigi_asan_sentetik_orani": round(float((ss > esik).mean()), 4),
            }
            print(sinif, sonuc["siniflar"][sinif], flush=True)

    cikti = ROOT / "results" / "v2" / f"ezber_analizi_{a.etiket}.json"
    cikti.parent.mkdir(parents=True, exist_ok=True)
    cikti.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("kaydedildi:", cikti)


if __name__ == "__main__":
    main()
