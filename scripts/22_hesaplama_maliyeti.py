"""
Hesaplama maliyeti tablosu (R5.9). Tum olcumler kayitli ciktilardan; tek GPU (NVIDIA RTX 3060, 12 GB).

  CLAP ince-ayari : results/v2/clap_finetune_*.json (epoch basina sure_dk, toplam egitim_saat)
  Uretim          : klip basina sure = ayni klasordeki ardisik WAV yazilma zamanlari arasindaki farkin
                    medyani (AudioLDM2-full, 200 DDIM adimi, 10 sn klip); duraklatma bosluklari medyani etkilemez
  Siniflandirici  : results/v2/siniflandirma/<konfig>_s<seed>.json "sure_dk" (egitim + epoch ici degerlendirme)
  BEATs           : gomme suresi (log) + kafa egitimi (sure_dk)

Kullanim: python scripts/22_hesaplama_maliyeti.py   ->  results/v2/hesaplama_maliyeti.json
"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    s = {"gpu": "NVIDIA GeForce RTX 3060 (12 GB), tek GPU"}
    clap = {}
    for f in sorted((ROOT / "results" / "v2").glob("clap_finetune_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ep = [x["sure_dk"] for x in d["tarih_detay"] if "sure_dk" in x]
        clap[d["dongu"]] = {"epoch": len(ep), "epoch_basina_dk_medyan": round(float(np.median(ep)), 1),
                            "toplam_saat": d["egitim_saat"]}
    s["clap_ince_ayar"] = clap

    uretim = {}
    for k in ("C2p", "C3p", "C4p", "H1ek"):
        zaman = sorted(w.stat().st_mtime for w in (ROOT / "data" / "synthetic_v2" / k).glob("*/*.wav"))
        fark = np.diff(zaman)
        fark = fark[(fark > 5) & (fark < 600)]                 # ayni anda yazilan / duraklatma bosluklari disarida
        uretim[k] = {"n_klip": len(zaman), "klip_basina_sn_medyan": round(float(np.median(fark)), 1)}
    tum = np.median([v["klip_basina_sn_medyan"] for v in uretim.values()])
    s["uretim"] = {"konfig": uretim, "klip_basina_sn_medyan": round(float(tum), 1),
                   "toplam_klip": int(sum(v["n_klip"] for v in uretim.values()))}

    sinif = {}
    for f in sorted((ROOT / "results" / "v2" / "siniflandirma").glob("*_s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        sinif.setdefault(d["konfig"], []).append(d["sure_dk"])
    resnet = {k: v for k, v in sinif.items() if not k.startswith("beats_")}
    hepsi = [x for v in resnet.values() for x in v]
    s["resnet18"] = {"n_kosu": len(hepsi), "kosu_basina_dk_medyan": round(float(np.median(hepsi)), 1),
                     "toplam_saat": round(sum(hepsi) / 60, 1),
                     "konfig_medyan_dk": {k: round(float(np.median(v)), 1) for k, v in resnet.items()}}
    log = (ROOT / "outputs" / "v2_loglar" / "kuyruk" / "B0_beats_gomme.log").read_text(encoding="utf-8", errors="ignore")
    m = re.findall(r"Gommeler tamam: .*\(([\d.]+) dk\)", log)
    beats = [x for k, v in sinif.items() if k.startswith("beats_") for x in v]
    s["beats"] = {"gomme_dk": float(m[-1]) if m else None, "n_kafa": len(beats),
                  "kafa_basina_dk_medyan": round(float(np.median(beats)), 1) if beats else None}

    toplam = (sum(v["toplam_saat"] for v in clap.values())
              + s["uretim"]["toplam_klip"] * s["uretim"]["klip_basina_sn_medyan"] / 3600
              + s["resnet18"]["toplam_saat"] + ((s["beats"]["gomme_dk"] or 0) + sum(beats)) / 60)
    s["toplam_gpu_saat_yaklasik"] = round(toplam, 1)
    s["not"] = ("Uretim toplami v2'de uretilen klipler icindir (H1 klipleri ilk calismadan yeniden kullanildi); "
                "kalite metrikleri ve analizler dahil degildir.")
    cikti = ROOT / "results" / "v2" / "hesaplama_maliyeti.json"
    cikti.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
