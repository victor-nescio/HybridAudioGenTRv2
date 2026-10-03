# -*- coding: utf-8 -*-
"""v1 FAD referans kümesi (06_measure_fad.sinif_dosya_sec, maks=300) kaç BENZERSİZ kayıt içeriyordu?
data/raw aynı kayıtların kopyalarını içerdiği için referans kümesinde tekrarlar olabilir."""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fad", ROOT / "scripts" / "06_measure_fad.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
KAT = re.compile(r"(\d{8})\.wav$", re.I)

sonuc = {}
for s in ["kopek_sesleri", "kus_sesleri", "hava_olaylari", "diger_hayvanlar",
          "insaat_sesleri", "dini_mekan", "acil_servis"]:
    sec = F.sinif_dosya_sec(ROOT / "data" / "raw" / s, maks=300)
    anahtar = [KAT.search(p.name).group(1) if KAT.search(p.name) else p.name for p in sec]
    sonuc[s] = {"secilen_dosya": len(sec), "benzersiz_kayit": len(set(anahtar)),
                "sinifin_tum_benzersiz_kaydi": len({(KAT.search(p.name).group(1) if KAT.search(p.name) else p.name)
                                                    for p in (ROOT / "data" / "raw" / s).rglob("*.wav")})}
    print(s, sonuc[s])
(ROOT / "results" / "denetim" / "fad_referans.json").write_text(json.dumps(sonuc, indent=1), encoding="utf-8")
