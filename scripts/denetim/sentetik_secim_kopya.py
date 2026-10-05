# -*- coding: utf-8 -*-
"""15_siniflandirma_v2.sentetik_sec'in sectigi kliplerde icerik kopyasi (ayni MD5) var mi?
Eski H1 klasorlerinde sinif basina 51 dosya / 50 benzersiz icerik vardi."""
import hashlib
import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("v2", ROOT / "scripts" / "15_siniflandirma_v2.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

H1 = ["data/synthetic_turcoh:kopek_sesleri,kus_sesleri,hava_olaylari,diger_hayvanlar",
      "data/synthetic:insaat_sesleri,dini_mekan,acil_servis", "data/synthetic_v2/H1ek:tarim_ciftlik"]
TUM = "kopek_sesleri,kus_sesleri,hava_olaylari,insaat_sesleri,dini_mekan,tarim_ciftlik,acil_servis,diger_hayvanlar"
for ad, k in [("H1", H1), ("C2p", [f"data/synthetic_v2/C2p:{TUM}"]), ("C4p", [f"data/synthetic_v2/C4p:{TUM}"])]:
    d = m.sentetik_sec(k, 50)
    sorun = 0
    for s, g in d.groupby("sinif"):
        h = [hashlib.md5((ROOT / f).read_bytes()).hexdigest() for f in g.dosya]
        say = Counter(h)
        kopyalar = [Path(f).name for f, c in zip(g.dosya, h) if say[c] > 1]
        if kopyalar:
            sorun += 1
            print(f"{ad} {s}: secilen 50'de {len(h) - len(set(h))} kopya -> {kopyalar}")
    print(f"{ad}: {len(d)} klip, kopyali sinif sayisi {sorun}")
