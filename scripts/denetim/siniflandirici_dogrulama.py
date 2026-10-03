# -*- coding: utf-8 -*-
"""Gece testlerinin karşılaştırmaları.
  devam   : kesintisiz koşu (_testA) ile kesinti+devam koşusu (_testB) birebir aynı mı?
  uzlasma : v2 kodu + seed 42 baseline, v1 baseline'ını (results/v1/cnn_20260427_0312_sonuclar.json)
            birebir üretiyor mu? (test macro-F1 ve epoch geçmişi)
Kullanım: python siniflandirici_dogrulama.py devam|uzlasma"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / "results" / "v2" / "siniflandirma"
M = ROOT / "models" / "v2"
mod = sys.argv[1]

if mod == "devam":
    a = json.loads((S / "_testA_s7.json").read_text(encoding="utf-8"))
    b = json.loads((S / "_testB_s7.json").read_text(encoding="utf-8"))
    wa = torch.load(M / "cnn__testA_s7.pt", map_location="cpu")["model_state"]
    wb = torch.load(M / "cnn__testB_s7.pt", map_location="cpu")["model_state"]
    fark = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa if wa[k].is_floating_point())
    ayni = a["tarih"] == b["tarih"] and a["test_f1"] == b["test_f1"] and fark == 0.0
    print(f"DEVAM TESTI: {'GECTI' if ayni else 'KALDI'}  test_f1 A/B={a['test_f1']}/{b['test_f1']}  "
          f"gecmis ayni={a['tarih'] == b['tarih']}  agirlik maks fark={fark}")
    print("A gecmis:", a["tarih"]); print("B gecmis:", b["tarih"])
elif mod == "uzlasma":
    v1 = json.loads((ROOT / "results" / "v1" / "cnn_20260427_0312_sonuclar.json").read_text(encoding="utf-8"))
    v2 = json.loads((S / "baseline_s42.json").read_text(encoding="utf-8"))
    g1 = [(e["epoch"], round(e["val_f1"], 4)) for e in v1["epoch_tarih"]]
    g2 = [(e["epoch"], round(e["val_f1"], 4)) for e in v2["tarih"]]
    ayni = abs(v1["test_macro_f1"] - v2["test_f1"]) < 1e-4 and g1 == g2
    print(f"UZLASMA: {'GECTI' if ayni else 'FARK VAR'}  v1 test={v1['test_macro_f1']}  v2 test={v2['test_f1']}  "
          f"epoch gecmisi ayni={g1 == g2}")
    if g1 != g2:
        for x, y in zip(g1, g2):
            if x != y:
                print("  ilk fark:", x, y)
                break
