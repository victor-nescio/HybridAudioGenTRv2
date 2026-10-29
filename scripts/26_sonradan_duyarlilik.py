"""
Sonradan eklenen (ON KAYITTA YOK) duyarlilik analizleri — revizyonun bagimsiz incelemesi uzerine (30 Ekim 2026).

  1) Birincil etki (C4' - baseline, Grup A, son epoch, 5 eslesmis seed) bolum bazinda: yalniz test, yalniz val,
     val+test (on kayitli analizin tekrari). Gerekce: val bolumu baseline'da model ailesi secimine katildi; ayrica her
     iki bolum ilk calismada kullanildi.
  2) Hic sentetik almayan 18 sinifin grup macro-F1 farki (her konfigurasyon - baseline; son epoch, val+test): 26 sinif
     macro-F1'in degismemesi tek tek siniflardaki kayiplari gizleyebilir (R3, Concern 1).

On kayitli analizle ayni uygulama: 16_birincil_analiz.py (segment tahminlerinden sinif F1; kayit kume bootstrap'i,
sinif ici katmanli, B = 10.000, uretec seed 2026) ve 20_ikincil_aile.py (ozet). Birincil analizin yerine gecmez.
Kullanim: python scripts/26_sonradan_duyarlilik.py  ->  results/v2/sonradan_duyarlilik.json
"""
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
s = importlib.util.spec_from_file_location("ia", ROOT / "scripts" / "20_ikincil_aile.py")
IA = importlib.util.module_from_spec(s)
s.loader.exec_module(IA)
BA = IA.BA

ARTIRILAN = set().union(*[BA.GRUPLAR[g] for g in ("A_belirgin", "B_dusukF1", "B_faz1_acil", "kontrol")])
BA.GRUPLAR["artirilmayan"] = [c for c in BA.SINIF_SIRASI if c not in ARTIRILAN]
KONFIG = ["H1", "C2p", "C3p", "C4p", "C4p_kaydir", "specaug", "specaugC4p"]


def yukle(konfigler, bolum):
    """bolum: 'test' | 'val' | 'valtest' (son epoch tahminleri)."""
    if bolum == "valtest":
        return IA.hazirla(konfigler, "son_valtest")
    tah, gt, dosya = {}, None, None
    for k in konfigler:
        tah[k] = {}
        for f in sorted((ROOT / "results" / "v2" / "tahminler").glob(f"{k}_s*.npz")):
            z = np.load(f)
            g, d, t = z[f"{bolum}_gercek"], z[f"{bolum}_dosya"], z[f"son_{bolum}_tahmin"]
            if gt is None:
                gt, dosya = g.astype(int), d
            assert np.array_equal(gt, g) and np.array_equal(dosya, d), f"{f.name}: segment sirasi farkli"
            tah[k][int(f.stem.rsplit("_s", 1)[1])] = t.astype(int)
    _, kayit = np.unique(dosya, return_inverse=True)
    kayit_sinif = np.zeros(kayit.max() + 1, dtype=int)
    kayit_sinif[kayit] = gt
    W = BA.bootstrap_agirliklari(kayit_sinif, 10000, np.random.default_rng(2026))
    return gt, kayit, kayit_sinif, tah, W


def main():
    sonuc = {"tarih": datetime.now().strftime("%Y%m%d_%H%M"), "on_kayitli": False, "B": 10000, "uretec_seed": 2026,
             "model": "son epoch", "birincil_bolum_bazinda": {}, "artirilmayan_siniflar": {}}
    for bolum in ("test", "val", "valtest"):
        ctx = yukle(["baseline", "C4p"], bolum)
        r = IA.kars(ctx, "C4p", "baseline", "A_belirgin")
        r["n_kayit"] = int(ctx[2].size)
        sonuc["birincil_bolum_bazinda"][bolum] = r
        print(f"C4p-baseline Grup A [{bolum:7s}] n={r['n_kayit']}: {r['delta_f1']:+.4f} "
              f"[{r['ga95'][0]:+.4f}, {r['ga95'][1]:+.4f}] p={r['p_bootstrap']:.4f}")
    ctx = yukle(["baseline"] + KONFIG, "valtest")
    sonuc["artirilmayan_sinif_listesi"] = BA.GRUPLAR["artirilmayan"]
    for k in KONFIG:
        r = IA.kars(ctx, k, "baseline", "artirilmayan")
        v = r["sinif"]
        r["sinif_min"], r["sinif_maks"] = min(v, key=v.get), max(v, key=v.get)
        sonuc["artirilmayan_siniflar"][k] = r
        print(f"18 sinif {k:11s}: {r['delta_f1']:+.4f} [{r['ga95'][0]:+.4f}, {r['ga95'][1]:+.4f}] "
              f"p={r['p_bootstrap']:.4f}  sinif {v[r['sinif_min']]:+.4f}..{v[r['sinif_maks']]:+.4f}")
    yol = ROOT / "results" / "v2" / "sonradan_duyarlilik.json"
    yol.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print("kaydedildi:", yol)


if __name__ == "__main__":
    main()
