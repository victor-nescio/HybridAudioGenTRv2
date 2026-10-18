"""
Revize makalenin tablolarindaki tum sayilar (elle sayi yok). Birincil degerlendirme: son epoch, val+test.

  tablo_aile        : P1 + on kayitli 11 ikincil karsilastirma (ikincil_analiz_tam.json)
  tablo_26sinif     : 26 sinif makro-F1, seed ortalamasi +- SS, eslesmis fark (betimsel json)
  tablo_specaug     : sinif bazinda F1 (kopek, kus, hava, diger hayvanlar) — baseline / SpecAug / C4' /
                      SpecAug+C4', seed 42-44 ortalamasi (ilk gonderimdeki Tablo V'in karsiligi)
  tablo_kalite      : FAD (VGGish) + kalibrasyon, CLAP skoru, siniflandirici gecerliligi (kalite_metrikleri.json)
  tablo_clap        : CLAP-test sifir-atis / kNN (clap_finetune_*.json)

Kullanim: python scripts/25_makale_tablolari.py  ->  results/v2/makale_tablolari.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results" / "v2"


def sinif_f1_son(konfig, seed, siniflar_idx):
    z = np.load(V2 / "tahminler" / f"{konfig}_s{seed}.npz")
    gt = np.concatenate([z["val_gercek"], z["test_gercek"]]).astype(int)
    pr = np.concatenate([z["son_val_tahmin"], z["son_test_tahmin"]]).astype(int)
    return f1_score(gt, pr, labels=siniflar_idx, average=None, zero_division=0)


def main():
    import importlib.util
    s = importlib.util.spec_from_file_location("ba", ROOT / "scripts" / "16_birincil_analiz.py")
    BA = importlib.util.module_from_spec(s)
    s.loader.exec_module(BA)
    out = {}
    d = json.loads((V2 / "ikincil_analiz_tam.json").read_text(encoding="utf-8"))
    out["tablo_aile"] = {"P1": d["P1_hatirlatma"], "aile": d["aile"]}

    # 26 sinif
    b = json.loads((V2 / "betimsel_baseline_C4p_H1_C2p_C3p_C4p_kaydir_specaug_specaugC4p_doz25_doz100_doz200.json")
                   .read_text(encoding="utf-8"))
    t26 = {}
    for k in ["baseline", "H1", "C2p", "C3p", "C4p", "C4p_kaydir", "specaug", "specaugC4p", "doz25", "doz100", "doz200",
              "beats_baseline", "beats_H1", "beats_C4p"]:
        js = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((V2 / "siniflandirma").glob(f"{k}_s*.json"))]
        v = np.array([j["son_epoch"]["val_f1"] for j in js]); t = np.array([j["son_epoch"]["test_f1"] for j in js])
        fark = b["farklar"].get(f"{k}-baseline:makro26_son_valtest")
        t26[k] = {"n_seed": len(js), "val": [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)],
                  "test": [round(float(t.mean()), 4), round(float(t.std(ddof=1)), 4)],
                  "fark_valtest": [round(fark["ort"], 4), round(fark["ss"], 4)] if fark else None}
    out["tablo_26sinif"] = t26

    # SpecAugment karsilastirmasi (sinif bazinda F1, seed 42-44)
    siniflar = ["kopek_sesleri", "kus_sesleri", "hava_olaylari", "diger_hayvanlar"]
    ix = [BA.SINIF_SIRASI.index(c) for c in siniflar]
    tsa = {}
    for k in ("baseline", "specaug", "C4p", "specaugC4p"):
        M = np.stack([sinif_f1_son(k, sd, ix) for sd in (42, 43, 44)])
        tsa[k] = {c: [round(float(M[:, i].mean()), 4), round(float(M[:, i].std(ddof=1)), 4)] for i, c in enumerate(siniflar)}
    out["tablo_specaug"] = {"not": "son epoch, val+test, seed 42-44 ortalamasi +- SS", "f1": tsa}

    # Kalite
    kf = V2 / "kalite_metrikleri.json"
    out["tablo_kalite"] = json.loads(kf.read_text(encoding="utf-8")) if kf.exists() else None

    # CLAP
    tc = {}
    for f in sorted(V2.glob("clap_finetune_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        tc["hazir"] = {k: j["test_hazir_clap"][k] for k in ("zs_en", "zs_tr", "knn_acc")}
        tc[j["dongu"]] = {k: j["test"][k] for k in ("zs_en", "zs_tr", "knn_acc", "nan_ses_orani")}
    out["tablo_clap"] = tc

    (V2 / "makale_tablolari.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["tablo_specaug"], ensure_ascii=False, indent=1))
    print("kalite:", "var" if out["tablo_kalite"] else "henuz yok")


if __name__ == "__main__":
    main()
