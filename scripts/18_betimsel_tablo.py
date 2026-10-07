"""
Betimsel tablo: konfigurasyon x seed, birincil degerlendirme (son epoch, val+test birlikte) ve
sureklilik icin (en iyi-val checkpoint, test). 26 sinif makro-F1 ve grup makro-F1'leri; seed'e gore
eslestirilmis farklar; ortalama +- SS.

Kullanim: python scripts/18_betimsel_tablo.py --konfigler baseline C4p
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("ba", ROOT / "scripts" / "16_birincil_analiz.py")
BA = importlib.util.module_from_spec(_s)
_s.loader.exec_module(BA)


def grup_f1(gt, pr, siniflar):
    ix = [BA.SINIF_SIRASI.index(s) for s in siniflar]
    return float(np.mean(f1_score(gt, pr, labels=ix, average=None, zero_division=0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--konfigler", nargs="+", required=True)
    a = p.parse_args()
    satirlar = {}
    for k in a.konfigler:
        for f in sorted((ROOT / "results" / "v2" / "tahminler").glob(f"{k}_s*.npz")):
            s = int(f.stem.rsplit("_s", 1)[1])
            z = np.load(f)
            gt = np.concatenate([z["val_gercek"], z["test_gercek"]]).astype(int)
            pr = np.concatenate([z["son_val_tahmin"], z["son_test_tahmin"]]).astype(int)
            r = {"makro26_son_valtest": f1_score(gt, pr, average="macro", zero_division=0),
                 "makro26_enIyiVal_test": f1_score(z["test_gercek"], z["test_tahmin"], average="macro", zero_division=0)}
            for g, sn in BA.GRUPLAR.items():
                r[g] = grup_f1(gt, pr, sn)
            satirlar[(k, s)] = r
    ref = a.konfigler[0]
    print(f"{'konfig':10s} {'seed':>4s} {'26-makro (son,val+test)':>24s} {'26-makro (enIyi,test)':>22s} " +
          " ".join(f"{g:>12s}" for g in BA.GRUPLAR))
    for (k, s), r in sorted(satirlar.items()):
        print(f"{k:10s} {s:4d} {r['makro26_son_valtest']:24.4f} {r['makro26_enIyiVal_test']:22.4f} " +
              " ".join(f"{r[g]:12.4f}" for g in BA.GRUPLAR))
    print("\nEslestirilmis farklar (konfig - " + ref + "), ortalama +- SS [seed bazli degerler]:")
    ozet = {}
    for k in a.konfigler[1:]:
        ortak = sorted(s for (kk, s) in satirlar if kk == k and (ref, s) in satirlar)
        for olcu in ["makro26_son_valtest"] + list(BA.GRUPLAR):
            d = [satirlar[(k, s)][olcu] - satirlar[(ref, s)][olcu] for s in ortak]
            ozet[f"{k}-{ref}:{olcu}"] = {"ort": float(np.mean(d)), "ss": float(np.std(d, ddof=1)) if len(d) > 1 else None,
                                         "seed": ortak, "degerler": [round(x, 4) for x in d]}
            print(f"  {k:8s} {olcu:22s} {np.mean(d):+.4f} +- {np.std(d, ddof=1) if len(d) > 1 else float('nan'):.4f}  {[f'{x:+.3f}' for x in d]}")
    cikti = ROOT / "results" / "v2" / f"betimsel_{'_'.join(a.konfigler)}.json"
    cikti.write_text(json.dumps({"satirlar": {f"{k}_s{s}": r for (k, s), r in satirlar.items()}, "farklar": ozet},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print("kaydedildi:", cikti)


if __name__ == "__main__":
    main()
