"""Yeni (cesitli klip) deney sonuclarini TUM JSON'lardan toplar ve
tek bir ozet (results_summary_new.json) + konsol tablosu uretir.

Kaynaklar:
  - FAD:          results/fad_sonuclar.json (Tur1), _tur2, _tur3, _4
  - Siniflandirma: results/iteratif_loop_{1,tur2,tur3,tur4}_*.json (EN YENI)
  - McNemar:       results/mcnemar_*.json (varsa)

Eksik turlar 'BEKLENIYOR' olarak isaretlenir (gece kosumu surarken de calisir).
Kullanim:  python scripts/sonuclari_topla.py
"""
import json, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES  = ROOT / "results"
HEDEF = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]

def yukle(p):
    try:
        with open(p, encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def en_yeni(desen):
    """Desenle eslesen EN YENI (mtime) dosyayi don."""
    fs = sorted(glob.glob(str(RES / desen)), key=os.path.getmtime)
    return fs[-1] if fs else None

# ── FAD ───────────────────────────────────────────────────────────────────────
FAD_DOSYA = {"Tur1": "fad_sonuclar.json", "Tur2": "fad_sonuclar_tur2.json",
             "Tur3": "fad_sonuclar_tur3.json", "Tur4": "fad_sonuclar_4.json"}
fad = {}
for tur, dosya in FAD_DOSYA.items():
    d = yukle(RES / dosya)
    if d:
        fad[tur] = {s: round(d[s]["fad"], 2) for s in HEDEF if s in d}
        fad[tur]["ORT"] = round(sum(d[s]["fad"] for s in HEDEF if s in d) / 4, 2)

# ── Siniflandirma (en yeni JSON her tur icin) ─────────────────────────────────
CLS_DESEN = {"Tur1": "iteratif_loop_1_2026*.json", "Tur2": "iteratif_loop_tur2_2026*.json",
             "Tur3": "iteratif_loop_tur3_2026*.json", "Tur4": "iteratif_loop_tur4_2026*.json"}
cls = {}
for tur, desen in CLS_DESEN.items():
    p = en_yeni(desen)
    # sadece BUGUNku (yeni) kosumlari al: dosya adinda 20260601+ tarih
    if p and ("20260601" in p or "20260602" in p or "20260603" in p):
        d = yukle(p)
        kayit = {"_dosya": os.path.basename(p)}
        for mdl in ["cnn", "lgbm"]:
            m = d.get(mdl, {})
            if m and "baseline" in m and "augmented" in m:
                b, a = m["baseline"], m["augmented"]
                kayit[mdl] = {
                    "baseline": round(b["test_f1"], 4),
                    "augmented": round(a["test_f1"], 4),
                    "delta": round(a["test_f1"] - b["test_f1"], 4),
                    "sinif": {s: {"b": round(b["sinif_f1"].get(s, 0), 4),
                                  "a": round(a["sinif_f1"].get(s, 0), 4),
                                  "d": round(a["sinif_f1"].get(s, 0) - b["sinif_f1"].get(s, 0), 4)}
                              for s in HEDEF},
                }
        cls[tur] = kayit

# ── McNemar (varsa) ───────────────────────────────────────────────────────────
mcnemar = {}
for p in glob.glob(str(RES / "mcnemar_*.json")):
    d = yukle(p)
    if d: mcnemar[os.path.basename(p)] = d

# ── Konsol ozeti ──────────────────────────────────────────────────────────────
print("=" * 64)
print("  YENI SONUC OZETI (cesitli klipler)")
print("=" * 64)
print("\n[FAD] (dusuk = iyi)")
print(f"  {'Sinif':<16} {'Tur1':>7} {'Tur2':>7} {'Tur3':>7} {'Tur4':>7}")
for s in HEDEF + ["ORT"]:
    row = f"  {s:<16}"
    for t in ["Tur1", "Tur2", "Tur3", "Tur4"]:
        v = fad.get(t, {}).get(s)
        row += f" {v:>7.2f}" if v is not None else f" {'BEKL':>7}"
    print(row)

print("\n[SINIFLANDIRMA] makro-F1 (baseline -> augmented, delta)")
for t in ["Tur1", "Tur2", "Tur3", "Tur4"]:
    k = cls.get(t)
    if not k:
        print(f"  {t}: BEKLENIYOR")
        continue
    for mdl in ["cnn", "lgbm"]:
        if mdl in k:
            m = k[mdl]
            print(f"  {t} {mdl.upper():<4}: {m['baseline']:.4f} -> {m['augmented']:.4f}  ({m['delta']:+.4f})")

print("\n[McNEMAR]", "bulundu: " + ", ".join(mcnemar) if mcnemar else "henuz yok")

# ── Kaydet ────────────────────────────────────────────────────────────────────
ozet = {"fad": fad, "siniflandirma": cls, "mcnemar_dosyalari": list(mcnemar)}
out = RES / "results_summary_new.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(ozet, f, ensure_ascii=False, indent=2)
print(f"\nOzet kaydedildi: {out}")
