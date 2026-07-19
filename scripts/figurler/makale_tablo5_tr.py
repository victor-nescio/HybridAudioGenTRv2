# -*- coding: utf-8 -*-
"""Tablo 5'in Türkçe versiyonu."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "outputs" / "makale_figurleri"
GREEN = "#1E8E5A"; GRAY = "#5D6D7E"; INK = "#2C3E50"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.dpi": 200})

BAS = ["Sınıf", "Taban", "+SpecAugment", "+Üretici (C4)", "+Birleşik"]
SATIR = [
    ("köpek", "0,665", "0,709", "0,705", "0,670"),
    ("kuş", "0,591", "0,638", "0,669", "0,614"),
    ("hava", "0,553", "0,573", "0,605", "0,596"),
    ("diğer hayvanlar (kontrol)", "0,582", "0,561", "0,578", "0,612"),
]
DIP = ("Sınıf bazlı F1; tüm modeller için aynı eğitim protokolü ve bölünme. McNemar (birleşik hedefler): "
       "üretici vs. SpecAugment χ²=0,06, p=0,81 (ns); birleşik vs. SpecAugment χ²=5,70, p=0,017 "
       "(kötüleşme; köpek p<10⁻⁶). Tüm test kümesi: SpecAugment vs. üretici p<0,001; birleşik vs. "
       "SpecAugment p=1,4×10⁻⁶.")

fig, ax = plt.subplots(figsize=(9.2, 3.1))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
kolx = [0.03, 0.42, 0.56, 0.74, 0.92]
baslik_y, cizgi_y, ilk, h = 0.90, 0.83, 0.72, 0.13
for x, b in zip(kolx, BAS):
    ax.text(x, baslik_y, b, fontsize=10, fontweight="bold", color=INK,
            ha="left" if x == kolx[0] else "center")
ax.plot([0.01, 0.99], [cizgi_y] * 2, color=INK, lw=1.2)
for i, sat in enumerate(SATIR):
    yy = ilk - i * h
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.01, yy - h * 0.42), 0.98, h * 0.88,
                     facecolor="#F4F6F7", edgecolor="none"))
    sayi = [float(s.replace(",", ".")) for s in sat[1:]]
    for j, (x, v) in enumerate(zip(kolx, sat)):
        en_iyi = j > 0 and float(v.replace(",", ".")) == max(sayi)
        ax.text(x, yy, v, fontsize=9.6, ha="left" if j == 0 else "center", va="center",
                color=GREEN if en_iyi else INK, fontweight="bold" if en_iyi else "normal")
alt = ilk - (len(SATIR) - 0.5) * h
ax.plot([0.01, 0.99], [alt] * 2, color="#BDC3C7", lw=0.8)
ax.text(0.01, alt - 0.06, DIP, fontsize=7.2, color=GRAY, ha="left", va="top", wrap=True)
fig.savefig(OUT / "tablo5_tr.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("tablo5_tr.png uretildi")
