"""k6 — M2: Tutarlilik ilkesi augmentasyon-turunden BAGIMSIZ.
Hem generative (Dongu2) hem geleneksel (SpecAugment, GLOBAL uygulanir) tutarli
siniflari iyilestirir; heterojen catch-all'i iyilestirmez (hatta bozar).
Cikti: outputs/report/coherence_figs/k6_specaug.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report" / "coherence_figs"
def jload(p): return json.load(open(p, encoding="utf-8"))

GREEN="#27AE60"; BLUE="#1380A1"; ORANGE="#E8891D"; RED="#C0392B"; GRAY="#5D6D7E"
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})

sa=jload(ROOT/"results"/"iteratif_loop_specaug_20260606_0129.json")
# H1: 4 sinifin dördü de (3 tutarli + catch-all kontrol) AYNI kosulda augmente
# edildigi TEK config -> generative icin temiz apples-to-apples.
h1=jload(ROOT/"results"/"iteratif_loop_coherent_20260604_0320.json")
base=h1["cnn"]["baseline"]["sinif_f1"]
sao=sa["cnn"]["baseline"]["sinif_f1"]      # SpecAugment-only (global)

SIN=["kopek_sesleri","kus_sesleri","hava_olaylari","diger_hayvanlar"]
ETK=["köpek","kuş","hava","diğer hayv.\n(catch-all)"]
def d(dd,s): return dd[s]-base[s]
spec=[d(sao,s) for s in SIN]
gen =[d(h1["cnn"]["augmented"]["sinif_f1"],s) for s in SIN]

fig,ax=plt.subplots(figsize=(8.5,4.8))
x=np.arange(len(SIN)); w=0.34
b1=ax.bar(x-w/2, gen,  w, label="Üretici artırma (hedefli)", color=GREEN, edgecolor="white")
b2=ax.bar(x+w/2, spec, w, label="SpecAugment (global)", color=BLUE,  edgecolor="white")
ax.axhline(0,color="black",lw=0.8)
ax.axvspan(-0.5,2.5,color=GREEN,alpha=0.05); ax.axvspan(2.5,3.5,color=RED,alpha=0.05)
ax.text(1.0, ax.get_ylim()[1]*0.93, "AKUSTİK OLARAK TUTARLI", ha="center", fontsize=8.5,
        color=GREEN, fontweight="bold")
ax.text(3.0, ax.get_ylim()[1]*0.93, "CATCH-ALL", ha="center", fontsize=8.5,
        color=RED, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ETK, fontsize=9.5)
ax.set_ylabel("Sınıf F1 değişimi (− taban)")
ax.set_title("Tutarlılık ilkesi augmentasyon türünden bağımsız: üretici VE SpecAugment\n"
             "tutarlı sınıfları iyileştirir; heterojen catch-all'ı iyileştirmez",
             fontsize=10.3, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
fig.savefig(OUT/"k6_specaug.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Kaydedildi:", OUT/"k6_specaug.png")
print(f"Makro-F1: taban 0.6716 | SpecA-only {sa['cnn']['baseline']['test_f1']:.4f} | "
      f"stacked {sa['cnn']['augmented']['test_f1']:.4f}")
print("catch-all: gen(H1)=%.4f  SpecA(global)=%.4f" % (gen[3], spec[3]))
