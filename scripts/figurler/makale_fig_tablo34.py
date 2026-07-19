# -*- coding: utf-8 -*-
"""Şekil 4 (sınıf F1, yeni ★ işaretleri) ve Tablo 3 görseli — EN + TR.
Oranlar: image6 ≈ 736x1063 (0.69), image12 ≈ 2.8 yatay."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "makale_figurleri"
GREEN = "#1E8E5A"; ORANGE = "#E8891D"; RED = "#B03A2E"; INK = "#2C3E50"; GRAY = "#5D6D7E"

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#BDC3C7", "axes.grid": True, "grid.color": "#ECF0F1",
    "axes.axisbelow": True, "savefig.dpi": 200, "figure.facecolor": "white",
})

def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

BASE = jload(ROOT / "results" / "iteratif_loop_coherent_20260604_0320.json")["cnn"]["baseline"]["sinif_f1"]
SECILEN = {"kopek_sesleri", "kus_sesleri", "hava_olaylari", "diger_hayvanlar"}

AD = {
 "ayak_sesleri": ("footstep sounds", "ayak sesleri"),
 "at_sesleri": ("horse sounds", "at sesleri"),
 "saat_can_sesleri": ("clock/bell", "saat–çan sesleri"),
 "cocuk_bebek_sesleri": ("children/baby", "çocuk–bebek sesleri"),
 "kapi_sesleri": ("door sounds", "kapı sesleri"),
 "ucak_sesleri": ("aircraft sounds", "uçak sesleri"),
 "kara_tasitlari": ("land vehicles", "kara taşıtları"),
 "gemi_tekne_sesleri": ("ship/boat", "gemi–tekne sesleri"),
 "tren_sesleri": ("train sounds", "tren sesleri"),
 "ev_ici_sesler": ("indoor sounds", "ev içi sesler"),
 "endustri_fabrika": ("industry/factory", "endüstri–fabrika"),
 "kopek_sesleri": ("dog sounds", "köpek sesleri"),
 "spor_eglence": ("sports/entertainment", "spor–eğlence"),
 "ofis_iletisim": ("office/communication", "ofis–iletişim"),
 "kalabalik_sesleri": ("crowd sounds", "kalabalık sesleri"),
 "silah_patlama": ("weapon/explosion", "silah–patlama"),
 "sehir_atmosferi": ("city atmosphere", "şehir atmosferi"),
 "kus_sesleri": ("bird sounds", "kuş sesleri"),
 "acil_servis": ("emergency service", "acil servis"),
 "doga_atmosferi": ("nature atmosphere", "doğa atmosferi"),
 "diger_hayvanlar": ("other animals", "diğer hayvanlar"),
 "su_sesleri": ("water sounds", "su sesleri"),
 "hava_olaylari": ("weather events", "hava olayları"),
 "tarim_ciftlik": ("farm/livestock", "tarım–çiftlik"),
 "dini_mekan": ("religious place", "dini mekân"),
 "insaat_sesleri": ("construction", "inşaat sesleri"),
}

MAKRO = 0.6716

L = {
 "en": dict(x="F1 score (test set)",
            leg=["F1 ≥ 0.75", "0.50 ≤ F1 < 0.75", "F1 < 0.50",
                 "★ selected for augmentation", f"macro-F1 mean = {MAKRO}"]),
 "tr": dict(x="F1 skoru (test kümesi)",
            leg=["F1 ≥ 0,75", "0,50 ≤ F1 < 0,75", "F1 < 0,50",
                 "★ artırma için seçilen sınıflar", f"makro-F1 ort. = {str(MAKRO).replace('.', ',')}"]),
}


def fig4_classf1(dil):
    t = L[dil]
    sirali = sorted(BASE.items(), key=lambda x: x[1])
    adlar = [AD[k][0 if dil == "en" else 1] + (" ★" if k in SECILEN else "")
             for k, _ in sirali]
    degerler = [v for _, v in sirali]
    renk = [GREEN if v >= 0.75 else (ORANGE if v >= 0.5 else RED) for v in degerler]
    fig, ax = plt.subplots(figsize=(7.4, 10.7))
    y = np.arange(len(adlar))
    ax.barh(y, degerler, 0.62, color=renk, edgecolor="white")
    for yi, v in zip(y, degerler):
        ax.text(v + 0.012, yi, f"{v:.3f}".replace(".", "," if dil == "tr" else "."),
                va="center", fontsize=7.8, color=INK)
    ax.axvline(MAKRO, color="#1F5FA8", ls="--", lw=1.2)
    ax.set_yticks(y)
    etiketler = ax.set_yticklabels(adlar, fontsize=8.2)
    for et, (k, _) in zip(etiketler, sirali):
        if k in SECILEN:
            et.set_fontweight("bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(t["x"], fontsize=9)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg = [Patch(facecolor=GREEN, label=t["leg"][0]),
           Patch(facecolor=ORANGE, label=t["leg"][1]),
           Patch(facecolor=RED, label=t["leg"][2]),
           Line2D([0], [0], marker="*", color="w", markerfacecolor=INK,
                  markersize=11, label=t["leg"][3]),
           Line2D([0], [0], color="#1F5FA8", ls="--", label=t["leg"][4])]
    ax.legend(handles=leg, fontsize=7.6, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig4_classf1_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


TABLO = {
 "en": dict(
   bas=["Target / Run", "χ²", "p", ""],
   satirlar=[
     ("Combined targets (H1, pretrained · EN)", "25.5", "4.4×10⁻⁷", "***"),
     ("Combined targets (C2, Cycle 1 · EN)", "20.6", "5.8×10⁻⁶", "***"),
     ("Combined targets (C3, Cycle 1 · TR)", "23.1", "1.5×10⁻⁶", "***"),
     ("Combined targets (C4, Cycle 2 · EN)", "66.4", "3.3×10⁻¹⁶", "***"),
     ("Control (other animals, C4)", "1.27", "0.261", "ns"),
   ],
   dip="Yates-corrected McNemar test (df=1). Combined targets: dog+bird+weather, n=2,343 test segments. "
       "Holm–Bonferroni correction applied across classes. *** p<0.001; ns: not significant."),
 "tr": dict(
   bas=["Hedef / Koşu", "χ²", "p", ""],
   satirlar=[
     ("Birleşik hedefler (H1, ön-eğitimli · EN)", "25,5", "4,4×10⁻⁷", "***"),
     ("Birleşik hedefler (C2, Döngü 1 · EN)", "20,6", "5,8×10⁻⁶", "***"),
     ("Birleşik hedefler (C3, Döngü 1 · TR)", "23,1", "1,5×10⁻⁶", "***"),
     ("Birleşik hedefler (C4, Döngü 2 · EN)", "66,4", "3,3×10⁻¹⁶", "***"),
     ("Kontrol (diğer hayvanlar, C4)", "1,27", "0,261", "ns"),
   ],
   dip="Yates düzeltmeli McNemar testi (sd=1). Birleşik hedefler: köpek+kuş+hava, n=2.343 test segmenti. "
       "Sınıflar arasında Holm–Bonferroni düzeltmesi uygulanmıştır. *** p<0,001; ns: anlamlı değil."),
}


def tablo3(dil):
    t = TABLO[dil]
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    ax.axis("off"); ax.grid(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    kolx = [0.03, 0.62, 0.76, 0.93]
    satir_h = 0.115
    baslik_y, cizgi_y, ilk_satir = 0.90, 0.845, 0.775
    for x, b in zip(kolx, t["bas"]):
        ax.text(x, baslik_y, b, fontsize=10.5, fontweight="bold",
                color=INK, ha="left" if x == kolx[0] else "center")
    ax.plot([0.01, 0.99], [cizgi_y] * 2, color=INK, lw=1.2)
    for i, (ad, chi, p, sig) in enumerate(t["satirlar"]):
        yy = ilk_satir - i * satir_h
        if sig == "***":
            ax.add_patch(plt.Rectangle((0.01, yy - satir_h * 0.42), 0.98, satir_h * 0.88,
                         facecolor="#E9F7EF", edgecolor="none"))
        elif i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, yy - satir_h * 0.42), 0.98, satir_h * 0.88,
                         facecolor="#F4F6F7", edgecolor="none"))
        ax.text(kolx[0], yy, ad, fontsize=9.8, color=INK, ha="left", va="center")
        ax.text(kolx[1], yy, chi, fontsize=9.8, color=INK, ha="center", va="center")
        ax.text(kolx[2], yy, p, fontsize=9.8, color=INK, ha="center", va="center")
        ax.text(kolx[3], yy, sig, fontsize=9.8, fontweight="bold", va="center",
                ha="center", color=GREEN if sig == "***" else GRAY)
    alt_y = ilk_satir - (len(t["satirlar"]) - 1) * satir_h - satir_h * 0.5
    ax.plot([0.01, 0.99], [alt_y] * 2, color="#BDC3C7", lw=0.8)
    ax.text(0.01, alt_y - 0.05, t["dip"], fontsize=7.6, color=GRAY,
            ha="left", va="top", wrap=True)
    fig.savefig(OUT / f"tablo3_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for dil in ("en", "tr"):
        fig4_classf1(dil); tablo3(dil)
    print("uretildi:", sorted(p.name for p in OUT.glob("fig4*")) +
          sorted(p.name for p in OUT.glob("tablo3*")))
