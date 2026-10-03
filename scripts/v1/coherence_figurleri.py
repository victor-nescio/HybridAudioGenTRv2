"""TUTARLILIK BULGUSU figurleri (POZITIF sonuc, 2026-06-05).
Generative augmentasyon akustik olarak tutarli dusuk-kaynakli siniflari
(kopek/kus/hava) anlamli iyilestiriyor; heterojen catch-all (diger_hayvanlar)
fayda gormuyor. 4 uretim konfigurasyonunda replike, hepsi McNemar ***.

Cikti: outputs/report/coherence_figs/
  k1_replikasyon_4konfig — sinif bazli F1 delta, 4 konfig (POZITIF, tutarli)
  k2_mcnemar             — tutarli-3 McNemar (4 kosu, hepsi ***) + sinif anlamlilik isi haritasi
  k3_mekanizma           — tutarli siniflar vs catch-all kontrol kontrasti
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "report" / "coherence_figs"
OUT.mkdir(parents=True, exist_ok=True)

BLUE="#1380A1"; RED="#C0392B"; GRAY="#5D6D7E"; GREEN="#27AE60"; ORANGE="#E8891D"; PURPLE="#7B2D8B"
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                     "savefig.dpi":200, "savefig.bbox":"tight", "figure.facecolor":"white"})

def kaydet(fig, ad):
    p = OUT / f"{ad}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  {p}")

def jload(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

# ── Veri kaynaklari ───────────────────────────────────────────────────────────
ITER = {
 "H1":  "iteratif_loop_coherent_20260604_0320.json",
 "coh2":"iteratif_loop_coherent2_Dongu1_EN_20260604_0942.json",
 "coh3":"iteratif_loop_coherent3_Dongu1_TR_20260604_2237.json",
 "coh4":"iteratif_loop_coherent4_Dongu2_EN_20260605_0726.json",
}
MCN = {
 "H1":"mcnemar_coherent.json", "coh2":"mcnemar_coh2.json",
 "coh3":"mcnemar_coh3.json",   "coh4":"mcnemar_coh4.json",
}
KONFIG = ["H1","coh2","coh3","coh4"]
KONFIG_AD = ["H1\n(ön-eğitimli+EN)","coh2\n(Döngü1+EN)","coh3\n(Döngü1+TR)","coh4\n(Döngü2+EN)"]
RENK = ["#AED6F1", BLUE, PURPLE, GREEN]

TUTARLI = ["kopek_sesleri","kus_sesleri","hava_olaylari"]
KONTROL = ["diger_hayvanlar"]
SINIF = TUTARLI + KONTROL
SINIF_K = ["köpek","kuş","hava","diğer hayv.\n(catch-all)"]

D  = {k: jload(ROOT/"results"/"v1"/ITER[k]) for k in KONFIG}
M  = {k: jload(ROOT/"results"/"v1"/MCN[k])  for k in KONFIG}
BASE = D["H1"]["cnn"]["baseline"]["sinif_f1"]

def yildiz(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))

# ── k1: 4-konfig replikasyon (sinif F1 delta) ─────────────────────────────────
def k1_replikasyon():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(SINIF)); w = 0.2
    for i,k in enumerate(KONFIG):
        aug = D[k]["cnn"]["augmented"]["sinif_f1"]
        dv = [aug[s]-BASE[s] for s in SINIF]
        ax.bar(x+(i-1.5)*w, dv, w*0.9, label=KONFIG_AD[i].replace("\n"," "),
               color=RENK[i], edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    # tutarli bolgeyi golge + etiketler icin ust pay
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi*1.22)
    ax.axvspan(-0.5, 2.5, color=GREEN, alpha=0.06)
    ax.axvspan(2.5, 3.5, color=RED, alpha=0.06)
    ax.text(1.0, ax.get_ylim()[1]*0.93, "AKUSTİK OLARAK TUTARLI\n(artırılan hedefler)",
            ha="center", va="top", fontsize=8.5, color=GREEN, fontweight="bold")
    ax.text(3.0, ax.get_ylim()[1]*0.93, "CATCH-ALL\n(kontrol)",
            ha="center", va="top", fontsize=8.5, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SINIF_K, fontsize=9.5)
    ax.set_ylabel("Sınıf F1 değişimi (artırılmış − taban)")
    ax.set_title("Üretici artırma tutarlı sınıfları iyileştiriyor — dört üretim "
                 "konfigürasyonunda replikasyon", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.22))
    fig.tight_layout(); kaydet(fig,"k1_replikasyon_4konfig")

# ── k2: McNemar tutarli-3 + sinif anlamlilik isi haritasi ─────────────────────
def k2_mcnemar():
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.4),gridspec_kw={"width_ratios":[1.1,1]})
    # (a) tutarli-3 chi2 bar + yildiz
    chi=[M[k]["testler"]["tutarli_3"]["chi2"] for k in KONFIG]
    ps =[M[k]["testler"]["tutarli_3"]["p"] for k in KONFIG]
    b=a1.bar(range(4), chi, color=RENK, edgecolor="white", width=0.62)
    for i,(bar,c,p) in enumerate(zip(b,chi,ps)):
        a1.text(bar.get_x()+bar.get_width()/2, c+1.5, f"{yildiz(p)}\np={p:.1e}",
                ha="center", fontsize=8.5, fontweight="bold", color=RED)
    a1.axhline(10.83, color=GRAY, ls="--", lw=1, )
    a1.text(3.45, 11.5, "p=0.001", fontsize=7.5, color=GRAY, ha="right")
    a1.set_xticks(range(4)); a1.set_xticklabels([k for k in KONFIG], fontsize=9)
    a1.set_ylabel("McNemar χ²  (Yates, df=1)")
    a1.set_title("Tutarlı-3 (köpek+kuş+hava) — dört koşu da ***", fontsize=10)
    a1.set_ylim(0, max(chi)*1.22)
    # (b) sinif x kosu anlamlilik isi haritasi (-log10 p)
    mat=np.zeros((len(SINIF),4)); ann=[]
    for j,k in enumerate(KONFIG):
        sat=[]
        for i,s in enumerate(SINIF):
            c=M[k]["testler"]["sinif_basina"][s]
            mat[i,j]=min(-np.log10(max(c["p"],1e-16)), 16)
            sat.append(c["anlam"])
        ann.append(sat)
    im=a2.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=6)
    a2.set_xticks(range(4)); a2.set_xticklabels(KONFIG, fontsize=9)
    a2.set_yticks(range(len(SINIF))); a2.set_yticklabels(SINIF_K, fontsize=9)
    for i in range(len(SINIF)):
        for j in range(4):
            a2.text(j,i, ann[j][i], ha="center", va="center", fontsize=9,
                    color="white" if mat[i,j]>3 else "black", fontweight="bold")
    a2.set_title("Sınıf bazlı anlamlılık (−log₁₀p)\nkuş & hava istikrarlı; catch-all hiç", fontsize=9.8)
    cb=fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04); cb.set_label("−log₁₀ p", fontsize=8)
    fig.suptitle("İstatistiksel anlamlılık: tutarlı hedeflerde augmentasyon güçlü ve tekrarlanabilir",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout(); kaydet(fig,"k2_mcnemar")

# ── k3: mekanizma — tutarli vs catch-all kontrast ─────────────────────────────
def k3_mekanizma():
    # her konfig icin tutarli-3 ortalama delta ve catch-all delta
    tut=[]; cat=[]
    for k in KONFIG:
        aug=D[k]["cnn"]["augmented"]["sinif_f1"]
        tut.append(np.mean([aug[s]-BASE[s] for s in TUTARLI]))
        cat.append(aug["diger_hayvanlar"]-BASE["diger_hayvanlar"])
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    x=np.arange(4); w=0.36
    b1=ax.bar(x-w/2, tut, w, label="Tutarlı-3 ort. (köpek+kuş+hava)", color=GREEN, edgecolor="white")
    b2=ax.bar(x+w/2, cat, w, label="Catch-all (diğer hayvanlar)", color=RED, edgecolor="white")
    for bar,v in list(zip(b1,tut))+list(zip(b2,cat)):
        ax.text(bar.get_x()+bar.get_width()/2, v+(0.001 if v>=0 else -0.003),
                f"{v:+.3f}", ha="center", fontsize=8,
                color=GREEN if v>=0 else RED, fontweight="bold",
                va="bottom" if v>=0 else "top")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(KONFIG_AD, fontsize=9)
    ax.set_ylabel("Ortalama F1 değişimi")
    ax.set_title("Mekanizma: akustik tutarlılık augmentasyon faydasının ön koşulu\n"
                 "(tutarlı hep ↑, catch-all hep ≈0/↓ — aynı eğitimde)",
                 fontsize=10.3, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); kaydet(fig,"k3_mekanizma")

if __name__=="__main__":
    print("Tutarlilik bulgusu figurleri uretiliyor...")
    k1_replikasyon(); k2_mcnemar(); k3_mekanizma()
    print(f"Tamamlandi -> {OUT}")
