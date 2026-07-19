# -*- coding: utf-8 -*-
"""Makale revizyonu için nötr (tutarlılık terminolojisiz) şekiller — EN + TR.

Değiştirilen görsellerin en-boy oranları korunur:
  fig1 pipeline : 736x1024  (0.72)
  fig6 targets  : 736x446   (1.65)
  fig7 mean     : 736x432   (1.70)
  fig8 mcnemar  : 738x351   (2.10)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "makale_figurleri"
OUT.mkdir(exist_ok=True)

# Palet: dört konfigürasyon — açıklık farkı yüksek sabit sıra (L≈80/50/35/62)
CFG_RENK = ["#A9CCE3", "#1380A1", "#7B2D8B", "#27AE60"]
GREEN = "#1E8E5A"; RED = "#C0392B"; GRAY = "#5D6D7E"
INK = "#2C3E50"; MUTED = "#7F8C8D"

plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#BDC3C7", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": "#ECF0F1", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "savefig.dpi": 200, "figure.facecolor": "white",
})

def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

ITER = {
    "H1":  "iteratif_loop_coherent_20260604_0320.json",
    "C2":  "iteratif_loop_coherent2_Dongu1_EN_20260604_0942.json",
    "C3":  "iteratif_loop_coherent3_Dongu1_TR_20260604_2237.json",
    "C4":  "iteratif_loop_coherent4_Dongu2_EN_20260605_0726.json",
}
MCN = {
    "H1": "mcnemar_coherent.json", "C2": "mcnemar_coh2.json",
    "C3": "mcnemar_coh3.json",     "C4": "mcnemar_coh4.json",
}
KONFIG = ["H1", "C2", "C3", "C4"]
D = {k: jload(ROOT / "results" / ITER[k]) for k in KONFIG}
M = {k: jload(ROOT / "results" / MCN[k]) for k in KONFIG}
BASE = D["H1"]["cnn"]["baseline"]["sinif_f1"]

HEDEF = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
KONTROL = "diger_hayvanlar"
SINIF = HEDEF + [KONTROL]

L = {
 "en": dict(
    cfg=["H1 (pretrained, EN)", "C2 (Cycle 1, EN)", "C3 (Cycle 1, TR)", "C4 (Cycle 2, EN)"],
    siniflar=["dog", "bird", "weather", "other animals\n(control)"],
    hedef_bant="AUGMENTED TARGET CLASSES", kontrol_bant="CONTROL",
    y_delta="Class F1 change (augmented − baseline)",
    t6="Targeted augmentation improves the selected classes —\nreplicated across four generation configurations",
    leg_hedef="Targets mean (dog + bird + weather)", leg_kontrol="Control (other animals)",
    y_mean="Mean F1 change",
    t7="Target classes improve in every configuration;\nthe heterogeneous control does not (same training runs)",
    t8a="Combined targets (dog+bird+weather) — significant in all four runs",
    t8b="Per-class significance (−log$_{10}$ p)",
    y_chi="McNemar $\\chi^2$ (Yates, df=1)",
 ),
 "tr": dict(
    cfg=["H1 (ön-eğitimli, EN)", "C2 (Döngü 1, EN)", "C3 (Döngü 1, TR)", "C4 (Döngü 2, EN)"],
    siniflar=["köpek", "kuş", "hava", "diğer hayvanlar\n(kontrol)"],
    hedef_bant="ARTIRILAN HEDEF SINIFLAR", kontrol_bant="KONTROL",
    y_delta="Sınıf F1 değişimi (artırılmış − taban)",
    t6="Hedefli artırma seçilen sınıfları iyileştiriyor —\ndört üretim konfigürasyonunda tekrarlanmıştır",
    leg_hedef="Hedef-3 ortalaması (köpek + kuş + hava)", leg_kontrol="Kontrol (diğer hayvanlar)",
    y_mean="Ortalama F1 değişimi",
    t7="Hedef sınıflar her konfigürasyonda iyileşiyor;\nheterojen kontrol iyileşmiyor (aynı eğitim koşuları)",
    t8a="Birleşik hedefler (köpek+kuş+hava) — dört koşuda da anlamlı",
    t8b="Sınıf bazlı anlamlılık (−log$_{10}$ p)",
    y_chi="McNemar $\\chi^2$ (Yates, sd=1)",
 ),
}


def fig6_targets(dil):
    t = L[dil]
    fig, ax = plt.subplots(figsize=(10.5, 6.36))
    x = np.arange(len(SINIF)); w = 0.2
    for i, k in enumerate(KONFIG):
        aug = D[k]["cnn"]["augmented"]["sinif_f1"]
        dv = [aug[s] - BASE[s] for s in SINIF]
        ax.bar(x + (i - 1.5) * w, dv, w * 0.9, label=t["cfg"][i],
               color=CFG_RENK[i], edgecolor="white", linewidth=1)
    ax.axhline(0, color=INK, lw=0.8)
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi * 1.25)
    ax.axvspan(-0.5, 2.5, color=GREEN, alpha=0.05)
    ax.axvspan(2.5, 3.5, color=RED, alpha=0.05)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, t["hedef_bant"], ha="center", va="top",
            fontsize=9, color=GREEN, fontweight="bold")
    ax.text(3.0, ax.get_ylim()[1] * 0.95, t["kontrol_bant"], ha="center", va="top",
            fontsize=9, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(t["siniflar"], fontsize=10)
    ax.set_ylabel(t["y_delta"], fontsize=9.5)
    ax.set_title(t["t6"], fontsize=11, fontweight="bold", color=INK)
    ax.legend(fontsize=8.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.24),
              frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig6_targets_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig7_mean(dil):
    t = L[dil]
    hed, kon = [], []
    for k in KONFIG:
        aug = D[k]["cnn"]["augmented"]["sinif_f1"]
        hed.append(np.mean([aug[s] - BASE[s] for s in HEDEF]))
        kon.append(aug[KONTROL] - BASE[KONTROL])
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(4); w = 0.36
    b1 = ax.bar(x - w / 2, hed, w, label=t["leg_hedef"], color=GREEN, edgecolor="white")
    b2 = ax.bar(x + w / 2, kon, w, label=t["leg_kontrol"], color=RED, edgecolor="white")
    for bar, v in list(zip(b1, hed)) + list(zip(b2, kon)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.0012 if v >= 0 else -0.0012),
                f"{v:+.3f}", ha="center", fontsize=8.5, fontweight="bold",
                color=GREEN if v >= 0 else RED, va="bottom" if v >= 0 else "top")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(t["cfg"], fontsize=8.5)
    ax.set_ylabel(t["y_mean"], fontsize=9.5)
    ax.set_title(t["t7"], fontsize=10.5, fontweight="bold", color=INK)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fig7_mean_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def yildiz(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


def fig8_mcnemar(dil):
    t = L[dil]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.7),
                                 gridspec_kw={"width_ratios": [1.1, 1]})
    chi = [M[k]["testler"]["tutarli_3"]["chi2"] for k in KONFIG]
    ps = [M[k]["testler"]["tutarli_3"]["p"] for k in KONFIG]
    b = a1.bar(range(4), chi, color=CFG_RENK, edgecolor="white", width=0.62)
    for bar, c, p in zip(b, chi, ps):
        a1.text(bar.get_x() + bar.get_width() / 2, c + 1.5,
                f"{yildiz(p)}\np={p:.1e}", ha="center", fontsize=8.5,
                fontweight="bold", color=RED)
    a1.axhline(10.83, color=GRAY, ls="--", lw=1)
    a1.text(3.45, 11.5, "p=0.001", fontsize=7.5, color=GRAY, ha="right")
    a1.set_xticks(range(4)); a1.set_xticklabels(KONFIG, fontsize=10)
    a1.set_ylabel(t["y_chi"], fontsize=9.5)
    a1.set_title(t["t8a"], fontsize=10)
    a1.set_ylim(0, max(chi) * 1.25)
    mat = np.zeros((len(SINIF), 4)); ann = []
    for j, k in enumerate(KONFIG):
        sat = []
        for i, s in enumerate(SINIF):
            c = M[k]["testler"]["sinif_basina"][s]
            mat[i, j] = min(-np.log10(max(c["p"], 1e-16)), 16)
            sat.append(c["anlam"])
        ann.append(sat)
    a2.grid(False)
    im = a2.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=6)
    a2.set_xticks(range(4)); a2.set_xticklabels(KONFIG, fontsize=10)
    a2.set_yticks(range(len(SINIF)))
    a2.set_yticklabels(t["siniflar"], fontsize=9)
    for i in range(len(SINIF)):
        for j in range(4):
            a2.text(j, i, ann[j][i], ha="center", va="center", fontsize=9.5,
                    color="white" if mat[i, j] > 3 else "black", fontweight="bold")
    a2.set_title(t["t8b"], fontsize=10)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.set_label("−log$_{10}$ p", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"fig8_mcnemar_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ── Şekil 1: pipeline şeması ──────────────────────────────────────────────────
S = {
 "en": [
   ("#F5F2E9", "#8A8265", "BBC Sound Effects Archive",
    "15,356 WAV · 26 classes · ~22× class imbalance"),
   ("#F5F2E9", "#8A8265", "Preprocessing and splitting",
    "16 kHz · 5 s non-overlapping segments · log-mel 128×128\n331,785 segments · file-level split (seed=42): 10,748/2,304/2,304"),
   ("KOL", "A) Classification baseline", "B) Generative augmentation pipeline"),
   ("#EBF3FB", "#1F5FA8", "Five model families",
    "LightGBM (Optuna) · BiLSTM+Attention\nAST · CNN/ResNet-18 · Late fusion"),
   ("#E9F7EF", "#1E8E5A", "CLAP domain fine-tuning",
    "HTSAT-base + RoBERTa-base · NT-Xent\nCycle 1 (BBC) → Cycle 2 (BBC+synthetic)"),
   ("#EBF3FB", "#1F5FA8", "Best baseline",
    "CNN/ResNet-18 · macro-F1 = 0.6716"),
   ("#E9F7EF", "#1E8E5A", "AudioLDM2-full generation",
    "200 DDIM steps · guidance 3.5 · TR/EN\n50 clips/class · dog, bird, weather + control"),
   ("#EFEDFA", "#5B4FBE", "Augmented training",
    "Synthetic clips added to the training set\nIdentical protocol · deterministic generation"),
   ("#EFEDFA", "#5B4FBE", "Evaluation",
    "Deterministic FAD (VGGish)\nYates–McNemar tests (Holm–Bonferroni)"),
   ("#FDF0EC", "#B03A2E", "Main finding: targeted augmentation works",
    "Significant, reproducible gains on the augmented target classes\nin all four configurations; none on the heterogeneous control"),
 ],
 "tr": [
   ("#F5F2E9", "#8A8265", "BBC Ses Efektleri Arşivi",
    "15.356 WAV · 26 sınıf · ~22× sınıf dengesizliği"),
   ("#F5F2E9", "#8A8265", "Ön işleme ve bölme",
    "16 kHz · 5 sn örtüşmesiz segment · log-mel 128×128\n331.785 segment · dosya düzeyi bölme (seed=42): 10.748/2.304/2.304"),
   ("KOL", "A) Sınıflandırma temeli", "B) Üretici artırma hattı"),
   ("#EBF3FB", "#1F5FA8", "Beş model ailesi",
    "LightGBM (Optuna) · BiLSTM+Dikkat\nAST · CNN/ResNet-18 · Geç birleşim"),
   ("#E9F7EF", "#1E8E5A", "CLAP alan ince ayarı",
    "HTSAT-base + RoBERTa-base · NT-Xent\nDöngü 1 (BBC) → Döngü 2 (BBC+sentetik)"),
   ("#EBF3FB", "#1F5FA8", "En iyi temel model",
    "CNN/ResNet-18 · makro-F1 = 0,6716"),
   ("#E9F7EF", "#1E8E5A", "AudioLDM2-full üretimi",
    "200 DDIM adımı · guidance 3,5 · TR/EN\n50 klip/sınıf · köpek, kuş, hava + kontrol"),
   ("#EFEDFA", "#5B4FBE", "Artırılmış eğitim",
    "Sentetik klipler eğitim kümesine eklenir\nAynı protokol · deterministik üretim"),
   ("#EFEDFA", "#5B4FBE", "Değerlendirme",
    "Deterministik FAD (VGGish)\nYates–McNemar testleri (Holm–Bonferroni)"),
   ("#FDF0EC", "#B03A2E", "Ana bulgu: hedefli artırma işe yarıyor",
    "Artırılan hedef sınıflarda dört konfigürasyonda da anlamlı,\ntekrarlanabilir kazanım; heterojen kontrolde yok"),
 ],
}


def kutu(ax, cx, cy, w, h, yuz, kenar, baslik, metin):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.012", facecolor=yuz, edgecolor=kenar, lw=1.4))
    ax.text(cx, cy + h * 0.22, baslik, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=kenar)
    ax.text(cx, cy - h * 0.16, metin, ha="center", va="center",
            fontsize=8.2, color=INK, linespacing=1.5)


def ok(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color="#5D6D7E", lw=1.3))


def fig1_pipeline(dil):
    s = S[dil]
    fig, ax = plt.subplots(figsize=(7.4, 10.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    # üst iki kutu
    kutu(ax, 0.5, 0.945, 0.62, 0.075, *s[0])
    ok(ax, 0.5, 0.906, 0.5, 0.886)
    kutu(ax, 0.5, 0.833, 0.82, 0.09, *s[1])
    # kol başlıkları
    ax.text(0.27, 0.762, s[2][1], ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(0.73, 0.762, s[2][2], ha="center", fontsize=10, fontweight="bold", color=INK)
    ok(ax, 0.40, 0.787, 0.27, 0.752)
    ok(ax, 0.60, 0.787, 0.73, 0.752)
    # sol kol
    kutu(ax, 0.26, 0.685, 0.42, 0.085, *s[3])
    ok(ax, 0.26, 0.641, 0.26, 0.616)
    kutu(ax, 0.26, 0.560, 0.42, 0.08, *s[5])
    # sağ kol
    kutu(ax, 0.74, 0.685, 0.44, 0.085, *s[4])
    ok(ax, 0.74, 0.641, 0.74, 0.616)
    kutu(ax, 0.74, 0.560, 0.44, 0.08, *s[6])
    # birleşme
    ok(ax, 0.26, 0.518, 0.44, 0.462)
    ok(ax, 0.74, 0.518, 0.56, 0.462)
    kutu(ax, 0.5, 0.405, 0.66, 0.085, *s[7])
    ok(ax, 0.5, 0.361, 0.5, 0.336)
    kutu(ax, 0.5, 0.280, 0.66, 0.085, *s[8])
    ok(ax, 0.5, 0.236, 0.5, 0.211)
    kutu(ax, 0.5, 0.145, 0.80, 0.10, *s[9])
    fig.savefig(OUT / f"fig1_pipeline_{dil}.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for dil in ("en", "tr"):
        fig6_targets(dil); fig7_mean(dil); fig8_mcnemar(dil); fig1_pipeline(dil)
    # McNemar tablo verisini docx tablosu icin dok
    tablo = {}
    for k in KONFIG:
        tt = M[k]["testler"]["tutarli_3"]
        tablo[k] = {"chi2": round(tt["chi2"], 1), "p": tt["p"], "anlam": tt["anlam"], "n": tt["n"]}
    ktr = M["C4"]["testler"]["sinif_basina"][KONTROL]
    tablo["kontrol_C4"] = {"chi2": round(ktr["chi2"], 2), "p": round(ktr["p"], 3),
                           "anlam": ktr["anlam"], "n": ktr["n"]}
    print(json.dumps(tablo, indent=1))
    print("PNG'ler:", sorted(p.name for p in OUT.glob("*.png")))
