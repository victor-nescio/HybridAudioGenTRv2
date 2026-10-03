# -*- coding: utf-8 -*-
"""fnt_fdc_son için TÜM figürleri büyük-font, tam-genişlik, tutarlı stille yeniden üretir.
Tüm figürler docx'te ~17 cm bant genişliğine yerleşecek şekilde tasarlanır → yazılar
IEEE gövde puntosuna yakın okunur. Çıktı: yeni_media/*_big.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "makale_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
SP = ROOT / "results"

CFG_RENK = ["#A9CCE3", "#1380A1", "#7B2D8B", "#27AE60"]
GREEN = "#1E8E5A"; RED = "#C0392B"; ORANGE = "#E8891D"; GRAY = "#5D6D7E"
INK = "#2C3E50"; BLUE = "#1F5FA8"

# Tam-genişlik bant hedefi: figsize genişliği ~9.2", docx'te 17cm(6.7")'ye küçülür (~0.73x)
# → 14pt font ≈ 10pt görünür. Bu değerler o telafiyle seçildi.
plt.rcParams.update({
    "font.size": 14, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#BDC3C7", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
    "axes.grid": True, "grid.color": "#ECF0F1", "grid.linewidth": 0.9, "axes.axisbelow": True,
    "savefig.dpi": 220, "figure.facecolor": "white",
})


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


ITER = {"H1": "iteratif_loop_coherent_20260604_0320.json",
        "C2": "iteratif_loop_coherent2_Dongu1_EN_20260604_0942.json",
        "C3": "iteratif_loop_coherent3_Dongu1_TR_20260604_2237.json",
        "C4": "iteratif_loop_coherent4_Dongu2_EN_20260605_0726.json"}
MCN = {"H1": "mcnemar_coherent.json", "C2": "mcnemar_coh2.json",
       "C3": "mcnemar_coh3.json", "C4": "mcnemar_coh4.json"}
KONFIG = ["H1", "C2", "C3", "C4"]
CFG_AD = ["H1 (pretrained, EN)", "C2 (Cycle 1, EN)", "C3 (Cycle 1, TR)", "C4 (Cycle 2, EN)"]
D = {k: jload(ROOT / "results" / ITER[k]) for k in KONFIG}
M = {k: jload(ROOT / "results" / MCN[k]) for k in KONFIG}
BASE = D["H1"]["cnn"]["baseline"]["sinif_f1"]
HEDEF = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
KONTROL = "diger_hayvanlar"
SINIF = HEDEF + [KONTROL]
SINIF_ET = ["dog", "bird", "weather", "other animals\n(control)"]

AD_EN = {
 "sehir_atmosferi": "city atmosphere", "kara_tasitlari": "land vehicles",
 "endustri_fabrika": "industry/factory", "ucak_sesleri": "aircraft sounds",
 "tren_sesleri": "train sounds", "kalabalik_sesleri": "crowd sounds",
 "doga_atmosferi": "nature atmosphere", "ofis_iletisim": "office/communication",
 "gemi_tekne_sesleri": "ship/boat", "spor_eglence": "sports/entertainment",
 "silah_patlama": "weapon/explosion", "su_sesleri": "water sounds",
 "cocuk_bebek_sesleri": "children/baby", "hava_olaylari": "weather events",
 "saat_can_sesleri": "clock/bell", "at_sesleri": "horse sounds",
 "ev_ici_sesler": "indoor sounds", "diger_hayvanlar": "other animals",
 "ayak_sesleri": "footstep sounds", "dini_mekan": "religious place",
 "acil_servis": "emergency service", "kopek_sesleri": "dog sounds",
 "insaat_sesleri": "construction", "kus_sesleri": "bird sounds",
 "tarim_ciftlik": "farm/livestock", "kapi_sesleri": "door sounds",
}


def kaydet(fig, ad):
    fig.savefig(OUT / f"{ad}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", ad)


# ── Şekil 2: sınıf dağılımı (YATAY, tam genişlik) ────────────────────────────
def fig2_dagilim():
    dag = jload(SP / "sinif_dagilim_test.json")
    items = sorted(dag.items(), key=lambda x: x[1])
    adlar = [AD_EN[k] for k, _ in items]
    deger = [v for _, v in items]
    renk = ["#C0392B" if v < 500 else ("#E8891D" if v < 2000 else "#1F5FA8") for v in deger]
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    y = np.arange(len(adlar))
    ax.barh(y, deger, 0.72, color=renk, edgecolor="white")
    for yi, v in zip(y, deger):
        ax.text(v + 60, yi, f"{v:,}", va="center", fontsize=11, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(adlar, fontsize=12)
    ax.set_xlabel("Test segment count", fontsize=13)
    ax.set_xlim(0, max(deger) * 1.12)
    ax.grid(axis="y", visible=False)
    leg = [Patch(facecolor="#1F5FA8", label="≥ 2000"),
           Patch(facecolor="#E8891D", label="500–2000"),
           Patch(facecolor="#C0392B", label="< 500")]
    ax.legend(handles=leg, fontsize=12, loc="lower right", frameon=False)
    fig.tight_layout()
    kaydet(fig, "fig2_dagilim_big")


# ── Şekil 3: model karşılaştırma (tam genişlik) ──────────────────────────────
def fig3_model():
    modeller = [("k-NN", 0.296, "c"), ("SVM (RBF)", 0.380, "c"), ("Extra Trees", 0.452, "c"),
                ("XGBoost", 0.537, "c"), ("HistGB", 0.548, "c"), ("LightGBM\n(Optuna)", 0.586, "l"),
                ("BiLSTM", 0.512, "d"), ("AST", 0.616, "d"), ("CNN\n(ResNet-18)", 0.672, "d"),
                ("Late fusion", 0.666, "f")]
    renk = {"c": "#5D6D7E", "l": "#E8891D", "d": "#1F5FA8", "f": "#7B2D8B"}
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    x = np.arange(len(modeller))
    b = ax.bar(x, [m[1] for m in modeller], 0.66,
               color=[renk[m[2]] for m in modeller], edgecolor="white")
    for bar, m in zip(b, modeller):
        ax.text(bar.get_x() + bar.get_width() / 2, m[1] + 0.008, f"{m[1]:.3f}",
                ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.axhline(0.5, color="#C0392B", ls="--", lw=1.2, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in modeller], fontsize=11.5)
    ax.set_ylabel("Test macro-F1", fontsize=13)
    ax.set_ylim(0, 0.75)
    leg = [Patch(facecolor="#5D6D7E", label="Classical ML"), Patch(facecolor="#E8891D", label="LightGBM (HPO)"),
           Patch(facecolor="#1F5FA8", label="Deep learning"), Patch(facecolor="#7B2D8B", label="Late fusion")]
    ax.legend(handles=leg, fontsize=12, loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    kaydet(fig, "fig3_model_big")


# ── Şekil 4: CNN sınıf F1 (YATAY, tam genişlik) ──────────────────────────────
def fig4_classf1():
    secilen = {"kopek_sesleri", "kus_sesleri", "hava_olaylari", "diger_hayvanlar"}
    items = sorted(BASE.items(), key=lambda x: x[1])
    adlar = [AD_EN[k] + ("  ★" if k in secilen else "") for k, _ in items]
    deger = [v for _, v in items]
    renk = [GREEN if v >= 0.75 else (ORANGE if v >= 0.5 else RED) for v in deger]
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    y = np.arange(len(adlar))
    ax.barh(y, deger, 0.72, color=renk, edgecolor="white")
    for yi, v in zip(y, deger):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=10.5, color=INK)
    ax.axvline(0.6716, color=BLUE, ls="--", lw=1.4)
    ax.set_yticks(y)
    lbls = ax.set_yticklabels(adlar, fontsize=11.5)
    for lb, (k, _) in zip(lbls, items):
        if k in secilen:
            lb.set_fontweight("bold")
    ax.set_xlim(0, 1.02); ax.set_xlabel("F1 score (test set)", fontsize=13)
    ax.grid(axis="y", visible=False)
    leg = [Patch(facecolor=GREEN, label="F1 ≥ 0.75"), Patch(facecolor=ORANGE, label="0.50 ≤ F1 < 0.75"),
           Patch(facecolor=RED, label="F1 < 0.50"),
           Line2D([0], [0], marker="*", color="w", markerfacecolor=INK, markersize=13,
                  label="★ selected for augmentation"),
           Line2D([0], [0], color=BLUE, ls="--", label="macro-F1 = 0.6716")]
    ax.legend(handles=leg, fontsize=11, loc="lower right", frameon=False)
    fig.tight_layout()
    kaydet(fig, "fig4_classf1_big")


# ── Şekil 5: çeşitlilik çöküşü (tam genişlik) ────────────────────────────────
def fig5_dup():
    siniflar = ["construction", "religious\nplace", "emergency\nservice", "other\nanimals"]
    before = {"R1": [12, 7, 6, 8], "R2": [5, 4, 5, 5], "R3": [4, 5, 4, 4], "R4": [8, 5, 5, 5]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    x = np.arange(4); w = 0.2
    renk = ["#A9CCE3", "#1380A1", "#7B2D8B", "#27AE60"]
    for i, (r, vals) in enumerate(before.items()):
        a1.bar(x + (i - 1.5) * w, vals, w * 0.9, label=r, color=renk[i], edgecolor="white")
    a1.axhline(50, color="#C0392B", ls="--", lw=1.4)
    a1.text(3.4, 47, "target (50)", fontsize=11, color="#C0392B", ha="right", va="top")
    a1.set_xticks(x); a1.set_xticklabels(siniflar, fontsize=11)
    a1.set_ylabel("Unique clip count", fontsize=13); a1.set_ylim(0, 55)
    a1.set_title("Before (fixed seed): 50 files, 5–12 unique", fontsize=12.5, color=INK)
    a1.legend(fontsize=11, ncol=4, loc="upper center", frameon=False)
    for i, r in enumerate(before):
        a2.bar(x + (i - 1.5) * w, [50] * 4, w * 0.9, color=renk[i], edgecolor="white")
    a2.axhline(50, color="#C0392B", ls="--", lw=1.4)
    a2.set_xticks(x); a2.set_xticklabels(siniflar, fontsize=11)
    a2.set_title("After (fixed): 50 / 50 unique", fontsize=12.5, color=INK); a2.set_ylim(0, 55)
    fig.tight_layout()
    kaydet(fig, "fig5_dup_big")


# ── Şekil 7: hedef sınıflar 4 konfig (tam genişlik) ──────────────────────────
def fig7_targets():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    x = np.arange(len(SINIF)); w = 0.2
    for i, k in enumerate(KONFIG):
        aug = D[k]["cnn"]["augmented"]["sinif_f1"]
        dv = [aug[s] - BASE[s] for s in SINIF]
        ax.bar(x + (i - 1.5) * w, dv, w * 0.9, label=CFG_AD[i], color=CFG_RENK[i],
               edgecolor="white", linewidth=1)
    ax.axhline(0, color=INK, lw=0.9)
    y_lo, y_hi = ax.get_ylim(); ax.set_ylim(y_lo, y_hi * 1.28)
    ax.axvspan(-0.5, 2.5, color=GREEN, alpha=0.05); ax.axvspan(2.5, 3.5, color=RED, alpha=0.05)
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "AUGMENTED TARGET CLASSES", ha="center", va="top",
            fontsize=12, color=GREEN, fontweight="bold")
    ax.text(3.0, ax.get_ylim()[1] * 0.95, "CONTROL", ha="center", va="top",
            fontsize=12, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(SINIF_ET, fontsize=13)
    ax.set_ylabel("Class F1 change (aug − baseline)", fontsize=13)
    ax.legend(fontsize=11.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    kaydet(fig, "fig7_targets_big")


# ── Şekil 8: hedef ortalama vs kontrol (tam genişlik) ────────────────────────
def fig8_mean():
    hed, kon = [], []
    for k in KONFIG:
        aug = D[k]["cnn"]["augmented"]["sinif_f1"]
        hed.append(np.mean([aug[s] - BASE[s] for s in HEDEF]))
        kon.append(aug[KONTROL] - BASE[KONTROL])
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(4); w = 0.36
    b1 = ax.bar(x - w / 2, hed, w, label="Targets mean (dog + bird + weather)", color=GREEN, edgecolor="white")
    b2 = ax.bar(x + w / 2, kon, w, label="Control (other animals)", color=RED, edgecolor="white")
    for bar, v in list(zip(b1, hed)) + list(zip(b2, kon)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + (0.0014 if v >= 0 else -0.0014),
                f"{v:+.3f}", ha="center", fontsize=11.5, fontweight="bold",
                color=GREEN if v >= 0 else RED, va="bottom" if v >= 0 else "top")
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(CFG_AD, fontsize=11.5)
    ax.set_ylabel("Mean F1 change", fontsize=13)
    ax.legend(fontsize=12, loc="upper left", frameon=False)
    fig.tight_layout()
    kaydet(fig, "fig8_mean_big")


def yildiz(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))


# ── Şekil 9: McNemar + ısı haritası (tam genişlik) ───────────────────────────
def fig9_mcnemar():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.05, 1]})
    chi = [M[k]["testler"]["tutarli_3"]["chi2"] for k in KONFIG]
    ps = [M[k]["testler"]["tutarli_3"]["p"] for k in KONFIG]
    b = a1.bar(range(4), chi, color=CFG_RENK, edgecolor="white", width=0.62)
    for bar, c, p in zip(b, chi, ps):
        a1.text(bar.get_x() + bar.get_width() / 2, c + 1.8, f"{yildiz(p)}\np={p:.1e}",
                ha="center", fontsize=11.5, fontweight="bold", color=RED)
    a1.axhline(10.83, color=GRAY, ls="--", lw=1.1)
    a1.text(3.45, 12.2, "p=0.001", fontsize=10.5, color=GRAY, ha="right")
    a1.set_xticks(range(4)); a1.set_xticklabels(KONFIG, fontsize=13)
    a1.set_ylabel("McNemar $\\chi^2$ (Yates, df=1)", fontsize=13)
    a1.set_title("Combined targets — significant in all four runs", fontsize=12.5)
    a1.set_ylim(0, max(chi) * 1.28)
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
    a2.set_xticks(range(4)); a2.set_xticklabels(KONFIG, fontsize=13)
    a2.set_yticks(range(len(SINIF)))
    a2.set_yticklabels(["dog", "bird", "weather", "other animals\n(control)"], fontsize=12)
    for i in range(len(SINIF)):
        for j in range(4):
            a2.text(j, i, ann[j][i], ha="center", va="center", fontsize=12.5,
                    color="white" if mat[i, j] > 3 else "black", fontweight="bold")
    a2.set_title("Per-class significance ($-\\log_{10}$ p)", fontsize=12.5)
    cb = fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    cb.set_label("$-\\log_{10}$ p", fontsize=11); cb.ax.tick_params(labelsize=10.5)
    fig.tight_layout()
    kaydet(fig, "fig9_mcnemar_big")


# ── Şekil 1: pipeline şeması (tam genişlik, büyük font) ───────────────────────
S = [
 ("#F5F2E9", "#8A8265", "BBC Sound Effects Archive", "15,356 WAV · 26 classes · ~22× class imbalance"),
 ("#F5F2E9", "#8A8265", "Preprocessing and splitting",
  "16 kHz · 5 s non-overlapping segments · log-mel 128×128\n331,785 segments · file-level split (seed=42): 10,748 / 2,304 / 2,304"),
 ("KOL", "A) Classification baseline", "B) Generative augmentation pipeline"),
 ("#EBF3FB", "#1F5FA8", "Five model families", "LightGBM (Optuna) · BiLSTM+Attention\nAST · CNN/ResNet-18 · Late fusion"),
 ("#E9F7EF", "#1E8E5A", "CLAP domain fine-tuning", "HTSAT-base + RoBERTa-base · NT-Xent\nCycle 1 (BBC) → Cycle 2 (BBC+synthetic)"),
 ("#EBF3FB", "#1F5FA8", "Best baseline", "CNN/ResNet-18 · macro-F1 = 0.6716"),
 ("#E9F7EF", "#1E8E5A", "AudioLDM2-full generation", "200 DDIM steps · guidance 3.5 · TR/EN\n50 clips/class · dog, bird, weather + control"),
 ("#EFEDFA", "#5B4FBE", "Augmented training", "Synthetic clips added to the training set\nIdentical protocol · deterministic generation"),
 ("#EFEDFA", "#5B4FBE", "Evaluation", "Deterministic FAD (VGGish)\nYates–McNemar + cluster-robust bootstrap"),
 ("#FDF0EC", "#B03A2E", "Main finding: targeted augmentation works",
  "Significant, reproducible gains on the augmented target classes\nin all four configurations; none on the heterogeneous control"),
]


def kutu(ax, cx, cy, w, h, yuz, kenar, baslik, metin):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.012",
                 facecolor=yuz, edgecolor=kenar, lw=1.6))
    ax.text(cx, cy + h * 0.24, baslik, ha="center", va="center", fontsize=14.5,
            fontweight="bold", color=kenar)
    ax.text(cx, cy - h * 0.18, metin, ha="center", va="center", fontsize=11.5,
            color=INK, linespacing=1.5)


def ok(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                 color="#5D6D7E", lw=1.6))


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 12.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    kutu(ax, 0.5, 0.955, 0.66, 0.062, *S[0]); ok(ax, 0.5, 0.923, 0.5, 0.905)
    kutu(ax, 0.5, 0.855, 0.86, 0.075, *S[1])
    ax.text(0.26, 0.783, S[2][1], ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(0.74, 0.783, S[2][2], ha="center", fontsize=14, fontweight="bold", color=INK)
    ok(ax, 0.40, 0.807, 0.26, 0.773); ok(ax, 0.60, 0.807, 0.74, 0.773)
    kutu(ax, 0.25, 0.705, 0.44, 0.072, *S[3]); ok(ax, 0.25, 0.669, 0.25, 0.648)
    kutu(ax, 0.25, 0.585, 0.44, 0.068, *S[5])
    kutu(ax, 0.75, 0.705, 0.46, 0.072, *S[4]); ok(ax, 0.75, 0.669, 0.75, 0.648)
    kutu(ax, 0.75, 0.585, 0.46, 0.068, *S[6])
    ok(ax, 0.25, 0.551, 0.44, 0.492); ok(ax, 0.75, 0.551, 0.56, 0.492)
    kutu(ax, 0.5, 0.435, 0.68, 0.072, *S[7]); ok(ax, 0.5, 0.399, 0.5, 0.378)
    kutu(ax, 0.5, 0.315, 0.68, 0.072, *S[8]); ok(ax, 0.5, 0.279, 0.5, 0.258)
    kutu(ax, 0.5, 0.185, 0.82, 0.085, *S[9])
    kaydet(fig, "fig1_pipeline_big")


if __name__ == "__main__":
    print("Buyuk-font figurler uretiliyor...")
    fig1_pipeline(); fig2_dagilim(); fig3_model(); fig4_classf1(); fig5_dup()
    fig7_targets(); fig8_mean(); fig9_mcnemar()
    print("Tamamlandi (spektrogram ayri scriptte).")
