"""
Makale 1 ve Makale 2 icin yayın kaliteli figürleri üretir.
Mevcut rapordan dönüştürülen + yeni Makale 2 figürleri.

Çikti: outputs/figures_makale/
  m1_fig1_model_karsilastirma.pdf/png   — Makale 1: tüm model ailesi
  m1_fig2_cnn_sinif_f1.pdf/png         — Makale 1: 26 sınıf CNN F1
  m1_fig3_augmentasyon_etkisi.pdf/png  — Makale 1: Tur1-3 sınıf F1
  m2_fig1_fad_turlar.pdf/png           — Makale 2: FAD 4 tur
  m2_fig2_tur4_delta.pdf/png           — Makale 2: Tur4 sınıf F1 delta (YENİ)
  m2_fig3_fad_f1_scatter.pdf/png       — Makale 2: FAD–ΔF1 paradoks scatter (YENİ)

Kullanim:
    cd f:/Drthesiscode
    python scripts/12_makale_figurleri.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "outputs" / "figures_makale"
OUT.mkdir(parents=True, exist_ok=True)

# ── Renk paleti (makalelerdeki BBC renk şemasıyla uyumlu) ─────────────────────
BBC_BLUE   = "#1380A1"
BBC_RED    = "#C0392B"
BBC_GRAY   = "#4E4E4E"
BBC_ORANGE = "#E8891D"
FLY_GREEN  = "#27AE60"
LIGHT_GRAY = "#F2F2F2"
GOLD       = "#F0B429"
PURPLE     = "#7B2D8B"
TEAL       = "#006D77"

def savefig(fig, name):
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"  Kaydedildi: {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# M1-FIG1: Tüm model ailesi karşılaştırması
# (genel_karsilastirma.png'nin başlıksız, publication-ready versiyonu)
# ══════════════════════════════════════════════════════════════════════════════
def m1_fig1_model_karsilastirma():
    models = [
        ("k-NN",           0.296, "Klasik ML"),
        ("SVM (RBF)",      0.380, "Klasik ML"),
        ("Extra Trees",    0.452, "Klasik ML"),
        ("XGBoost",        0.537, "Klasik ML"),
        ("HistGB",         0.548, "Klasik ML"),
        ("LightGBM\n(Optuna)",0.586,"LightGBM"),
        ("BiLSTM",         0.512, "Derin Öğrenme"),
        ("AST\n(9 ep.)",   0.616, "Derin Öğrenme"),
        ("CNN\n(ResNet-18)",0.672,"Derin Öğrenme"),
        ("Geç Birleşim\n(Val-F1)",0.666,"Geç Birleşim"),
    ]
    labels  = [m[0] for m in models]
    scores  = [m[1] for m in models]
    groups  = [m[2] for m in models]

    color_map = {
        "Klasik ML":        BBC_GRAY,
        "LightGBM":         BBC_ORANGE,
        "Derin Öğrenme":    BBC_BLUE,
        "Geç Birleşim":     PURPLE,
    }
    colors_list = [color_map[g] for g in groups]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(range(len(labels)), scores, color=colors_list,
                  edgecolor="white", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0.5, color=BBC_RED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test Makro-F1 Skoru", fontsize=9)
    ax.set_ylim(0, 0.80)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=BBC_GRAY,   label="Klasik ML"),
        mpatches.Patch(color=BBC_ORANGE, label="LightGBM (Optuna HPO)"),
        mpatches.Patch(color=BBC_BLUE,   label="Derin Öğrenme"),
        mpatches.Patch(color=PURPLE,     label="Geç Birleşim"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    savefig(fig, "m1_fig1_model_karsilastirma")


# ══════════════════════════════════════════════════════════════════════════════
# M1-FIG2: CNN (ResNet-18) 26-sınıf F1 — yatay bar
# ══════════════════════════════════════════════════════════════════════════════
def m1_fig2_cnn_sinif_f1():
    # Değerler: table2_class_report.csv / cnn_20260427 sinif_raporu (otoriter)
    siniflar = [
        ("Ayak sesleri",        0.9475),
        ("At sesleri",          0.8675),
        ("Saat/çan ses.",       0.8380),
        ("Çocuk/bebek ses.",    0.8351),
        ("Kapı sesleri",        0.8038),
        ("Uçak sesleri",        0.7693),
        ("Kara taşıtları",      0.7621),
        ("Gemi/tekne ses.",     0.7612),
        ("Tren sesleri",        0.7411),
        ("Ev içi sesler",       0.7328),
        ("Endüstri/fabrika",    0.6730),
        ("Köpek sesleri",       0.6652),
        ("Spor/eğlence",        0.6628),
        ("Ofis/iletişim",       0.6531),
        ("Kalabalık ses.",      0.6434),
        ("Silah/patlama",       0.6432),
        ("Şehir atmosferi",     0.6279),
        ("Kuş sesleri",         0.5913),
        ("Acil servis★",   0.5894),  # ★ = hedef sınıf
        ("Doğa atmosferi",      0.5871),
        ("Diğer hayvanlar★",0.5823),
        ("Su sesleri",          0.5567),
        ("Hava olayları",       0.5534),
        ("Tarım/çiftlik",       0.5249),
        ("Dini mekân★",    0.4614),
        ("İnşaat ses.★",   0.3877),
    ]
    # Düşükten yükseğe sırala — barh'ta y=0 altta, son index üstte olur
    # böylece en yüksek F1 en üstte görünür
    siniflar_sorted = sorted(siniflar, key=lambda x: x[1], reverse=False)
    labels = [s[0] for s in siniflar_sorted]
    scores = [s[1] for s in siniflar_sorted]

    def bar_color(score, label):
        if "★" in label:
            return BBC_RED if score < 0.50 else BBC_ORANGE
        return FLY_GREEN if score >= 0.75 else (BBC_ORANGE if score >= 0.50 else BBC_RED)

    bar_colors = [bar_color(s, l) for s, l in zip(scores, labels)]

    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=bar_colors, edgecolor="white", linewidth=0.4)

    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.005, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=7.5)

    ax.axvline(np.mean(scores), color=BBC_BLUE, linestyle="--", linewidth=1.0,
               alpha=0.8, label=f"Makro-F1 ort. ({np.mean(scores):.3f})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("F1-Skoru (Test Kümesi)", fontsize=9)
    ax.set_xlim(0, 1.07)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=FLY_GREEN,  label="F1 $\\geq$ 0,75"),
        mpatches.Patch(color=BBC_ORANGE, label="0,50 $\\leq$ F1 < 0,75"),
        mpatches.Patch(color=BBC_RED,    label="F1 < 0,50 (★ hedef sınıf)"),
        plt.Line2D([0], [0], color=BBC_BLUE, linestyle="--",
                   label=f"Makro-F1 ort. = {np.mean(scores):.4f}"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, "m1_fig2_cnn_sinif_f1")


# ══════════════════════════════════════════════════════════════════════════════
# M1-FIG3: Tur 1–3 — sınıf bazlı augmentasyon etkisi (CNN + LGBM)
# ══════════════════════════════════════════════════════════════════════════════
def m1_fig3_augmentasyon_etkisi():
    siniflar = ["Acil servis", "Diğer hayvanlar", "Dini mekân", "İnşaat ses."]

    # CNN F1 değerleri: [Baseline, T1_Aug, T2_Aug, T3_Aug]
    cnn_data = {
        "Acil servis":      [0.584, 0.580, 0.604, 0.584],
        "Diğer hayvanlar":  [0.582, 0.551, 0.560, 0.553],
        "Dini mekân":       [0.454, 0.466, 0.418, 0.430],
        "İnşaat ses.":      [0.384, 0.415, 0.390, 0.393],
    }
    # LGBM F1 değerleri: [Baseline, T1_Aug, T2_Aug, T3_Aug]
    # Kaynak: iteratif_loop_20260506_0348 (T1), _tur2_0316 (T2), _tur3_0852 (T3)
    lgbm_data = {
        "Acil servis":      [0.368, 0.405, 0.407, 0.401],
        "Diğer hayvanlar":  [0.501, 0.505, 0.482, 0.493],
        "Dini mekân":       [0.317, 0.303, 0.314, 0.315],
        "İnşaat ses.":      [0.346, 0.373, 0.352, 0.359],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    x = np.arange(len(siniflar))
    w = 0.18
    cnn_colors = ["#D6EAF8", "#1A5276", "#2471A3", "#117A65"]
    labels_cnn = ["Baseline", "T1 Aug (pretrained)", "T2 Aug (FT+EN)", "T3 Aug (FT+TR)"]

    for i, (clr, lbl) in enumerate(zip(cnn_colors, labels_cnn)):
        vals = [cnn_data[s][i] for s in siniflar]
        ax1.bar(x + (i - 1.5) * w, vals, w * 0.9, color=clr, label=lbl,
                edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(siniflar, fontsize=9)
    ax1.set_ylabel("Sınıf F1-Skoru", fontsize=9)
    ax1.set_ylim(0, 0.75)
    ax1.set_title("CNN (ResNet-18)", fontsize=10, pad=4)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # LGBM: aynı 4 bar düzeni (Baseline + T1 + T2 + T3)
    lgbm_colors = ["#FAD7A0", "#E59866", "#A04000", "#5D4037"]
    labels_lgbm = ["Baseline", "T1 Aug (pretrained)", "T2 Aug (FT+EN)", "T3 Aug (FT+TR)"]
    for i, (clr, lbl) in enumerate(zip(lgbm_colors, labels_lgbm)):
        vals = [lgbm_data[s][i] for s in siniflar]
        ax2.bar(x + (i - 1.5) * w, vals, w * 0.9, color=clr, label=lbl,
                edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(siniflar, fontsize=9)
    ax2.set_ylabel("Sınıf F1-Skoru", fontsize=9)
    ax2.set_ylim(0, 0.75)
    ax2.set_title("LightGBM", fontsize=10, pad=4)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    savefig(fig, "m1_fig3_augmentasyon_etkisi")


# ══════════════════════════════════════════════════════════════════════════════
# M2-FIG1: FAD 4 üretim turu karşılaştırması (sınıf bazlı + ortalama)
# ══════════════════════════════════════════════════════════════════════════════
def m2_fig1_fad_turlar():
    siniflar  = ["İnşaat ses.", "Dini mekân", "Acil servis", "Diğer hayvanlar"]
    tur_data  = {
        "Tur 1\n(Pretrained)": [21.76, 27.48, 43.38, 14.41],
        "Tur 2\n(Döngü1+EN)":  [19.35, 25.41, 26.77, 15.52],
        "Tur 3\n(Döngü1+TR)":  [26.88, 22.54, 23.82, 15.05],
        "Tur 4\n(Döngü2+EN)":  [21.88, 27.48, 43.70, 15.12],
    }
    tur_means = {
        "Tur 1\n(Pretrained)": 26.76,
        "Tur 2\n(Döngü1+EN)":  21.76,
        "Tur 3\n(Döngü1+TR)":  22.08,
        "Tur 4\n(Döngü2+EN)":  27.05,
    }
    tur_colors = ["#AED6F1", "#1A5276", "#4A235A", "#117A65"]
    tur_labels = list(tur_data.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [2.5, 1]})

    x = np.arange(len(siniflar))
    w = 0.20
    for i, (label, clr) in enumerate(zip(tur_labels, tur_colors)):
        vals = tur_data[label]
        bars = ax1.bar(x + (i - 1.5) * w, vals, w * 0.9,
                       color=clr, label=label, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(siniflar, fontsize=9)
    ax1.set_ylabel("FAD Skoru (↓ daha iyi)", fontsize=9)
    ax1.set_title("Sınıf Bazlı FAD — 4 Üretim Turu", fontsize=10, pad=4)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Sağ: ortalama FAD
    tur_short = ["Tur 1", "Tur 2", "Tur 3", "Tur 4"]
    means     = [tur_means[t] for t in tur_labels]
    bars2 = ax2.bar(range(4), means, color=tur_colors, edgecolor="white", linewidth=0.5)
    for bar, m, delta in zip(bars2, means,
                             [None, -5.00, -4.68, +0.29]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if delta is not None:
            clr_d = FLY_GREEN if delta < 0 else BBC_RED
            sign  = "+" if delta > 0 else ""
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{sign}{delta:.2f}\nvs Tur1", ha="center", fontsize=7.5,
                     color=clr_d, fontweight="bold")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(tur_short, fontsize=9)
    ax2.set_ylabel("Ortalama FAD", fontsize=9)
    ax2.set_title("Ortalama FAD — 4 Tur", fontsize=10, pad=4)
    ax2.set_ylim(0, 36)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    savefig(fig, "m2_fig1_fad_turlar")


# ══════════════════════════════════════════════════════════════════════════════
# M2-FIG2: Tur 4 sınıf F1 delta (baseline → augmented) — YENİ
# ══════════════════════════════════════════════════════════════════════════════
def m2_fig2_tur4_delta():
    siniflar = ["Acil\nservis", "Diğer\nhayvanlar", "Dini\nmekân", "İnşaat\nses."]

    cnn_base = [0.6078, 0.5833, 0.3887, 0.3721]
    cnn_aug  = [0.5711, 0.5399, 0.4682, 0.4430]
    cnn_delta= [a - b for a, b in zip(cnn_aug, cnn_base)]

    lgbm_base = [0.3679, 0.5008, 0.3174, 0.3461]
    lgbm_aug  = [0.3729, 0.4972, 0.3286, 0.3518]
    lgbm_delta= [a - b for a, b in zip(lgbm_aug, lgbm_base)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, deltas, base_vals, aug_vals, title, clr_pos, clr_neg, bar_w in [
        (axes[0], cnn_delta,  cnn_base,  cnn_aug,  "CNN (ResNet-18)",  BBC_BLUE,  BBC_RED, 0.32),
        (axes[1], lgbm_delta, lgbm_base, lgbm_aug, "LightGBM",        BBC_ORANGE,BBC_RED, 0.32),
    ]:
        x = np.arange(len(siniflar))
        # Baseline + Augmented grouped bars
        b1 = ax.bar(x - bar_w/2, base_vals, bar_w, label="Baseline",
                    color=LIGHT_GRAY, edgecolor=BBC_GRAY, linewidth=0.8)
        b2 = ax.bar(x + bar_w/2, aug_vals,  bar_w, label="Tur 4 Aug.",
                    color=[clr_pos if d >= 0 else clr_neg for d in deltas],
                    edgecolor="white", linewidth=0.5)

        # Delta annotation
        for i, (bar, delta) in enumerate(zip(b2, deltas)):
            sign  = "+" if delta >= 0 else ""
            col   = clr_pos if delta >= 0 else BBC_RED
            ypos  = max(base_vals[i], aug_vals[i]) + 0.012
            ax.text(bar.get_x() + bar.get_width()/2 - bar_w/2,
                    ypos, f"{sign}{delta:.3f}",
                    ha="center", fontsize=9, color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(siniflar, fontsize=10)
        ax.set_ylabel("Sınıf F1-Skoru", fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_ylim(0, 0.70)
        ax.legend(fontsize=8.5, loc="upper right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    savefig(fig, "m2_fig2_tur4_delta")


# ══════════════════════════════════════════════════════════════════════════════
# M2-FIG3: FAD–ΔF1 Scatter — Paradoks Görselleştirmesi — YENİ
# ══════════════════════════════════════════════════════════════════════════════
def m2_fig3_fad_f1_scatter():
    from scipy.stats import spearmanr

    # Tur 4 FAD değerleri ve ΔF1
    # sıralama: insaat, dini, acil, diger
    siniflar = ["İnşaat ses.", "Dini mekân", "Acil servis", "Diğer hayvanlar"]
    fad      = [21.88, 27.48, 43.70, 15.12]
    cnn_df1  = [+0.0709, +0.0795, -0.0367, -0.0434]
    lgbm_df1 = [+0.0057, +0.0112, +0.0050, -0.0036]

    # Etiket offset'leri (x, y) — her sınıf için elle ayarlandı
    cnn_label_offsets  = [(+3, +10), (+3, +10), (+3, -16), (-60, -16)]
    lgbm_label_offsets = [(+3, +10), (+3, +10), (+3, -16), (-60, -16)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, df1_vals, title, model_lbl, offsets in [
        (ax1, cnn_df1,  "CNN (ResNet-18) — FAD ile $\\Delta$F1 İlişkisi", "CNN", cnn_label_offsets),
        (ax2, lgbm_df1, "LightGBM — FAD ile $\\Delta$F1 İlişkisi",        "LGBM", lgbm_label_offsets),
    ]:
        # Renk: pozitif ΔF1 = mavi (kazanım), negatif = kırmızı (kayıp)
        scatter_colors = [FLY_GREEN if d >= 0 else BBC_RED for d in df1_vals]
        ax.scatter(fad, df1_vals, s=180, c=scatter_colors,
                   edgecolors="white", linewidths=1.2, zorder=5)

        # Nokta etiketleri
        for i, (fx, fy, lbl, (ox, oy)) in enumerate(
                zip(fad, df1_vals, siniflar, offsets)):
            ax.annotate(lbl, (fx, fy),
                        xytext=(ox, oy), textcoords="offset points",
                        fontsize=8, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7))

        # Sıfır çizgisi (kazanım / kayıp sınırı)
        ax.axhline(0, color=BBC_GRAY, linestyle="-", linewidth=0.9, alpha=0.5)

        # Spearman korelasyonu
        rho, pval = spearmanr(fad, df1_vals)
        rho_txt = f"Spearman $\\rho$ = {rho:+.2f}\n$p$ = {pval:.2f}  (n = 4)"
        ax.text(0.97, 0.05, rho_txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

        # Beklenen yön etiketi (sol üst köşe, veri koordinatına bağlı değil)
        yrange = max(df1_vals) - min(df1_vals)
        ax.text(0.03, 0.93,
                "Beklenen yön:\nFAD↓ → $\\Delta$F1↑",
                transform=ax.transAxes, fontsize=8, color="#888888",
                style="italic", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9F9F9",
                          edgecolor="#CCCCCC", alpha=0.8))

        ax.set_xlabel("FAD Skoru (Tur 4,  $\\downarrow$ daha iyi)", fontsize=9)
        ax.set_ylabel(f"$\\Delta$F1 ({model_lbl}: augmented $-$ baseline)", fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Renk legend
        legend_handles = [
            mpatches.Patch(color=FLY_GREEN, label="$\\Delta$F1 > 0 (kazanım)"),
            mpatches.Patch(color=BBC_RED,   label="$\\Delta$F1 < 0 (kayıp)"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper right",
                  framealpha=0.85)

    fig.tight_layout(pad=2.5)
    savefig(fig, "m2_fig3_fad_f1_scatter")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Makale figürleri üretiliyor...")
    print("\n[M1] Fig 1 — Model karşılaştırması")
    m1_fig1_model_karsilastirma()
    print("[M1] Fig 2 — CNN 26-sınıf F1")
    m1_fig2_cnn_sinif_f1()
    print("[M1] Fig 3 — Tur 1–3 augmentasyon etkisi")
    m1_fig3_augmentasyon_etkisi()
    print("\n[M2] Fig 1 — FAD 4 üretim turu")
    m2_fig1_fad_turlar()
    print("[M2] Fig 2 — Tur 4 sınıf F1 delta (YENİ)")
    m2_fig2_tur4_delta()
    print("[M2] Fig 3 -- FAD-dF1 paradoks scatter (YENI)")
    m2_fig3_fad_f1_scatter()
    print(f"\nTamamlandı. Çıktı: f:/Drthesiscode/outputs/figures_makale/")
