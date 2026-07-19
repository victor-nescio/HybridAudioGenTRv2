"""
Doktora tezi icin kapsamli PDF raporu uretir.

Kullanim:
    python scripts/11_phd_report.py

Cikti:
    outputs/report/BBC_SoundClassification_Report.pdf
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate, FrameBreak, HRFlowable, Image, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.frames import Frame
from reportlab.pdfbase import pdfmetrics as _pm
from reportlab.pdfbase.ttfonts import TTFont as _TTF
import matplotlib as _mpl, os as _os
_ttfd = _os.path.join(_os.path.dirname(_mpl.__file__), 'mpl-data', 'fonts', 'ttf')
try:
    _pm.registerFont(_TTF('DejaVuSans', _os.path.join(_ttfd, 'DejaVuSans.ttf')))
    _pm.registerFont(_TTF('DejaVuSans-Bold', _os.path.join(_ttfd, 'DejaVuSans-Bold.ttf')))
    _TR_FONT, _TR_FONT_B = 'DejaVuSans', 'DejaVuSans-Bold'
except Exception:
    _TR_FONT, _TR_FONT_B = 'Helvetica', 'Helvetica-Bold'

# ── Sabitler ──────────────────────────────────────────────────────────────────
OUT     = ROOT / "outputs" / "report"
FIG_DIR = OUT / "report_figs_new"          # YENI figurler ayri klasorde
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_YOLU = OUT / "BBC_SoundClassification_Report_new_v2.pdf"   # eski PDF korunur

W, H = A4   # 595.28 x 841.89 pt

SINIF_ISIMLERI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]

# ── Veri yukle ────────────────────────────────────────────────────────────────
def json_oku(yol):
    with open(yol, encoding="utf-8") as f:
        return json.load(f)

cnn_d  = json_oku(ROOT / "results/cnn_20260427_0312_sonuclar.json")
lstm_d = json_oku(ROOT / "results/lstm_20260427_0732_sonuclar.json")
ast_d  = json_oku(ROOT / "results/ast_20260427_1430_sonuclar.json")
ms_df  = pd.read_csv(ROOT / "outputs/model_selection_results.csv")
lgbm_m = json_oku(ROOT / "outputs/lgbm_metrics.json")

fusion_files = sorted((ROOT / "results").glob("fusion_*_sonuclar.json"))
fusion_d = json_oku(fusion_files[-1]) if fusion_files else {}

# Iteratif loop sonuclari — YENI (cesitli klip) kanonik kosumlar
# (tur2_re sadece-CNN ve eski Mayis kosumlari HARIC tutulur)
KANONIK_ITER = {
    "tur1": ROOT / "results/iteratif_loop_1_20260601_1725.json",
    "tur2": ROOT / "results/iteratif_loop_tur2_20260601_2132.json",
    "tur3": ROOT / "results/iteratif_loop_tur3_20260602_0303.json",
    "tur4": ROOT / "results/iteratif_loop_tur4_20260602_0701.json",
}
iter_d = json_oku(KANONIK_ITER["tur4"]) if KANONIK_ITER["tur4"].exists() else {}

# FAD sonuclari
fad_yol = ROOT / "results" / "fad_sonuclar.json"
fad_d = json_oku(fad_yol) if fad_yol.exists() else {}

# CLAP fine-tuning sonuclari — dongu1 (BBC-only) ve dongu2 (BBC+Tur2-sentetik)
clap_files = sorted((ROOT / "results").glob("clap_finetune_*.json"))
_d1_yol = ROOT / "results" / "clap_finetune_20260510_1712.json"
_d2_yol = ROOT / "results" / "clap_finetune_20260523_0658.json"
clap_d1 = json_oku(_d1_yol) if _d1_yol.exists() else (json_oku(clap_files[0]) if clap_files else {})
clap_d2 = json_oku(_d2_yol) if _d2_yol.exists() else (json_oku(clap_files[-1]) if clap_files else {})
clap_d  = clap_d1  # backward compat: dongu1 (ilk CLAP fine-tune)

# Sentetik klip sayilari
tur2_wav = list((ROOT / "data" / "synthetic_tur2").glob("**/*.wav")) if (ROOT / "data" / "synthetic_tur2").exists() else []
tur3_wav = list((ROOT / "data" / "synthetic_tur3").glob("**/*.wav")) if (ROOT / "data" / "synthetic_tur3").exists() else []
tur4_wav = list((ROOT / "data" / "synthetic_tur4").glob("**/*.wav")) if (ROOT / "data" / "synthetic_tur4").exists() else []

# FAD sonuclari — Tur4
fad_tur4_yol = ROOT / "results" / "fad_sonuclar_4.json"
fad_tur4_d = json_oku(fad_tur4_yol) if fad_tur4_yol.exists() else {}

# ── Matplotlib tema ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = {"cnn": "#1565C0", "lstm": "#BF360C", "ast": "#2E7D32",
     "lgbm": "#F57F17", "gray": "#607D8B"}


# ═════════════════════════════════════════════════════════════════════════════
# FIG URETIMI
# ═════════════════════════════════════════════════════════════════════════════

def fig_kaydet(fig, isim):
    p = FIG_DIR / f"{isim}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p)


def fig_sinif_dagilimi():
    rapor = cnn_d["sinif_raporu"]
    destekler = [int(rapor[str(i)]["support"]) for i in range(26)]
    renkler = [C["cnn"] if s >= 2000 else C["lgbm"] if s >= 500 else C["lstm"]
               for s in destekler]
    sira = np.argsort(destekler)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(range(26), [destekler[i] for i in sira],
            color=[renkler[i] for i in sira], height=0.7, edgecolor="white")
    ax.set_yticks(range(26))
    ax.set_yticklabels([SINIF_ISIMLERI[i] for i in sira], fontsize=7.5)
    ax.set_xlabel("Test Segmenti Sayisi")
    ax.set_title("Sinif Bazli Test Kumesi Dagilimi (48,144 segment toplam)")
    ax.axvline(np.mean(destekler), color="black", linestyle="--",
               linewidth=1, label=f"Ortalama: {np.mean(destekler):.0f}")
    ax.legend(fontsize=8)
    mavi  = mpatches.Patch(color=C["cnn"],  label=">= 2000")
    sari  = mpatches.Patch(color=C["lgbm"], label="500-2000")
    kirmizi = mpatches.Patch(color=C["lstm"], label="< 500")
    ax.legend(handles=[mavi, sari, kirmizi], title="Segment sayisi", fontsize=7)
    plt.tight_layout()
    return fig_kaydet(fig, "sinif_dagilimi")


def fig_model_secim():
    siralar = ms_df.sort_values("test_macro_f1").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    renkler_bar = [C["lgbm"] if "LightGBM" in r else C["gray"]
                   for r in siralar["ad"]]
    bars = ax.barh(range(len(siralar)), siralar["test_macro_f1"],
                   color=renkler_bar, height=0.6, edgecolor="white")
    ax.set_yticks(range(len(siralar)))
    ax.set_yticklabels(siralar["ad"], fontsize=8)
    ax.set_xlabel("Test Macro-F1")
    ax.set_xlim(0, 0.75)
    ax.set_title("Klasik ML Algoritma Secim Sonuclari (6 Algoritma, 467-dim Ozellik)")
    for bar in bars:
        h = bar.get_width()
        ax.text(h + 0.005, bar.get_y() + bar.get_height()/2,
                f"{h:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    return fig_kaydet(fig, "model_secim")


def fig_egitim_sureci():
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    modeller = [("CNN (ResNet-18)", cnn_d, C["cnn"]),
                ("BiLSTM + Attention", lstm_d, C["lstm"]),
                ("AST Transformer", ast_d, C["ast"])]

    for col, (isim, veri, renk) in enumerate(modeller):
        tarih = veri["epoch_tarih"]
        eps   = [t["epoch"] for t in tarih]

        ax_loss = axes[0, col]
        ax_loss.plot(eps, [t["train_loss"] for t in tarih], color=renk,
                     lw=2, label="Train")
        ax_loss.plot(eps, [t["val_loss"]   for t in tarih], color=renk,
                     lw=2, ls="--", alpha=0.6, label="Val")
        ax_loss.set_title(isim, fontweight="bold", color=renk, fontsize=9)
        ax_loss.set_ylabel("Cross-Entropy Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend(fontsize=7)

        ax_f1 = axes[1, col]
        vf1s = [t["val_f1"] for t in tarih]
        ax_f1.plot(eps, vf1s, color=renk, lw=2, marker="o", ms=3)
        best_i   = int(np.argmax(vf1s))
        best_e   = eps[best_i]
        best_f1  = vf1s[best_i]
        ax_f1.axvline(best_e, color="gray", ls=":", alpha=0.6)
        ax_f1.scatter([best_e], [best_f1], color="red", zorder=5, s=50,
                      label=f"Best: {best_f1:.4f} (ep.{best_e})")
        if veri.get("test_macro_f1"):
            ax_f1.axhline(veri["test_macro_f1"], color="black", ls="-.",
                          lw=1, label=f"Test: {veri['test_macro_f1']:.4f}")
        ax_f1.set_ylabel("Val Macro-F1")
        ax_f1.set_xlabel("Epoch")
        ax_f1.legend(fontsize=7, loc="lower right")

    plt.suptitle("Derin Ogrenme Modelleri Egitim Süreci", fontsize=11,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_kaydet(fig, "egitim_sureci")


def fig_genel_karsilastirma():
    modeller = [
        ("k-NN",       0.296,  "klasik"),
        ("SVM (RBF)",  0.380,  "klasik"),
        ("Extra Trees",0.452,  "klasik"),
        ("XGBoost",    0.537,  "klasik"),
        ("HistGB",     0.548,  "klasik"),
        ("LightGBM\n(Optuna)", lgbm_m["test_macro_f1"], "lgbm"),
        ("BiLSTM",     lstm_d["test_macro_f1"],          "derin"),
        ("AST\n(9 ep.)",fusion_d.get("bireysel_test", {}).get("ast", 0.6156), "derin"),
        ("CNN\n(ResNet-18)", cnn_d["test_macro_f1"],     "derin"),
        ("Fusion\n(Val-F1)", fusion_d.get("fusion_sonuclari", {}).get("val_f1_agirlik", {}).get("test", 0.666), "fusion"),
    ]
    renk_map = {"klasik": C["gray"], "lgbm": C["lgbm"],
                "derin": C["cnn"], "fusion": "#6A1B9A"}
    isimler = [m[0] for m in modeller]
    degerler = [m[1] for m in modeller]
    renkler  = [renk_map[m[2]] for m in modeller]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(modeller)), degerler, color=renkler,
                  edgecolor="white", width=0.6)
    ax.set_xticks(range(len(modeller)))
    ax.set_xticklabels(isimler, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test Macro-F1 Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Tum Model Ailelerinin Test Performansi Karsilastirmasi (26 Sinif)")
    for bar, v in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="red", ls=":", alpha=0.35, lw=1)

    legend_els = [
        mpatches.Patch(color=C["gray"],  label="Klasik ML"),
        mpatches.Patch(color=C["lgbm"],  label="LightGBM (Optuna HPO)"),
        mpatches.Patch(color=C["cnn"],   label="Derin Ogrenme"),
        mpatches.Patch(color="#6A1B9A",  label="Late Fusion"),
    ]
    ax.legend(handles=legend_els, fontsize=7, loc="upper left")
    plt.tight_layout()
    return fig_kaydet(fig, "genel_karsilastirma")


def fig_sinif_f1_heatmap():
    rapor_cnn  = cnn_d["sinif_raporu"]
    rapor_lstm = lstm_d["sinif_raporu"]
    cnn_f1s  = [rapor_cnn[str(i)]["f1-score"]  for i in range(26)]
    lstm_f1s = [rapor_lstm[str(i)]["f1-score"] for i in range(26)]
    matris = np.array([cnn_f1s, lstm_f1s])

    fig, ax = plt.subplots(figsize=(13, 2.8))
    im = ax.imshow(matris, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(26))
    ax.set_xticklabels(SINIF_ISIMLERI, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["CNN\n(ResNet-18)", "BiLSTM"], fontsize=8)
    ax.set_title("Sinif Bazli F1 Skorlari Isi Haritasi — CNN vs BiLSTM (Test Kumesi)")
    for i in range(2):
        for j in range(26):
            v = matris[i, j]
            tx = "white" if v < 0.35 or v > 0.78 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=tx)
    plt.colorbar(im, ax=ax, shrink=0.8, label="F1-Score")
    plt.tight_layout()
    return fig_kaydet(fig, "sinif_heatmap")


def fig_fusion():
    if not fusion_d:
        return None
    fs = fusion_d["fusion_sonuclari"]
    bireysel = fusion_d.get("bireysel_test", {})

    etiketler = ["BiLSTM", "AST\n(9 ep.)", "CNN\n(ResNet-18)",
                 "Esit\nAgirlik", "Val-F1\nAgirlik", "Meta-\nLearner", "Oracle\nUst Sinir"]
    degerler  = [
        bireysel.get("lstm", lstm_d["test_macro_f1"]),
        bireysel.get("ast",  0.6156),
        bireysel.get("cnn",  cnn_d["test_macro_f1"]),
        fs["esit_agirlik"]["test"],
        fs["val_f1_agirlik"]["test"],
        fs["meta_learner"]["test"],
        fs["oracle"]["test"],
    ]
    renkler = [C["lstm"], C["ast"], C["cnn"],
               "#42A5F5", "#1565C0", "#E53935", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    bars = ax.bar(range(len(etiketler)), degerler, color=renkler,
                  edgecolor="white", width=0.6)
    ax.set_xticks(range(len(etiketler)))
    ax.set_xticklabels(etiketler, fontsize=8.5)
    ax.set_ylabel("Test Macro-F1")
    ax.set_ylim(0, 0.9)
    ax.set_title("Late Fusion Yontemleri Karsilastirmasi")
    ax.axvline(2.5, color="gray", ls=":", lw=1.2, alpha=0.5)
    ax.text(1.0, 0.83, "Bireysel Modeller", ha="center", fontsize=7.5, color="gray")
    ax.text(4.5, 0.83, "Fusion Yontemleri",  ha="center", fontsize=7.5, color="gray")
    for bar, v in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    plt.tight_layout()
    return fig_kaydet(fig, "fusion")


def fig_cnn_sinif_cubuk():
    rapor = cnn_d["sinif_raporu"]
    f1s   = [rapor[str(i)]["f1-score"] for i in range(26)]
    sira  = np.argsort(f1s)
    renkler = ["#4CAF50" if f1s[i] >= 0.75 else
               "#FF9800" if f1s[i] >= 0.50 else "#F44336" for i in sira]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(range(26), [f1s[i] for i in sira], color=renkler,
            height=0.7, edgecolor="white")
    ax.set_yticks(range(26))
    ax.set_yticklabels([SINIF_ISIMLERI[i] for i in sira], fontsize=7.5)
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.05)
    ax.axvline(cnn_d["test_macro_f1"], color="navy", ls="--", lw=1.5,
               label=f"Macro-F1={cnn_d['test_macro_f1']:.4f}")
    ax.set_title("CNN (ResNet-18) Sinif Bazli F1 Skorlari (Test Kumesi)")
    for i, idx in enumerate(sira):
        ax.text(f1s[idx] + 0.01, i, f"{f1s[idx]:.3f}", va="center", fontsize=7.5)
    leg_els = [mpatches.Patch(color="#4CAF50", label=">= 0.75"),
               mpatches.Patch(color="#FF9800", label="0.50-0.75"),
               mpatches.Patch(color="#F44336", label="< 0.50")]
    ax.legend(handles=leg_els, fontsize=7, title="F1 aralik")
    plt.tight_layout()
    return fig_kaydet(fig, "cnn_sinif_cubuk")


def fig_pipeline():
    """Arastirma pipeline diyagrami — 5 asama."""
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 1); ax.axis("off")

    asamalar = [
        ("Bolum 1\nBaseline\nSiniflandirma", "#1565C0", "CNN · LSTM\nAST · LightGBM\nFusion"),
        ("Bolum 2\nTur 1\nAugmentasyon", "#2E7D32", "Pretrained\nAudioLDM2\n+ EN prompt"),
        ("Bolum 3\nTur 2\nFine-tuned Aug", "#E65100", "BBC CLAP\nFine-tune\n+ EN prompt"),
        ("Bolum 4\nTur 3\nTurkce Uretim", "#6A1B9A", "Fine-tuned\nCLAP\n+ TR prompt"),
        ("Bolum 5\nIteratif\nDongu", "#BF360C", "Self-play\nData flywheel\n(Arastirma)"),
    ]
    genislik = 1.9
    bosluk   = 0.4
    for i, (baslik, renk, icerik) in enumerate(asamalar):
        x = i * (genislik + bosluk) + 0.2
        kutu = plt.Rectangle((x, 0.12), genislik, 0.76,
                              facecolor=renk, alpha=0.15, edgecolor=renk, linewidth=2)
        ax.add_patch(kutu)
        ax.text(x + genislik/2, 0.78, baslik, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=renk,
                multialignment="center")
        ax.text(x + genislik/2, 0.38, icerik, ha="center", va="center",
                fontsize=7, color="#37474F", multialignment="center")
        if i < len(asamalar) - 1:
            ax.annotate("", xy=(x + genislik + bosluk, 0.5), xytext=(x + genislik, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#607D8B", lw=1.5))

    ax.set_title("Arastirma Mimarisi — 5 Asama", fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout()
    return fig_kaydet(fig, "pipeline")


def fig_fad_tur1():
    """FAD skorlari — Tur 1 (VGGish embedding)."""
    if not fad_d:
        return None
    siniflar = list(fad_d.keys())
    fadlar   = [fad_d[s]["fad"] for s in siniflar]
    renkler  = ["#F44336" if f > 35 else "#FF9800" if f > 20 else "#4CAF50" for f in fadlar]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(range(len(siniflar)), fadlar, color=renkler, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(siniflar)))
    ax.set_xticklabels([s.replace("_", "\n") for s in siniflar], fontsize=8)
    ax.set_ylabel("FAD Skoru (VGGish)")
    ax.set_title("Tur 1 FAD Skorlari — Gercek vs Sentetik (Dusuk = Daha Iyi)")
    for bar, v in zip(bars, fadlar):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.axhline(np.mean(fadlar), color="navy", ls="--", lw=1.2,
               label=f"Ortalama: {np.mean(fadlar):.1f}")
    ax.legend(fontsize=8)
    legend_els = [mpatches.Patch(color="#4CAF50", label="< 20 (Iyi)"),
                  mpatches.Patch(color="#FF9800", label="20-35 (Orta)"),
                  mpatches.Patch(color="#F44336", label="> 35 (Zayif)")]
    ax.legend(handles=legend_els, fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig_kaydet(fig, "fad_tur1")


def fig_clap_egitim():
    """CLAP fine-tuning egitim grafigi — loss + kNN."""
    if not clap_d or not clap_d.get("tarih_detay"):
        return None
    tarih   = clap_d["tarih_detay"]
    # Epoch 1 checkpoint'i de ekle
    ep1_val = {"epoch": 1, "train_loss": None,
                "val_kayip": float("nan"), "knn_acc": 0.0137,
                "A2T_R@1": 0.0007}
    tum_tarih = [ep1_val] + tarih

    epochlar  = [t["epoch"] for t in tum_tarih]
    val_kayip = [t["val_kayip"] for t in tum_tarih]
    knn_acc   = [t["knn_acc"]   for t in tum_tarih]
    a2t_r1    = [t["A2T_R@1"]   for t in tum_tarih]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.plot(epochlar, val_kayip, color="#E65100", lw=2, marker="o", ms=5, label="Val Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title("CLAP Fine-Tuning — Val Loss", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(epochlar, knn_acc, color="#1565C0", lw=2, marker="s", ms=5, label="kNN Acc")
    ax.plot(epochlar, a2t_r1,  color="#2E7D32", lw=2, marker="^", ms=5, label="A2T R@1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Skor")
    ax.set_title("CLAP Fine-Tuning — Geri Cagri Metrikleri", fontweight="bold")
    ax.legend(fontsize=8)

    plt.suptitle(f"CLAP BBC Fine-Tuning (5 Epoch, 26 saat, batch=16, lr=3e-5)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig_kaydet(fig, "clap_egitim")


def fig_fad_uc_tur():
    """FAD dort tur karsilastirmasi — Tur1 vs Tur2 vs Tur3 vs Tur4."""
    fad_tur2_yol = ROOT / "results" / "fad_sonuclar_tur2.json"
    fad_tur3_yol = ROOT / "results" / "fad_sonuclar_tur3.json"
    if not fad_tur2_yol.exists() or not fad_tur3_yol.exists():
        return None
    fad2 = json_oku(fad_tur2_yol)
    fad3 = json_oku(fad_tur3_yol)
    har_tur4 = fad_tur4_d and len(fad_tur4_d) >= 4

    siniflar = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]
    x = np.arange(len(siniflar))
    w = 0.19 if har_tur4 else 0.25

    fad1_vals = [fad_d[s]["fad"]  for s in siniflar]
    fad2_vals = [fad2[s]["fad"]   for s in siniflar]
    fad3_vals = [fad3[s]["fad"]   for s in siniflar]
    fad4_vals = [fad_tur4_d[s]["fad"] for s in siniflar] if har_tur4 else []

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    offset = 1.5 if har_tur4 else 1.0
    b1 = ax.bar(x - offset*w, fad1_vals, w, label="Tur 1\n(Pretrained+EN)",  color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x - offset*w/3*1, fad2_vals, w, label="Tur 2\n(Dongu1+EN)", color="#1565C0", edgecolor="white")
    b3 = ax.bar(x + offset*w/3*1, fad3_vals, w, label="Tur 3\n(Dongu1+TR)", color="#6A1B9A", edgecolor="white")
    all_bars = [b1, b2, b3]
    if har_tur4:
        b4 = ax.bar(x + offset*w, fad4_vals, w, label="Tur 4\n(Dongu2+EN)", color="#00838F", edgecolor="white")
        all_bars.append(b4)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in siniflar], fontsize=8.5)
    ax.set_ylabel("FAD Skoru (VGGish) — Dusuk = Daha Iyi")
    ax.set_title("Tur 1 / 2 / 3 / 4 FAD Karsilastirmasi (Sinif Bazli)")
    ax.legend(fontsize=7, loc="upper right")
    for bars in all_bars:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=6.5)

    ax2 = axes[1]
    ortalamalar = [np.mean(fad1_vals), np.mean(fad2_vals), np.mean(fad3_vals)]
    etiketler   = ["Tur 1\nPretrained+EN", "Tur 2\nDongu1+EN", "Tur 3\nDongu1+TR"]
    renkler     = ["#90CAF9", "#1565C0", "#6A1B9A"]
    if har_tur4:
        ortalamalar.append(np.mean(fad4_vals))
        etiketler.append("Tur 4\nDongu2+EN")
        renkler.append("#00838F")
    bars2 = ax2.bar(range(len(ortalamalar)), ortalamalar, color=renkler, edgecolor="white", width=0.5)
    ax2.set_xticks(range(len(ortalamalar))); ax2.set_xticklabels(etiketler, fontsize=8)
    ax2.set_ylabel("Ortalama FAD"); ax2.set_title("Ortalama FAD — 4 Tur Karsilastirmasi")
    for bar, v in zip(bars2, ortalamalar):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for i in range(1, len(ortalamalar)):
        d = ortalamalar[i] - ortalamalar[0]
        renk = "#2E7D32" if d < 0 else "#C62828"
        ax2.text(i, ortalamalar[i] + 2.2, f"{d:+.2f}\nvs Tur1", ha="center",
                 fontsize=7.5, color=renk, fontweight="bold")
    ax2.set_ylim(0, max(ortalamalar) * 1.4)

    plt.suptitle("FAD Analizi: 4 Uretim Turu Karsilastirmasi (VGGish Embedding)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig_kaydet(fig, "fad_uc_tur")


def fig_tur_ozet():
    """Tur 1 augmentasyon etkisi — hedef 4 sinif."""
    cnn_aug  = iter_d.get("cnn",  {})
    lgbm_aug = iter_d.get("lgbm", {})
    if not cnn_aug or not lgbm_aug:
        return None

    hedef = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]
    cnn_base = cnn_aug.get("baseline", {}).get("sinif_f1", {})
    cnn_aug2 = cnn_aug.get("augmented",{}).get("sinif_f1", {})
    lgb_base = lgbm_aug.get("baseline", {}).get("sinif_f1", {})
    lgb_aug2 = lgbm_aug.get("augmented",{}).get("sinif_f1", {})

    x = np.arange(len(hedef)); w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - 1.5*w, [cnn_base.get(s,0)  for s in hedef], w, label="CNN Baseline",  color="#90CAF9")
    b2 = ax.bar(x - 0.5*w, [cnn_aug2.get(s,0)  for s in hedef], w, label="CNN Aug (Tur1)", color="#1565C0")
    b3 = ax.bar(x + 0.5*w, [lgb_base.get(s,0)  for s in hedef], w, label="LGBM Baseline", color="#FFCC80")
    b4 = ax.bar(x + 1.5*w, [lgb_aug2.get(s,0)  for s in hedef], w, label="LGBM Aug (Tur1)",color="#E65100")

    ax.set_xticks(x); ax.set_xticklabels([s.replace("_","\n") for s in hedef], fontsize=8.5)
    ax.set_ylabel("Sinif F1-Score"); ax.set_ylim(0, 0.75)
    ax.set_title("Tur 1 Augmentasyon Etkisi — Hedef 4 Sinif (Baseline vs Augmented)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")

    # Delta oklar
    for i, s in enumerate(hedef):
        d_cnn = cnn_aug2.get(s,0) - cnn_base.get(s,0)
        d_lgb = lgb_aug2.get(s,0) - lgb_base.get(s,0)
        renk_c = "#2E7D32" if d_cnn >= 0 else "#C62828"
        renk_l = "#2E7D32" if d_lgb >= 0 else "#C62828"
        ax.text(i - 0.5*w, cnn_aug2.get(s,0) + 0.012, f"{d_cnn:+.3f}",
                ha="center", fontsize=7, color=renk_c, fontweight="bold")
        ax.text(i + 1.5*w, lgb_aug2.get(s,0) + 0.012, f"{d_lgb:+.3f}",
                ha="center", fontsize=7, color=renk_l, fontweight="bold")
    plt.tight_layout()
    return fig_kaydet(fig, "tur1_augmentasyon")


def fig_flywheel_sema():
    """
    Data flywheel dongu semasi — iki panelli:
    Sol: lineer adim akisi (1→6), Sag: 3 tur karsilastirma tablosu.
    """
    fig = plt.figure(figsize=(14, 7))
    # Sol panel: akis semasi, Sag panel: tur karsilastirma
    ax_sol = fig.add_axes([0.01, 0.05, 0.56, 0.88])
    ax_sag = fig.add_axes([0.60, 0.05, 0.38, 0.88])
    ax_sol.set_xlim(0, 10); ax_sol.set_ylim(0, 10.5); ax_sol.axis("off")
    ax_sag.axis("off")

    # ── Sol panel: adim kutulari ──────────────────────────────────────────────
    adimlar = [
        (1, "#1565C0", "#E3F2FD", "ADIM 1 — BBC Gercek Veri",
         "15,356 WAV  |  26 sinif  |  331,785 segment\nStratified %70/%15/%15 (10.748/2.304/2.304 dosya)"),
        (2, "#2E7D32", "#E8F5E9", "ADIM 2 — Baseline Siniflandirici Egitimi",
         "CNN (ResNet-18): Test Macro-F1 = 0.6716\nEn iyi 5 model ailesi karsilastirildi"),
        (3, "#6A1B9A", "#F3E5F5", "ADIM 3 — CLAP Alan Ince-Ayari",
         "BBC ses-metin ciftleri  |  TR/EN %50-%50\nDongu1 (BBC)  →  Dongu2 (BBC + sentetik)"),
        (4, "#E65100", "#FFF3E0", "ADIM 4 — AudioLDM2 ile Sentetik Uretim",
         "DDIM 200 adim | guidance 3,5 | 50 benzersiz klip/sinif\nDeterministik FAD: EN turlar ozdes (13,6-14,0); TR=15,9\nFAD ortalama uretim kalitesini DEGISTIRMEZ"),
        (5, "#00838F", "#E0F7FA", "ADIM 5 — Augmented Yeniden Egitim",
         "Sentetik + gercek veri birlestirildi (ozdes protokol)\nDeterministik egitim; her konfigurasyon icin yeniden"),
        (6, "#B71C1C", "#FFEBEE", "ADIM 6 — Degerlendirme & Ana Bulgu",
         "FAD + Yates-McNemar + VGGish tutarlilik\nTutarli siniflar (kopek/kus/hava): dort kosuda da ***\nCatch-all kontrol: hicbir kosuda fayda yok"),
    ]

    y_baslangic = 9.8
    kutu_yukseklik = 1.35
    bosluk = 0.22
    ok_rengi = "#546E7A"

    for i, (no, kenar_renk, ic_renk, baslik, detay) in enumerate(adimlar):
        y = y_baslangic - i * (kutu_yukseklik + bosluk)
        cy = y - kutu_yukseklik / 2

        # Kutu
        r = mpatches.FancyBboxPatch((0.3, cy), 9.2, kutu_yukseklik,
            boxstyle="round,pad=0.1", facecolor=ic_renk,
            edgecolor=kenar_renk, linewidth=2)
        ax_sol.add_patch(r)

        # Adim numarasi dairesi
        daire = plt.Circle((1.05, cy + kutu_yukseklik/2), 0.38,
                            color=kenar_renk, zorder=3)
        ax_sol.add_patch(daire)
        ax_sol.text(1.05, cy + kutu_yukseklik/2, str(no),
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=4)

        # Baslik
        ax_sol.text(1.75, cy + kutu_yukseklik - 0.32, baslik,
                    ha="left", va="center", fontsize=9,
                    fontweight="bold", color=kenar_renk)

        # Detay
        ax_sol.text(1.75, cy + 0.38, detay,
                    ha="left", va="center", fontsize=7.8,
                    color="#333333", linespacing=1.4)

        # Ok (son adim haric)
        if i < len(adimlar) - 1:
            ok_y = cy - 0.05
            ax_sol.annotate("", xy=(4.9, ok_y - 0.12),
                            xytext=(4.9, ok_y),
                            arrowprops=dict(arrowstyle="-|>",
                                           color=ok_rengi, lw=1.5))

    # Dongu geri ok (sol kenar, adim 6 → adim 2)
    ax_sol.annotate("", xy=(0.3, y_baslangic - 1*(kutu_yukseklik+bosluk) - 0.05),
                    xytext=(0.3, y_baslangic - 5*(kutu_yukseklik+bosluk) - kutu_yukseklik + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#B71C1C",
                                   lw=2, connectionstyle="arc3,rad=0.0"))
    ax_sol.text(0.05, y_baslangic - 3.3*(kutu_yukseklik+bosluk),
                "Dongu\n2\n(Arast.)", ha="center", va="center",
                fontsize=7.5, color="#B71C1C", fontweight="bold",
                rotation=90)

    ax_sol.set_title("Iteratif Data Flywheel — Adim Adim Akis",
                     fontsize=11, fontweight="bold", pad=6, color="#0D1B2A")

    # ── Sag panel: 4 tutarli konfigurasyon (ana bulgu) tablosu ────────────────
    ax_sag.set_title("4 Tutarli Konfigurasyon — Ana Bulgu", fontsize=11,
                     fontweight="bold", pad=6, color="#0D1B2A")

    # Hedef: kopek+kus+hava (tum konfiglerde ayni) + catch-all kontrol.
    # FAD: deterministik (item 9). McNemar: tutarli-3 birlesik (makale Tablo 2).
    satirlar = [
        ["",              "H1",         "coh2",       "coh3",       "coh4"],
        ["CLAP",          "Pretrained", "Dongu1 FT",  "Dongu1 FT",  "Dongu2 FT"],
        ["Prompt Dili",   "Ingilizce",  "Ingilizce",  "Turkce",     "Ingilizce"],
        ["Klip/sinif",    "50",         "50",         "50",         "50"],
        ["FAD ort.",      "13.64",      "13.59",      "15.90",      "14.00"],
        ["Tutarli-3 chi2","25.5",       "20.6",       "23.1",       "66.4"],
        ["McNemar p",     "4e-7***",    "6e-6***",    "2e-6***",    "3e-16***"],
        ["Catch-all",     "ns",         "ns",         "ns",         "ns"],
        ["Durum",         "TAMAM",      "TAMAM",      "TAMAM",      "TAMAM"],
    ]

    n_row = len(satirlar)
    n_col = 5
    col_w  = [1.9, 1.3, 1.3, 1.3, 1.3]
    row_h  = 0.68
    x0, y0 = 0.05, 9.5

    renk_map = {
        0: "#1565C0",   # Tur1
        1: "#6A1B9A",   # Tur2
        2: "#E65100",   # Tur3
        3: "#00838F",   # Tur4
    }
    baslik_bg  = "#0D47A1"
    satir_bg   = ["#F5F5F5", "#FFFFFF"]

    for ri, satir in enumerate(satirlar):
        yy = y0 - ri * row_h
        x_cursor = 0.0
        for ci, hucre in enumerate(satir):
            cw = col_w[ci]
            if ri == 0:
                bg = baslik_bg
                fg = "white"
                fw = "bold"
                fs = 8.0
            else:
                bg = satir_bg[ri % 2]
                fg = renk_map.get(ci - 1, "#1A1A1A") if ci > 0 else "#1A1A1A"
                fw = "bold" if ci == 0 else "normal"
                fs = 7.8
                if "▼" in hucre or "EVET" in hucre or "★" in hucre:
                    bg = "#E8F5E9"; fg = "#1B5E20"; fw = "bold"
                if "▲" in hucre:
                    bg = "#FFEBEE"; fg = "#C62828"; fw = "bold"
                if "***" in hucre:
                    bg = "#FCE4EC"; fg = "#880E4F"; fw = "bold"
                if "DEVAM" in hucre:
                    bg = "#FFF9C4"; fg = "#E65100"; fw = "bold"

            rect = mpatches.Rectangle((x_cursor, yy - row_h),
                                       cw, row_h,
                                       facecolor=bg,
                                       edgecolor="#CFD8DC", linewidth=0.8)
            ax_sag.add_patch(rect)
            ax_sag.text(x_cursor + cw/2, yy - row_h/2, hucre,
                        ha="center", va="center",
                        fontsize=fs, fontweight=fw, color=fg)
            x_cursor += cw

    ax_sag.set_xlim(0, sum(col_w))
    ax_sag.set_ylim(y0 - n_row * row_h - 0.3, y0 + 0.3)

    fig.patch.set_facecolor("white")
    return fig_kaydet(fig, "flywheel_sema")


def fig_siniflandirma_tur_karsilastirma():
    """Tur1 / Tur2 / Tur3 augmented vs baseline siniflandirma karsilastirmasi."""
    # YENI kanonik kosumlar (tur2_re / eski Mayis kosumlari HARIC)
    tur1_d = json_oku(KANONIK_ITER["tur1"]) if KANONIK_ITER["tur1"].exists() else {}
    tur2_d = json_oku(KANONIK_ITER["tur2"]) if KANONIK_ITER["tur2"].exists() else {}
    tur3_d = json_oku(KANONIK_ITER["tur3"]) if KANONIK_ITER["tur3"].exists() else {}

    hedef = ["acil_servis", "diger_hayvanlar", "dini_mekan", "insaat_sesleri"]
    kisa  = ["acil\nservis", "diger\nhayvanlar", "dini\nmekan", "insaat\nsesleri"]

    def f1_al(d, model, tip, sinif=None):
        m = d.get(model, {})
        t = m.get(tip, {})
        if sinif:
            return t.get("sinif_f1", {}).get(sinif, 0)
        return t.get("test_f1", 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_i, (model, ax, renk_set) in enumerate([
        ("cnn",  axes[0], ["#90CAF9", "#1565C0", "#81D4FA", "#0277BD", "#B2EBF2", "#006064"]),
        ("lgbm", axes[1], ["#FFCC80", "#E65100", "#FFAB40", "#BF360C", "#FFD54F", "#F57F17"]),
    ]):
        x = np.arange(len(hedef)); w = 0.13
        etik = f"CNN (ResNet-18)" if model == "cnn" else "LightGBM"

        # Tur1
        b1 = [f1_al(tur1_d, model, "baseline",  s) for s in hedef]
        b2 = [f1_al(tur1_d, model, "augmented", s) for s in hedef]
        # Tur2
        b3 = [f1_al(tur2_d, model, "baseline",  s) for s in hedef]
        b4 = [f1_al(tur2_d, model, "augmented", s) for s in hedef]
        # Tur3
        b5 = [f1_al(tur3_d, model, "baseline",  s) for s in hedef]
        b6 = [f1_al(tur3_d, model, "augmented", s) for s in hedef]

        ax.bar(x - 2.5*w, b1, w, label="Baseline",            color=renk_set[0], edgecolor="white")
        ax.bar(x - 1.5*w, b2, w, label="Tur1 Aug (pretrained+EN)", color=renk_set[1], edgecolor="white")
        ax.bar(x - 0.5*w, b3, w, label="Tur2 Base",           color=renk_set[0], edgecolor="white", alpha=0.5)
        ax.bar(x + 0.5*w, b4, w, label="Tur2 Aug (FT+EN)",    color=renk_set[3], edgecolor="white")
        ax.bar(x + 1.5*w, b5, w, label="Tur3 Base",           color=renk_set[0], edgecolor="white", alpha=0.3)
        ax.bar(x + 2.5*w, b6, w, label="Tur3 Aug (FT+TR)",    color=renk_set[5], edgecolor="white")

        ax.set_xticks(x); ax.set_xticklabels(kisa, fontsize=8.5)
        ax.set_ylabel("Sinif F1-Score"); ax.set_ylim(0, 0.80)
        ax.set_title(f"{etik} — Tur 1 / 2 / 3 Augmentasyon Etkisi", fontsize=9.5)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("Sinif Bazli F1: Baseline vs Augmented (3 Uretim Turu Karsilastirmasi)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig_kaydet(fig, "siniflandirma_tur_karsilastirma")


def fig_makro_f1_ozet():
    """Makro F1 ozet cubuk — 3 tur x 2 model x baseline/augmented."""
    # YENI kanonik kosumlar
    tur1_d = json_oku(KANONIK_ITER["tur1"]) if KANONIK_ITER["tur1"].exists() else {}
    tur2_d = json_oku(KANONIK_ITER["tur2"]) if KANONIK_ITER["tur2"].exists() else {}
    tur3_d = json_oku(KANONIK_ITER["tur3"]) if KANONIK_ITER["tur3"].exists() else {}

    def f1(d, model, tip):
        return d.get(model, {}).get(tip, {}).get("test_f1", 0)

    etik   = ["Tur1\nBaseline", "Tur1\nAug", "Tur2\nBaseline", "Tur2\nAug", "Tur3\nBaseline", "Tur3\nAug"]
    cnn_v  = [f1(tur1_d,"cnn","baseline"), f1(tur1_d,"cnn","augmented"),
               f1(tur2_d,"cnn","baseline"), f1(tur2_d,"cnn","augmented"),
               f1(tur3_d,"cnn","baseline"), f1(tur3_d,"cnn","augmented")]
    lgbm_v = [f1(tur1_d,"lgbm","baseline"), f1(tur1_d,"lgbm","augmented"),
               f1(tur2_d,"lgbm","baseline"), f1(tur2_d,"lgbm","augmented"),
               f1(tur3_d,"lgbm","baseline"), f1(tur3_d,"lgbm","augmented")]

    x = np.arange(len(etik)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    b1 = ax.bar(x - w/2, cnn_v,  w, label="CNN (ResNet-18)", color="#1565C0", edgecolor="white")
    b2 = ax.bar(x + w/2, lgbm_v, w, label="LightGBM",        color="#E65100", edgecolor="white")

    for bar in list(b1) + list(b2):
        v = bar.get_height()
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.004, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Tur ayirici cizgiler
    for xv in [1.5, 3.5]:
        ax.axvline(xv, color="#BDBDBD", lw=1.2, ls="--")
    ax.text(0.5, 0.69, "Tur 1\n(Pretrained+EN)", ha="center", fontsize=8, color="#37474F")
    ax.text(2.5, 0.69, "Tur 2\n(FT-CLAP+EN)",   ha="center", fontsize=8, color="#37474F")
    ax.text(4.5, 0.69, "Tur 3\n(FT-CLAP+TR)",   ha="center", fontsize=8, color="#37474F")

    ax.set_xticks(x); ax.set_xticklabels(etik, fontsize=8.5)
    ax.set_ylabel("Test Macro-F1"); ax.set_ylim(0, 0.74)
    ax.set_title("Makro F1 Ozeti: 3 Uretim Turu x 2 Model x Baseline/Augmented")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_kaydet(fig, "makro_f1_ozet")


# ═════════════════════════════════════════════════════════════════════════════
# REPORTLAB STILLERI
# ═════════════════════════════════════════════════════════════════════════════

def stiller_olustur():
    base = getSampleStyleSheet()

    s = {}
    s["baslik"] = ParagraphStyle(
        "baslik", parent=base["Title"],
        fontSize=20, leading=26, spaceAfter=6,
        textColor=colors.HexColor("#0D1B2A"), alignment=TA_CENTER,
    )
    s["alt_baslik"] = ParagraphStyle(
        "alt_baslik", parent=base["Normal"],
        fontSize=11, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#37474F"), alignment=TA_CENTER,
    )
    s["bolum"] = ParagraphStyle(
        "bolum", parent=base["Heading1"],
        fontSize=13, leading=18, spaceBefore=16, spaceAfter=6,
        textColor=colors.HexColor("#0D47A1"),
        borderPad=0, leftIndent=0,
    )
    s["alt_bolum"] = ParagraphStyle(
        "alt_bolum", parent=base["Heading2"],
        fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#1565C0"),
    )
    s["govde"] = ParagraphStyle(
        "govde", parent=base["Normal"],
        fontSize=9.5, leading=14, spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    s["ozet_kutu"] = ParagraphStyle(
        "ozet_kutu", parent=base["Normal"],
        fontSize=9, leading=13, spaceAfter=4,
        alignment=TA_JUSTIFY,
        leftIndent=10, rightIndent=10,
    )
    s["tablo_baslik"] = ParagraphStyle(
        "tablo_baslik", parent=base["Normal"],
        fontSize=9, leading=12, spaceAfter=4,
        alignment=TA_CENTER, textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    s["sekil_altyazi"] = ParagraphStyle(
        "sekil_altyazi", parent=base["Normal"],
        fontSize=8, leading=11, spaceAfter=8, spaceBefore=3,
        alignment=TA_CENTER, textColor=colors.HexColor("#546E7A"),
    )
    s["kucuk"] = ParagraphStyle(
        "kucuk", parent=base["Normal"],
        fontSize=8, leading=11, spaceAfter=4,
        textColor=colors.HexColor("#546E7A"),
    )
    return s


# ═════════════════════════════════════════════════════════════════════════════
# TABLO YARDIMCILARI
# ═════════════════════════════════════════════════════════════════════════════

TABLO_MAVI  = colors.HexColor("#0D47A1")
TABLO_ACIK  = colors.HexColor("#E3F2FD")
TABLO_ZEBRA = colors.HexColor("#F5F9FF")

def tablo_stili_temel(satirlar):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), TABLO_MAVI),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 8.5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE",    (0,1), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, TABLO_ZEBRA]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#B0BEC5")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ])


def img(yol, genislik, en_boy=None):
    """en_boy: yukseklik/genislik orani (orn. 0.6). Verilmezse PNG'den hesaplanir."""
    from PIL import Image as PILImage
    with PILImage.open(yol) as pil:
        w_px, h_px = pil.size
    oran = en_boy if en_boy else h_px / w_px
    return Image(yol, width=genislik, height=genislik * oran)


# ═════════════════════════════════════════════════════════════════════════════
# BELGE OLUSTUR
# ═════════════════════════════════════════════════════════════════════════════

def belge_olustur(figler, st):
    """Tum icerik ogelerini (flowables) dondurur."""
    icerik = []
    tarih  = datetime.now().strftime("%d %B %Y")
    P = Paragraph

    # ── KAPAK ────────────────────────────────────────────────────────────────
    icerik += [
        Spacer(1, 2.5*cm),
        P("BBC Ses Arsivi Cok-Sinifli Ses Siniflandirmasi:", st["baslik"]),
        P("Geleneksel Makine Ogrenimi, Derin Ogrenme ve Late Fusion<br/>Karsilastirmali Analizi", st["baslik"]),
        Spacer(1, 0.6*cm),
        HRFlowable(width="80%", thickness=2, color=TABLO_MAVI, spaceAfter=10),
        P("Doktora Tezi Ara Raporu — Ses Isleme Arastirma Grubu", st["alt_baslik"]),
        P(f"Rapor Tarihi: {tarih}", st["alt_baslik"]),
        Spacer(1, 1.2*cm),
    ]

    # Ozet kutusu
    ozet_metni = (
        "Bu rapor, BBC Ses Arsivi uzerinde yurutulen doktora tezi arastirmasinin son halini "
        "sunmaktadir. Calisma iki bilesenlidir: (1) 26 sinifli saglam bir siniflandirma temeli "
        "— en iyi model CNN/ResNet-18 (test makro-F1=0.6716); BiLSTM, AST, Optuna-ayarli LightGBM "
        "ve Gec Birlesim karsilastirilmistir. (2) Uretici yapay zeka (AudioLDM2 + alana-ozgu "
        "ince-ayarli CLAP) ile veri artirma. Degerlendirmede once iki metodolojik tuzak "
        "(uretimde sabit-tohum cesitlilik cokusu; FAD non-determinizmi) tespit edilip duzeltilmis; "
        "deterministik FAD Ingilizce turlarda neredeyse ozdestir (13.6-14.0; Turkce=15.9). "
        "Duzeltme sonrasi ANA BULGU: augmentasyon faydasi HEDEF AKUSTIK TUTARLILIGINA baglidir. "
        "Akustik olarak tutarli uc dusuk-kaynakli sinifta (kopek, kus, hava) augmentasyon "
        "istatistiksel olarak ANLAMLI ve dort uretim konfigurasyonunda (H1, coh2, coh3, coh4) "
        "TEKRARLANABILIR bir kazanim saglar (tutarli-3 birlesik McNemar dort kosuda da p<1e-5, ***; "
        "kus icin +10.6 puan dogruluk); buna karsilik es zamanli egitilen heterojen catch-all "
        "kontrol hicbir kosuda fayda gormez. Ilke nicel olarak dogrulanmis (VGGish sinif-ici dagilim "
        "kazanimi ongorur, r=-0.76; ayni uzayda FAD ongormez) ve augmentasyon turunden bagimsizdir "
        "(geleneksel SpecAugment ayni deseni gosterir). En guclu kazanim, iki kez ince-ayarli "
        "(Dongu2) coh4 konfigurasyonundandir."
    )
    ozet_tablo = Table(
        [[P(ozet_metni, st["ozet_kutu"])]],
        colWidths=[15*cm],
    )
    ozet_tablo.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 1, TABLO_MAVI),
        ("BACKGROUND", (0,0), (-1,-1), TABLO_ACIK),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING",(0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0), (-1,-1), 12),
    ]))
    icerik += [ozet_tablo, Spacer(1, 0.8*cm)]

    # Ozet metrik tablosu
    ozet_satirlar = [
        ["Metrik", "Deger"],
        ["Toplam WAV Dosyasi", "15.356"],
        ["Toplam Segment (5 sn)", "331.785"],
        ["Sinif Sayisi", "26"],
        ["Ozellik Boyutu (Klasik ML)", "467-dim"],
        ["Spectrogram Boyutu (DL)", "(1, 128, 313)"],
        ["En Iyi Klasik ML (LightGBM)", "Test F1 = 0.5857"],
        ["En Iyi Derin Ogrenme (CNN)", "Test F1 = 0.6716"],
        ["En Iyi Fusion (Val-F1 Agirlik)", f"Test F1 = {fusion_d.get('fusion_sonuclari',{}).get('val_f1_agirlik',{}).get('test',0.666):.4f}"],
        ["Oracle Ust Siniri", "Test F1 = 0.7812"],
    ]
    t = Table(ozet_satirlar, colWidths=[9*cm, 6*cm])
    t.setStyle(tablo_stili_temel(ozet_satirlar))
    icerik += [
        P("Tablo 1. Calısma Ozeti", st["sekil_altyazi"]),
        t,
        PageBreak(),
    ]

    # ── BOLUM 1: GIRIS ───────────────────────────────────────────────────────
    icerik += [
        P("1. Giris", st["bolum"]),
        P("Ses siniflandirmasi, makine ogrenimi ve sinyal islemede uzun suredir aktif "
          "arastirma alanlari arasinda yer almaktadir. BBC Ses Arsivi, sirket icinde "
          "kullanilmak uzere yillar boyunca derlenen 26 farkli akustik kategoriyi "
          "kapsayan 15.356 adet profesyonel kayit icerir. Bu veri seti; dogal sesler, "
          "kentsel atmosferler, ulasim araclari, hayvan sesleri ve insansi aktiviteler "
          "gibi genis bir akustik yelpazeyi temsil etmektedir.", st["govde"]),
        P("Bu calismada iki temel bilimsel soru ele alinmaktadir: (1) Geleneksel el-yapimi "
          "ozelliklere dayali makine ogrenimi modelleri ile uctan-uca ogrenme yapan derin "
          "sinir aglari arasindaki performans farki nedir? (2) Birden fazla modelin "
          "geç füzyon ile birlestirimi bireysel model performansini anlamsali olcude "
          "iyilestirebilir mi?", st["govde"]),
        Spacer(1, 0.3*cm),

        P("2. Veri Seti ve On Isleme", st["bolum"]),
        P("2.1 Veri Seti Ozellikleri", st["alt_bolum"]),
        P("Ham veri seti F: diskinde 26 klasorde organize edilmis 15.356 WAV "
          "dosyasindan olusmaktadir. Dosyalarin uzunlugu 5 saniyeden 10 dakikaya "
          "kadar degismektedir. Tek orneklem frekansi olarak 16 kHz kullanilmis; stereo "
          "kayitlar mono'ya donusturulmustur. Her dosya, 5 saniyelik otusan olmayan "
          "dilimlere bolunmus ve toplamda 331.785 egitim segmenti elde edilmistir.",
          st["govde"]),
        P("Siniflandirma problemi icin dosya-duzeyli stratified train/val/test bolumleme "
          "uygulanmistir: %70 egitim, %15 dogrulama, %15 test (10.748/2.304/2.304 dosya). "
          "Bu yaklasim, ayni dosyadan "
          "gelen segmentlerin farkli bolumlere dustugunde olusabilecek veri sizintisini "
          "engeller.", st["govde"]),
        Spacer(1, 0.2*cm),
        img(figler["sinif_dagilimi"], 14*cm),
        P("Sekil 1. Test kumesinin sinif bazli dagilimi. Yesil >= 2000 segment, "
          "turuncu 500-2000, kirmizi < 500 segmenti temsil eder. kapi_sesleri (139) "
          "ve tarim_ciftlik (218) siniflarinda belirgin dengesizlik gorusu dikkat cekicidir.",
          st["sekil_altyazi"]),

        P("2.2 Ozellik Cikarimi", st["alt_bolum"]),
        P("Geleneksel ML modelleri icin toplam 467-boyutlu el-yapimi ozellik vektoru "
          "cikarilmistir:", st["govde"]),
    ]

    ozellik_satirlar = [
        ["Ozellik Grubu", "Boyut", "Aciklama"],
        ["MFCC (delta + delta2)", "120", "Mel-frekans kepstral katsayilari ve turevleri"],
        ["Mel-filterbank Enerjisi", "80",  "Log-mel spektrum (delta + delta2 dahil)"],
        ["Spectral Centroid",  "20", "Spektral merkez, bant genisligi, kontrast"],
        ["Spectral Rolloff",   "20", "Yuzde 85 ve 95 rolloff frekanslari"],
        ["Chroma STFT",        "84", "12-pitch sinif profili + harmonik analiz"],
        ["Zero Crossing Rate", "3",  "ZCR + varyas + geri cagri orani"],
        ["Tonnetz",            "36", "Tonal centroid ozellikler"],
        ["RMS Enerji",         "20", "Cerceve-duzeyli RMS enerjisi ve istatistikleri"],
        ["Spectral Contrast",  "84", "7 alt-bant spektral kontrast"],
        ["TOPLAM",             "467", "—"],
    ]
    t = Table(ozellik_satirlar, colWidths=[5*cm, 2*cm, 8*cm])
    t.setStyle(tablo_stili_temel(ozellik_satirlar))
    # Son satiri kalin yap
    t.setStyle(TableStyle([
        ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,-1), (-1,-1), colors.HexColor("#BBDEFB")),
    ]))
    icerik += [
        P("Tablo 2. Ozellik Vektoru Bilesimi (467 boyut)", st["sekil_altyazi"]),
        t,
        Spacer(1, 0.4*cm),
        P("Derin ogrenme modelleri icin 128-mel log-mel spektrogram kullanilmistir. "
          "5 saniyelik ses segmentleri icin (SR=16kHz, n_fft=1024, hop=256) "
          "boyutlar (1, 128, 313) olmaktadir. Tum mel cache'ler float16 .npy dosyalari "
          "olarak on-hesaplanmis (toplam 26.5 GB) ve egitim sirasinda RAM'e "
          "yuklenmistir (~8-9 GB).", st["govde"]),
        PageBreak(),
    ]

    # ── 2.3 VERI NASIL GORUNUR (aciklayici) ─────────────────────────────────
    veri_ornek = [
        ["WAV ses kaydi (dosya adi)", "Sinif", "Idx", "Segment\n(5 sn)"],
        ["Airedale barking angrily.wav",   "kopek_sesleri",  "15", "2"],
        ["Cicadas stridulating.wav",       "kus_sesleri",    "16", "63"],
        ["Avalanche actuality.wav",        "hava_olaylari",  "10", "4"],
        ["Ambulance, exterior.wav",        "acil_servis",    "0",  "8"],
        ["Monastery handbell sounded.wav", "dini_mekan",     "5",  "13"],
        ["Wall being pulled down.wav",     "insaat_sesleri", "11", "1"],
    ]
    t_veri = Table(veri_ornek, colWidths=[6.6*cm, 3.6*cm, 1.4*cm, 2.0*cm])
    t_veri.setStyle(tablo_stili_temel(veri_ornek))
    ozn_ornek = [
        ["f0", "f1", "f2", "f3", "f4", "...", "f466"],
        ["-307.4", "-59.8", "-26.6", "45.5", "-11.5", "...", "(toplam 467)"],
    ]
    t_ozn = Table(ozn_ornek, colWidths=[1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.9*cm, 1.2*cm, 2.6*cm])
    t_ozn.setStyle(tablo_stili_temel(ozn_ornek))
    icerik += [
        P("2.3 Veri Nasil Gorunur? — Ornek Kayitlar ve Donusum", st["alt_bolum"]),
        P("Ses verisiyle calismamis bir okuyucu icin, verinin ham halini ve modele "
          "verilmeden once gectigi donusumu somutlastirmak yararlidir. Calismanin "
          "cekirdek veri tablosu (veri_listesi) her satirinda bir WAV ses kaydini "
          "tutar: dosya adi, ait oldugu sinif, sinif numarasi (idx) ve kayittan cikan "
          "5 saniyelik parca (segment) sayisi.", st["govde"]),
        P("Tablo 3. Ornek ham veri kayitlari (veri_listesi, 15.356 satirdan secme)", st["sekil_altyazi"]),
        t_veri, Spacer(1, 0.3*cm),
        P("Okuma: Her WAV dosyasi 5 saniyelik ORTUSMEYEN parcalara bolunur. Ornegin "
          "'Cicadas stridulating' (agustosbocegi) kaydi 63 segment uretir (~5 dakikalik "
          "kayit), 'Wall being pulled down' (yikim) ise yalnizca 1 segment (~5 sn). "
          "Toplam 15.356 dosya, 331.785 segmente bolunur.", st["govde"]),
        P("Donusum: Her 5 sn'lik segment, modele verilmeden once bir LOG-MEL "
          "SPEKTROGRAMA donusturulur — 128x128'lik bir sayi matrisi, yani siyah-beyaz "
          "bir GORUNTU gibi (yatay eksen: zaman, dikey eksen: frekans, parlaklik: enerji). "
          "Boylece ses, goruntu siniflandirmada kanitlanmis CNN/ResNet yontemleriyle "
          "islenebilir hale gelir. Klasik makine ogrenimi (LightGBM) icin ise ayni "
          "segment 467 sayisal ozellige indirgenir:", st["govde"]),
        P("Tablo 4. Bir segmentin 467-boyutlu oznitelik vektoru (ilk degerler)", st["sekil_altyazi"]),
        t_ozn, Spacer(1, 0.2*cm),
        P("Ozetle: ses dosyasi -> 5 sn segmentler -> her segment ya bir 128x128 goruntuye "
          "(derin ogrenme) ya da 467 sayiya (klasik ML) donusur. Modeller dogrudan bu "
          "sayisal temsiller uzerinde calisir; insanin duydugu 'ses' degil, onun olcumlenmis "
          "halidir.", st["govde"]),
        PageBreak(),
    ]

    # ── BOLUM 3: YONTEMLER ────────────────────────────────────────────────────
    icerik += [
        P("3. Yontemler", st["bolum"]),
        P("3.1 Geleneksel Makine Ogrenimi", st["alt_bolum"]),
        P("Algoritma secim asamasinda 467-boyutlu ozellik uzayinda 6 farkli algoritma "
          "karsilastirilmistir. SVM ve k-NN gibi hesaplama yuku yuksek modeller icin "
          "%10 alt-orneklem uygulanmistir. GPU hizlandirmasindan faydalanmak icin "
          "XGBoost (device='cuda', tree_method='hist') ve LightGBM (device='gpu') "
          "kullanilmistir.", st["govde"]),
        P("Kazanan algoritma LightGBM uzerinde Optuna kutuphanesi ile 50 deneme "
          "boyunca hiperparametre optimizasyonu (HPO) gerceklestirilmistir. "
          "5-fold stratified cross-validation ile en iyi parametreler belirlenmistir.",
          st["govde"]),
        Spacer(1, 0.2*cm),
        img(figler["model_secim"], 13*cm),
        P("Sekil 2. Klasik ML algoritma secim sonuclari. LightGBM test kumesinde "
          "0.558 ile en yuksek Macro-F1'e ulasarak HPO icin secilmistir.",
          st["sekil_altyazi"]),

        P("3.2 Derin Ogrenme Modelleri", st["alt_bolum"]),
    ]

    mimari_satirlar = [
        ["Model",        "Mimari",          "Parametre",  "Giris Boyutu",      "Ozellikler"],
        ["CNN",          "ResNet-18 + 1-ch conv", "11.18M", "(1, 128, 313)",  "Pretrained ImageNet"],
        ["BiLSTM",       "3-katman BiLSTM\n+ Temporal Attention", "~2.8M", "(313, 128)", "Sifirdan egitim"],
        ["AST",          "Vision Transformer\n(12 katman, 768 dim)", "85.5M", "(128, 313)", "Pretrained AudioSet"],
    ]
    t = Table(mimari_satirlar, colWidths=[2.5*cm, 4*cm, 2.2*cm, 2.8*cm, 3.5*cm])
    t.setStyle(tablo_stili_temel(mimari_satirlar))
    icerik += [
        P("Tablo 5. Derin Ogrenme Mimarileri", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        P("Tum derin ogrenme modelleri icin ortak egitim protokolu: AdamW optimizer "
          "(lr=3e-4, weight_decay=1e-4), CosineAnnealingLR (eta_min=lr/100), "
          "label_smoothing=0.1, batch=32, en fazla 30 epoch, patience=7 erken "
          "durdurma, gradyan kirpma (max-norm 5,0) ve AMP (Automatic Mixed Precision) "
          "kullanilmistir. Tum kosular sabit tohumla (seed=42) deterministiktir; "
          "RTX 3060 12 GB GPU'da egitim gerceklestirilmistir.", st["govde"]),

        P("3.3 Late Fusion", st["alt_bolum"]),
        P("Egitilmis modellerden her biri test ornekleri icin 26-boyutlu softmax olasilik "
          "vektoru uretmektedir. Bu vektorler dort farkli yontemle birlestirilmistir: "
          "(1) esit agirlikli ortalama, (2) val-F1 orantili agirlikli ortalama, "
          "(3) 78-boyutlu birlestirilmis vektoru girdi olarak alan LogisticRegression "
          "meta-learner, (4) her ornek icin en az bir modelin dogru tahmin ettigini "
          "varsayan oracle ust-sinir.", st["govde"]),
        PageBreak(),
    ]

    # ── BOLUM 4: SONUCLAR ─────────────────────────────────────────────────────
    icerik += [
        P("4. Deneysel Sonuclar", st["bolum"]),
        P("4.1 Genel Model Karsilastirmasi", st["alt_bolum"]),
        img(figler["genel_karsilastirma"], 15*cm),
        P("Sekil 3. Tum model ailelerinin test kumesi Macro-F1 performanslari. "
          "CNN (ResNet-18) bireysel en yuksek degere ulasirken; LightGBM geleneksel "
          "ML'de acik ara lider konumdadir.",
          st["sekil_altyazi"]),
        Spacer(1, 0.2*cm),
    ]

    # Tam sonuc tablosu
    bireysel_test = fusion_d.get("bireysel_test", {})
    tam_satirlar = [
        ["Model", "Kategori", "Val Macro-F1", "Test Macro-F1", "Test Acc.", "Egitim"],
        ["k-NN (k=5)",        "Klasik ML",  "0.3132", "0.2960", "0.3512", "< 1 s"],
        ["SVM (RBF)",         "Klasik ML",  "0.3938", "0.3803", "0.4432", "31 s *"],
        ["Extra Trees",       "Klasik ML",  "0.4652", "0.4523", "0.5033", "88 s"],
        ["XGBoost (GPU)",     "Klasik ML",  "0.5417", "0.5369", "0.5680", "160 s"],
        ["HistGradBoost",     "Klasik ML",  "0.5511", "0.5480", "0.5788", "1150 s"],
        ["LightGBM+Optuna",   "Klasik ML",  "0.5573", "0.5857", "0.5903", "~1800 s"],
        ["BiLSTM+Attention",  "Derin OG",   f"{lstm_d['en_iyi_val_f1']:.4f}", f"{bireysel_test.get('lstm', lstm_d['test_macro_f1']):.4f}", "0.5452", f"{lstm_d['egitim_dakika']:.0f} dk"],
        ["AST Transformer",   "Derin OG",   f"{ast_d['en_iyi_val_f1']:.4f}", f"{bireysel_test.get('ast', 0.6156):.4f}", "N/A", f"{ast_d['egitim_dakika']:.0f} dk †"],
        ["CNN (ResNet-18)",   "Derin OG",   f"{cnn_d['en_iyi_val_f1']:.4f}", f"{bireysel_test.get('cnn', cnn_d['test_macro_f1']):.4f}", "0.6859", f"{cnn_d['egitim_dakika']:.0f} dk"],
        ["Fusion (Esit)",     "Late Fusion", "0.6568", f"{fusion_d.get('fusion_sonuclari',{}).get('esit_agirlik',{}).get('test',0):.4f}", "—", "—"],
        ["Fusion (Val-F1)",   "Late Fusion", "0.6603", f"{fusion_d.get('fusion_sonuclari',{}).get('val_f1_agirlik',{}).get('test',0):.4f}", "—", "—"],
        ["Fusion (LogReg)",   "Late Fusion", "0.7027", f"{fusion_d.get('fusion_sonuclari',{}).get('meta_learner',{}).get('test',0):.4f}", "—", "—"],
        ["Oracle Ust Sinir",  "Teorik",      "0.7778", "0.7812", "—", "—"],
    ]
    t = Table(tam_satirlar, colWidths=[4*cm, 2.6*cm, 2.4*cm, 2.8*cm, 2.2*cm, 2*cm])
    t.setStyle(tablo_stili_temel(tam_satirlar))
    # CNN satirini vurgula
    for row_idx, row in enumerate(tam_satirlar):
        if row[0] == "CNN (ResNet-18)":
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#BBDEFB")),
                ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
        if row[0] == "Oracle Ust Sinir":
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F3E5F5")),
                ("FONTNAME",   (0, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
    icerik += [
        P("Tablo 6. Tam Model Karsilastirma Tablosu — * alt-orneklem, † erken durduruldu",
          st["sekil_altyazi"]),
        t, PageBreak(),

        P("4.2 Derin Ogrenme Egitim Surecleri", st["alt_bolum"]),
        img(figler["egitim_sureci"], 15*cm),
        P("Sekil 4. Uc derin ogrenme modelinin egitim sürecleri. Ust sira: "
          "train ve val loss. Alt sira: val Macro-F1 (kirmizi nokta = en iyi epoch, "
          "kesik cizgi = test F1). CNN 25. epoch'ta en yüksege ulasirken BiLSTM "
          "23. epoch'ta, AST ise 7. epoch'ta en iyi val F1'ini elde etmistir.",
          st["sekil_altyazi"]),
        Spacer(1, 0.3*cm),
        P("CNN'in egitim grafiginde train acc'in hizla >%98'e cikmasina kars. "
          "val loss'un sabit kalmasi hafif ama yonetilebilir bir overfitting isaretini "
          "gostermektedir; label smoothing ve weight decay bu etkiyi sinirlamistir. "
          "BiLSTM tum 30 epoch boyunca daha yavas ve duzenli bir ogrenme egrisi "
          "sergilemistir. AST ise 7. epoch'tan itibaren val loss'un artmasi nedeniyle "
          "overfitting'e girmis; 85M parametreli modelin 234K egitim segmentiyle "
          "regularizasyon olmaksizin optimize edilmesindeki gucluge isaret etmektedir.",
          st["govde"]),
        PageBreak(),

        P("4.3 Sinif Bazli Analiz", st["alt_bolum"]),
        img(figler["sinif_heatmap"], 15*cm),
        P("Sekil 5. CNN ve BiLSTM icin sinif bazli F1 isi haritasi. Yesil >= 0.75, "
          "sari 0.50-0.75, kirmizi < 0.50 bolgelerini gostermektedir. CNN tutarli "
          "sekilde tum siniflarda BiLSTM'i gecmektedir.",
          st["sekil_altyazi"]),
        Spacer(1, 0.3*cm),
        img(figler["cnn_sinif_cubuk"], 14*cm),
        P("Sekil 6. CNN (ResNet-18) sinif bazli F1 skorlari, dusukten yuksege "
          "siralanmis. Yesil F1>=0.75, turuncu 0.50-0.75, kirmizi <0.50.",
          st["sekil_altyazi"]),
        PageBreak(),
    ]

    # En iyi / en kotu sinif tablosu
    rapor = cnn_d["sinif_raporu"]
    f1_lst = [(SINIF_ISIMLERI[i],
               rapor[str(i)]["f1-score"],
               rapor[str(i)]["precision"],
               rapor[str(i)]["recall"],
               int(rapor[str(i)]["support"])) for i in range(26)]
    f1_lst.sort(key=lambda x: x[1], reverse=True)

    iyi_satirlar = [["Sinif", "F1", "Precision", "Recall", "Support"]]
    for sinif, f1, p, r, s in f1_lst[:8]:
        iyi_satirlar.append([sinif, f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", str(s)])

    kotu_satirlar = [["Sinif", "F1", "Precision", "Recall", "Support"]]
    for sinif, f1, p, r, s in f1_lst[-8:]:
        kotu_satirlar.append([sinif, f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", str(s)])

    t_iyi  = Table(iyi_satirlar,  colWidths=[5*cm, 2*cm, 2.5*cm, 2.5*cm, 2*cm])
    t_kotu = Table(kotu_satirlar, colWidths=[5*cm, 2*cm, 2.5*cm, 2.5*cm, 2*cm])
    t_iyi.setStyle(tablo_stili_temel(iyi_satirlar))
    t_kotu.setStyle(tablo_stili_temel(kotu_satirlar))

    icerik += [
        P("4.3.1 CNN En Basarili 8 Sinif (Test Kumesi)", st["alt_bolum"]),
        t_iyi, Spacer(1, 0.5*cm),
        P("4.3.2 CNN En Dusuk Performansli 8 Sinif (Test Kumesi)", st["alt_bolum"]),
        t_kotu, Spacer(1, 0.3*cm),
        P("Dusuk performansin iki temel nedeni vardir: (1) siniflar arasi akustik "
          "benzerlik — insaat_sesleri ve endustri_fabrika gibi kategoriler benzer "
          "frekans iceriklerine sahiptir; (2) veri dengesizligi — kapi_sesleri "
          "(139 test segmenti) ve tarim_ciftlik (218 segment) gibi az ornekli "
          "siniflar genel performansi asagi cekmektedir. Bu durum, Asamali "
          "Tur 1 uretici artirmanin hedef siniflarini belirlemektedir.",
          st["govde"]),
        PageBreak(),

        P("4.4 Late Fusion Sonuclari", st["alt_bolum"]),
        img(figler["fusion"], 14*cm),
        P("Sekil 7. Late fusion yontemlerinin bireysel modellerle karsilastirilmasi.",
          st["sekil_altyazi"]),
        Spacer(1, 0.2*cm),
    ]

    fusion_satirlar = [
        ["Yontem", "Val Macro-F1", "Test Macro-F1", "CNN'e Goreceli"],
        ["Bireysel CNN",        f"{cnn_d['en_iyi_val_f1']:.4f}", f"{bireysel_test.get('cnn', cnn_d['test_macro_f1']):.4f}", "Referans"],
        ["Esit Agirlik",        "0.6568", f"{fusion_d.get('fusion_sonuclari',{}).get('esit_agirlik',{}).get('test',0):.4f}", f"{fusion_d.get('fusion_sonuclari',{}).get('esit_agirlik',{}).get('test',0) - bireysel_test.get('cnn', cnn_d['test_macro_f1']):+.4f}"],
        ["Val-F1 Agirlikli",    "0.6603", f"{fusion_d.get('fusion_sonuclari',{}).get('val_f1_agirlik',{}).get('test',0):.4f}", f"{fusion_d.get('fusion_sonuclari',{}).get('val_f1_agirlik',{}).get('test',0) - bireysel_test.get('cnn', cnn_d['test_macro_f1']):+.4f}"],
        ["Meta-Learner (LogReg)","0.7027", f"{fusion_d.get('fusion_sonuclari',{}).get('meta_learner',{}).get('test',0):.4f}", f"{fusion_d.get('fusion_sonuclari',{}).get('meta_learner',{}).get('test',0) - bireysel_test.get('cnn', cnn_d['test_macro_f1']):+.4f}"],
        ["Oracle Ust Siniri",   "0.7778", "0.7812", "+0.1096"],
    ]
    t = Table(fusion_satirlar, colWidths=[5*cm, 3.5*cm, 3.5*cm, 3*cm])
    t.setStyle(tablo_stili_temel(fusion_satirlar))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F3E5F5")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
    ]))
    icerik += [
        P("Tablo 7. Late Fusion Sonuc Tablosu", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        P("Meta-Learner val kumesinde 0.7027 gibi yuksek bir skor elde etmis; ancak "
          "test kumesinde 0.6397'ye duserek overfitting sergilemistir. Bunun temel "
          "nedeni, meta-learner'in 78-boyutlu giris uzayinda validasyon kumesine "
          "asiri uyum saglamasidir. Oracle ust siniri (0.7812) ile en iyi fusion "
          "yontemi (0.6664) arasindaki 0.115 bosluk, modellerin yuksek derecede "
          "korele hatalar yaptigina ve daha cesitli model mimarilerine (ornegin "
          "conformer ya da wav2vec2) ihtiyac duyulduguna isaret etmektedir.",
          st["govde"]),
        PageBreak(),
    ]

    # ── BOLUM 5: ARASTIRMA MIMARISI ──────────────────────────────────────────
    icerik += [
        P("5. Uretici Yapay Zeka ile Veri Artirma", st["bolum"]),
        P("5.1 Arastirma Pipeline Genel Bakis", st["alt_bolum"]),
        P("Calismada 5 asama uzerine kurulu butunlesik bir arastirma mimarisi "
          "tasarlanmistir. Her asama bir oncekinin uzerine insa edilmekte; "
          "klasik baseline'dan iteratif ince ayar dongusuне kadar "
          "sistematik bir ilerleme izlenmektedir.", st["govde"]),
    ]
    if figler.get("pipeline"):
        icerik += [
            img(str(ROOT / "outputs" / "report" / "coherence_figs" / "sema_pipeline.png"), 16*cm),
            P("Sekil 8. Calismanin genel akisi (sistem semasi). BBC arsivi ortak "
              "on-isleme ve dosya-duzeyi bolmeden (seed=42) gecer; ardindan iki kol "
              "ayrisir: (A) bes model ailesiyle siniflandirma temeli ve (B) CLAP alan "
              "ince-ayariyla yonlendirilen AudioLDM2 uretim hatti. Sentetik klipler ozdes "
              "protokolle egitime eklenir; degerlendirme FAD, McNemar ve VGGish tutarlilik "
              "olcutunu birlestirir.", st["sekil_altyazi"]),
            Spacer(1, 0.3*cm),
            img(str(ROOT / "outputs" / "report" / "coherence_figs" / "sema_folder.png"), 12.5*cm),
            P("Sekil 9. Proje deposu yapisi — veri, cekirdek kod, pipeline scriptleri, "
              "egitilmis modeller, sayisal sonuclar ve ciktilar. Yildizli ogeler (yildiz "
              "isareti) anahtar bilesenlerdir: uretim promptlari, CLAP fine-tune ciftleri, "
              "dort konfigurasyon agirliklari ve anlamlilik/tutarlilik sonuclari.",
              st["sekil_altyazi"]),
            Spacer(1, 0.2*cm),
        ]

    tur_tum_satirlar = [
        ["Tur", "CLAP Modeli", "Prompt Dili", "Klip Sayisi", "Amac", "Durum"],
        ["Tur 1", "Pretrained\n(LAION/AudioSet)", "Ingilizce", "4 sinif x 50\n= 200 klip",
         "Zero-shot ablation\n(baseline karsilastirma)", "TAMAMLANDI"],
        ["Tur 2", "BBC Fine-tuned\n(dongu1, 26 saat)", "Ingilizce", "4 sinif x 50\n= 200 klip",
         "Fine-tuning katkisini\nolcme", "TAMAMLANDI"],
        ["Tur 3", "BBC Fine-tuned\n(dongu1, ayni ckpt)", "Turkce", "4 sinif x 50\n= 200 klip",
         "Turkce uretim\nkapasitesi testi", "TAMAMLANDI"],
        ["Tur 4", "BBC+Tur2 Fine-tuned\n(dongu2, 8.4 saat)", "Ingilizce", f"4 sinif x 50\n= 200 klip\n({len(tur4_wav)} uretildi)",
         "Iteratif ince ayar\n(Dongu2) katkisi", "TAMAMLANDI"],
    ]
    t = Table(tur_tum_satirlar, colWidths=[1.6*cm, 3*cm, 2.2*cm, 2.5*cm, 3.5*cm, 2.2*cm])
    t.setStyle(tablo_stili_temel(tur_tum_satirlar))
    for ri, row in enumerate(tur_tum_satirlar):
        if "TAMAMLANDI" in str(row[-1]):
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (-1, ri), (-1, ri), colors.HexColor("#2E7D32")),
                ("FONTNAME",  (-1, ri), (-1, ri), "Helvetica-Bold"),
            ]))
        elif "FAD" in str(row[-1]):
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#FFF9C4")),
                ("TEXTCOLOR",  (-1, ri), (-1, ri), colors.HexColor("#E65100")),
                ("FONTNAME",   (-1, ri), (-1, ri), "Helvetica-Bold"),
            ]))
    icerik += [
        P("Tablo 8. Iteratif Uretici Artirma — Tam Tur Plani ve Durum", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        PageBreak(),
    ]

    # ── 5.2 CLAP FINE-TUNING ─────────────────────────────────────────────────
    # Prompt ornekleri (gercek Turkce -> DejaVuSans). Tutarli + heterojen siniflar.
    prompt_ornek = [
        ["Sinif", "Turkce prompt (ornek)", "Ingilizce prompt (ornek)"],
        ["kopek", "köpek öfkeyle güçlü sesle havlıyor", "dog barking loudly and aggressively"],
        ["kuş", "sabahın erken saatlerinde kuş korosu", "birds singing at early morning dawn chorus"],
        ["hava", "gök gürültülü yoğun yağmur fırtınası", "thunderstorm with heavy driving rain"],
        ["acil servis", "şehirde ambulans sireni çalıyor", "ambulance siren wailing in city traffic"],
        ["dini mekân", "katedralin içinde org müziği yankılanıyor", "cathedral organ music echoing in nave"],
        ["inşaat", "pnömatik matkap inşaat sahasında", "pneumatic drill breaking up road surface"],
    ]
    t_prompt = Table(prompt_ornek, colWidths=[2.4*cm, 6.3*cm, 6.3*cm])
    t_prompt.setStyle(tablo_stili_temel(prompt_ornek))
    t_prompt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _TR_FONT_B),
        ("FONTNAME", (0, 1), (-1, -1), _TR_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    cift_ornek = [
        ["Gercek BBC kaydi", "Turkce aciklama", "Ingilizce aciklama"],
        ["Airedale barking angrily.wav", "Airedale öfkeyle havlıyor", "Airedale barking angrily"],
        ["Monastery handbell sounded.wav", "Manastır el çanı çalındı", "Monastery handbell sounded"],
    ]
    t_cift = Table(cift_ornek, colWidths=[6.0*cm, 4.5*cm, 4.5*cm])
    t_cift.setStyle(tablo_stili_temel(cift_ornek))
    t_cift.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), _TR_FONT_B),
        ("FONTNAME", (0, 1), (-1, -1), _TR_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    icerik += [
        P("5.2 CLAP Fine-Tuning — BBC Turkce/Ingilizce Hizalama", st["alt_bolum"]),
        P("AudioLDM2'nin metin kodlayicisi olan CLAP (Contrastive Language-Audio "
          "Pretraining — HTSAT-base + RoBERTa-base, toplam 199.56M param), "
          "15.356 gercek BBC ses-metin ciftini iceren ses_metin_eslestirme.parquet "
          "veri seti ile fine-tune edilmistir. Hedef: CLAP'in Turkce ve BBC'ye "
          "ozgu Ingilizce promptlari akustik uzaya dogru eslestirmeyi ogrenip "
          "ogrenmedigini test etmektir.", st["govde"]),
        P("Uretici modele NE uretecegini ve CLAP'a hangi sesin hangi metne "
          "karsilik geldigini, METIN PROMPT'lariyla anlatiriz. Her sinif icin "
          "5 Turkce + 5 Ingilizce prompt elle yazilmistir; model bu cumleyi okuyup "
          "ona uygun 5 sn'lik sentetik ses uretir. Bu olcekte Turkce prompt "
          "kullanimi bilgimiz dahilinde ilk kez denenmistir.", st["govde"]),
        P("Tablo 9. Sinif basina Turkce/Ingilizce prompt ornekleri (uretim icin)", st["sekil_altyazi"]),
        t_prompt, Spacer(1, 0.3*cm),
        P("CLAP ince-ayari icin ise her gercek BBC kaydi, dosya adindan turetilen "
          "bir Turkce ve Ingilizce aciklamayla eslestirilir (ses-metin cifti):", st["govde"]),
        P("Tablo 10. CLAP egitim verisi — ses-metin ciftleri (ses_metin_eslestirme)", st["sekil_altyazi"]),
        t_cift, Spacer(1, 0.25*cm),
    ]

    clap_test1 = clap_d1.get("test", {}) if clap_d1 else {}
    clap_test2 = clap_d2.get("test", {}) if clap_d2 else {}
    clap_satirlar = [
        ["Parametre", "Dongu 1\n(BBC-only)", "Dongu 2\n(BBC+Tur2)"],
        ["Egitim Verisi", "15.356 BBC ses-metin\n(gercek)", "15.356 BBC + 200\nTur2 sentetik"],
        ["Epoch", "5", "5 (1→5 resume)"],
        ["Batch Size", "16", "16"],
        ["Learning Rate", "3e-5", "3e-5"],
        ["Hard-Neg Beta", "0.3", "0.3"],
        ["Donuk Audio", "6 blok", "6 blok"],
        ["Donuk Text", "8 katman", "8 katman"],
        ["Toplam Sure", f"{clap_d1.get('egitim_saat', 26.07):.1f} saat", f"{clap_d2.get('egitim_saat', 8.42):.1f} saat"],
        ["Checkpoint", "clap_bbc_best.pt", "clap_bbc_dongu2_best.pt"],
    ]
    t = Table(clap_satirlar, colWidths=[4.5*cm, 5.0*cm, 5.5*cm])
    t.setStyle(tablo_stili_temel(clap_satirlar))
    icerik += [
        P("Tablo 11. CLAP Fine-Tuning Konfigurasyonu — Dongu 1 vs Dongu 2", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
    ]

    if figler.get("clap_egitim"):
        icerik += [
            img(figler["clap_egitim"], 14*cm),
            P("Sekil 10. CLAP Dongu 1 fine-tuning egitim sureci. Sol: NT-Xent validation loss "
              "epochlar boyunca. Sag: kNN siniflandirma dogrulugu ve A2T R@1 geri cagri "
              "metrigi. kNN ~0.056 ile sifirdan yuksek bir alignment kazanilmistir.",
              st["sekil_altyazi"]),
            Spacer(1, 0.2*cm),
        ]

    clap_met_satirlar = [
        ["Metrik", "Dongu 1\n(BBC-only)", "Dongu 2\n(BBC+Tur2)", "Aciklama"],
        ["A2T R@1",      f"{clap_test1.get('A2T_R@1', 0.0013):.4f}",  f"{clap_test2.get('A2T_R@1', 0.0):.4f}",
         "15K+ ses → dogru metin (1. sirada)"],
        ["T2A R@1",      f"{clap_test1.get('T2A_R@1', 0.0007):.4f}",  f"{clap_test2.get('T2A_R@1', 0.0006):.4f}",
         "15K+ metin → dogru ses (1. sirada)"],
        ["A2T R@10",     f"{clap_test1.get('A2T_R@10', 0.0072):.4f}", f"{clap_test2.get('A2T_R@10', 0.0051):.4f}",
         "Dogru eslesme ilk 10 icinde"],
        ["kNN Acc",      f"{clap_test1.get('knn_acc', 0.0566):.4f}",  f"{clap_test2.get('knn_acc', 0.0450):.4f}",
         "Embedding sinif ayirt edilebilirligi"],
        ["Val NT-Xent",  f"{clap_test1.get('val_kayip', 3.51):.4f}",  f"{clap_test2.get('val_kayip', 3.5026):.4f}",
         "Kontrast kayip (dusuk = iyi)"],
        ["En Iyi Epoch", str(clap_d1.get('en_iyi_epoch', 1)),          str(clap_d2.get('en_iyi_epoch', 1)),
         "Checkpoint epoch"],
    ]
    t = Table(clap_met_satirlar, colWidths=[2.5*cm, 2.2*cm, 2.2*cm, 8.1*cm])
    t.setStyle(tablo_stili_temel(clap_met_satirlar))
    icerik += [
        P("Tablo 12. CLAP Fine-Tuning Test Metrikleri — Dongu 1 vs Dongu 2", st["sekil_altyazi"]),
        t, Spacer(1, 0.2*cm),
        P("NOT: R@1 metrikleri mutlak deger olarak dusuk gorunse de, "
          "15.356 ses-metin cifti arasinda rastgele secimin R@1=0.000065 "
          "olacagi dikkate alindiginda kNN=0.056 ve A2T R@1=0.0013 anlamli "
          "bir alignment kazanimi gostermektedir. Dongu 2, kNN=0.045 ile "
          "Dongu 1'e yakin kalitede hizalama saglamistir; "
          "FAD iyilesme potansiyeli Tur 4 uretimi ile dogrulanacaktir.",
          st["govde"]),
        PageBreak(),
    ]

    # ── 5.3 TUR 1 SONUCLARI ──────────────────────────────────────────────────
    icerik += [
        P("5.3 Tur 1 — Pretrained AudioLDM2 + Ingilizce Prompt", st["alt_bolum"]),
        P("4 hedef sinif (insaat_sesleri, dini_mekan, acil_servis, diger_hayvanlar) "
          "icin pretrained AudioLDM2 ile 50'ser klip uretilmis; CNN (30 epoch) ve "
          "LightGBM modelleri augmented egitim setiyle yeniden egitilmistir.",
          st["govde"]),
    ]
    if figler.get("tur1_augmentasyon"):
        icerik += [
            img(figler["tur1_augmentasyon"], 14*cm),
            P("Sekil 11. Tur 1 augmentasyon etkisi. Mavi tonlar CNN, turuncu tonlar "
              "LightGBM sonuclarini gostermektedir. Delta degerleri bar ustunde gosterilmistir.",
              st["sekil_altyazi"]),
            Spacer(1, 0.2*cm),
        ]

    tur1_satirlar = [
        ["Sinif", "CNN\nBaseline", "CNN\nAug Tur1", "CNN\nDelta",
         "LGBM\nBaseline", "LGBM\nAug Tur1", "LGBM\nDelta"],
        ["insaat_sesleri", "0.3877", "0.4150", "+0.0273", "0.3329", "0.3586", "+0.0257"],
        ["dini_mekan",     "0.4614", "0.4655", "+0.0041", "0.3296", "0.3155", "-0.0141"],
        ["acil_servis",    "0.5894", "0.5797", "-0.0097", "0.3645", "0.4012", "+0.0367"],
        ["diger_hayvanlar","0.5823", "0.5510", "-0.0313", "0.4883", "0.4925", "+0.0042"],
        ["Genel (macro)",  "0.6716", "0.6667", "-0.0049", "0.5422", "0.5463", "+0.0041"],
    ]
    t2 = Table(tur1_satirlar, colWidths=[3.8*cm, 1.9*cm, 2.1*cm, 1.7*cm, 1.9*cm, 2.1*cm, 1.7*cm])
    t2.setStyle(tablo_stili_temel(tur1_satirlar))
    for ri, row in enumerate(tur1_satirlar[1:], start=1):
        for ci, val in enumerate([row[3], row[6]]):
            col = 3 if ci == 0 else 6
            if val.startswith("+"):
                t2.setStyle(TableStyle([("TEXTCOLOR", (col, ri), (col, ri), colors.HexColor("#2E7D32"))]))
            elif val.startswith("-"):
                t2.setStyle(TableStyle([("TEXTCOLOR", (col, ri), (col, ri), colors.HexColor("#C62828"))]))
    icerik += [
        P("Tablo 13. Tur 1 Sinif Bazinda F1 Karsilastirmasi", st["sekil_altyazi"]),
        t2, Spacer(1, 0.3*cm),
        P("Tur 1 bulgulari: Zero-shot pretrained AudioLDM2, en zayif sinif olan "
          "insaat_sesleri'nde tutarli iyilesme saglamistir (CNN +0.027, LGBM +0.026). "
          "Genel makro-F1 neredeyse degismemistir; bu, zero-shot modelin BBC akustik "
          "karakteristiklerini tam olarak yakalayamadigini ve domain-specific "
          "fine-tuning'in gerekliligi dogrulamaktadir.", st["govde"]),
    ]

    # FAD Tur 1
    if figler.get("fad_tur1"):
        icerik += [
            img(figler["fad_tur1"], 13*cm),
            P("Sekil 12. Tur 1 FAD (Frechet Audio Distance) skorlari — VGGish embedding. "
              "Dusuk FAD = sentetik sesler gercege daha yakin. acil_servis (43.4) "
              "en yuksek uzaklikla en zorlayici sinif olarak one cikmaktadir.",
              st["sekil_altyazi"]),
            Spacer(1, 0.3*cm),
        ]

    fad_satirlar = [["Sinif", "FAD (VGGish)", "Gercek Ornek", "Sentetik Klip", "Kalite"]]
    for sinif in ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]:
        if sinif in fad_d:
            v = fad_d[sinif]
            kalite = "Iyi" if v["fad"] < 20 else "Orta" if v["fad"] < 35 else "Zayif"
            fad_satirlar.append([sinif, f"{v['fad']:.2f}", str(v["n_gercek"]),
                                  str(v["n_sentetik"]), kalite])
    fad_satirlar.append(["ORTALAMA",
                          f"{np.mean([fad_d[s]['fad'] for s in fad_d]):.2f}",
                          "—", "—", "—"])
    t = Table(fad_satirlar, colWidths=[4.5*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.1*cm])
    t.setStyle(tablo_stili_temel(fad_satirlar))
    icerik += [
        P("Tablo 14. Tur 1 FAD Sonuclari (VGGish, Gercek vs Sentetik)", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        PageBreak(),
    ]

    # ── 5.4 TUR 2 VE TUR 3 ───────────────────────────────────────────────────
    icerik += [
        P("5.4 Tur 2 ve Tur 3 — Fine-tuned CLAP ile Uretim", st["alt_bolum"]),
        P("Fine-tuned CLAP checkpoint'i AudioLDM2 checkpoint'ine entegre edilerek "
          "iki farkli deneysel kosul olusturulmustur:", st["govde"]),
    ]

    tur23_satirlar = [
        ["Kosul", "CLAP", "Prompt", "Klip", "Cikti Klasoru", "Durum"],
        ["Tur 2", "BBC Fine-tuned\n(clap_bbc_best.pt, Epoch 1)", "Ingilizce\n(5 prompt/sinif)",
         f"200\n({len(tur2_wav)} uretildi)", "data/synthetic_tur2/", "TAMAMLANDI"],
        ["Tur 3", "BBC Fine-tuned\n(ayni checkpoint)", "Turkce\n(5 prompt/sinif — ilk kez!)",
         f"200\n({len(tur3_wav)} uretildi)", "data/synthetic_tur3/", "TAMAMLANDI"],
    ]
    t = Table(tur23_satirlar, colWidths=[1.5*cm, 3.5*cm, 3*cm, 2*cm, 3.5*cm, 2.5*cm])
    t.setStyle(tablo_stili_temel(tur23_satirlar))
    for ri, row in enumerate(tur23_satirlar):
        if "TAMAMLANDI" in str(row[-1]):
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (-1, ri), (-1, ri), colors.HexColor("#2E7D32")),
                ("FONTNAME",  (-1, ri), (-1, ri), "Helvetica-Bold"),
            ]))
    icerik += [
        P("Tablo 15. Tur 2 ve Tur 3 Uretim Ozeti", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        P("Tur 2 vs Tur 1 karsilastirmasi: Ayni 4 sinif, ayni prompt dili, farkli CLAP. "
          "Bu karsilastirma dogrudan fine-tuning'in uretim kalitesine katkisini izole eder. "
          "FAD(Tur2) < FAD(Tur1) ise fine-tuning BBC akustigini daha iyi modellemektedir.",
          st["govde"]),
        P("Tur 3 — Turkce Prompt Deneyi: BBC arsivi icin ilk kez Turkce metin-ses uretimi "
          "gerceklestirilmistir. fine-tuned CLAP, NT-Xent kaybi ile Turkce promptlari "
          "BBC ses embedding uzayina hizalamistir. Tur 2 vs Tur 3 FAD karsilastirmasi "
          "Turkce fine-tuning'in Ingilizce'ye gore daha iyi mi yoksa daha kotu mu "
          "uretim sagladigini ortaya koyacaktir.", st["govde"]),
        Spacer(1, 0.3*cm),
    ]

    # FAD 3 tur karsilastirma tablosu
    fad_tur2_yol = ROOT / "results" / "fad_sonuclar_tur2.json"
    fad_tur3_yol = ROOT / "results" / "fad_sonuclar_tur3.json"
    if fad_tur2_yol.exists() and fad_tur3_yol.exists():
        import json as _json
        fad2 = _json.loads(fad_tur2_yol.read_text(encoding="utf-8"))
        fad3 = _json.loads(fad_tur3_yol.read_text(encoding="utf-8"))
        siniflar_fad = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]

        f1v = [fad_d[s]["fad"]      for s in siniflar_fad]
        f2v = [fad2[s]["fad"]       for s in siniflar_fad]
        f3v = [fad3[s]["fad"]       for s in siniflar_fad]
        har_t4 = bool(fad_tur4_d)
        f4v = [fad_tur4_d[s]["fad"] for s in siniflar_fad] if har_t4 else []

        baslik_row = ["Sinif", "FAD Tur1\n(Pretrained+EN)", "FAD Tur2\n(Dongu1+EN)",
                      "Delta\nTur2-Tur1", "FAD Tur3\n(Dongu1+TR)", "Delta\nTur3-Tur1"]
        if har_t4:
            baslik_row += ["FAD Tur4\n(Dongu2+EN)", "Delta\nTur4-Tur2"]
        fad_uc_satirlar = [baslik_row]

        for i, s in enumerate(siniflar_fad):
            d2 = f2v[i] - f1v[i]; d3 = f3v[i] - f1v[i]
            row = [s, f"{f1v[i]:.2f}", f"{f2v[i]:.2f}", f"{d2:+.2f}",
                      f"{f3v[i]:.2f}", f"{d3:+.2f}"]
            if har_t4:
                d4 = f4v[i] - f2v[i]
                row += [f"{f4v[i]:.2f}", f"{d4:+.2f}"]
            fad_uc_satirlar.append(row)

        ort_row = ["ORTALAMA",
                   f"{np.mean(f1v):.2f}", f"{np.mean(f2v):.2f}",
                   f"{np.mean(f2v)-np.mean(f1v):+.2f}",
                   f"{np.mean(f3v):.2f}", f"{np.mean(f3v)-np.mean(f1v):+.2f}"]
        if har_t4:
            ort_row += [f"{np.mean(f4v):.2f}", f"{np.mean(f4v)-np.mean(f2v):+.2f}"]
        fad_uc_satirlar.append(ort_row)

        col_w = [3.5*cm, 2.1*cm, 2.1*cm, 1.8*cm, 2.1*cm, 1.8*cm]
        if har_t4:
            col_w += [2.1*cm, 1.8*cm]
        t_fad = Table(fad_uc_satirlar, colWidths=col_w)
        t_fad.setStyle(tablo_stili_temel(fad_uc_satirlar))
        delta_cols = [3, 5, 7] if har_t4 else [3, 5]
        for ri, row in enumerate(fad_uc_satirlar[1:], start=1):
            for ci in delta_cols:
                if ci < len(row):
                    val = row[ci]
                    if val.startswith("-"):
                        t_fad.setStyle(TableStyle([("TEXTCOLOR",(ci,ri),(ci,ri),colors.HexColor("#2E7D32")),
                                                   ("FONTNAME",(ci,ri),(ci,ri),"Helvetica-Bold")]))
                    elif val.startswith("+"):
                        t_fad.setStyle(TableStyle([("TEXTCOLOR",(ci,ri),(ci,ri),colors.HexColor("#C62828"))]))
        t_fad.setStyle(TableStyle([
            ("BACKGROUND",(0,-1),(-1,-1),colors.HexColor("#BBDEFB")),
            ("FONTNAME",(0,-1),(-1,-1),"Helvetica-Bold"),
        ]))

        icerik += [
            P("Tablo 16. FAD Karsilastirmasi — 4 Tur Tam Sonuclar (VGGish)", st["sekil_altyazi"]),
            t_fad, Spacer(1, 0.3*cm),
        ]

        if figler.get("fad_uc_tur"):
            icerik += [
                img(figler["fad_uc_tur"], 15*cm),
                P("Sekil 13. Tur 1 / 2 / 3 / 4 FAD karsilastirmasi (DETERMINISTIK olcum). "
                  "Sol: sinif bazli gruplu bar grafigi. Sag: ortalama FAD ve Tur1'e gore delta. "
                  "Ingilizce promptlu turlar (Tur1=13.64, Tur2=13.59, Tur4=14.00) neredeyse "
                  "OZDESTIR: CLAP ince ayar dongusu ortalama uretim kalitesini degistirmemektedir. "
                  "Turkce promptlu Tur3 (15.90) farkli bir sinif-bazli desen sergiler.",
                  st["sekil_altyazi"]),
                Spacer(1, 0.2*cm),
            ]

        icerik += [
            P("FAD bulgulari (DETERMINISTIK olcum — onceki rapordaki '%19 iyilesme' "
              "bulgusu, uretim cesitlilik hatasi ve FAD non-determinizmi duzeltilmeden "
              "once olculmus bir ARTEFAKTTI). Duzeltme sonrasi: Ingilizce promptlu uc tur "
              "neredeyse ozdes ortalama FAD vermektedir — Tur1 (pretrained) 13.64, "
              "Tur2 (Dongu1) 13.59, Tur4 (Dongu2) 14.00. Yani CLAP ince ayar dongusu "
              "ortalama uretim kalitesini ISTATISTIKSEL OLARAK DEGISTIRMEMEKTEDIR; "
              "pretrained ile iki kez ince ayarli model arasinda anlamli fark yoktur.",
              st["govde"]),
            P("Turkce prompt deneyi (Tur3): Turkce promptlardan uretilen seslerin ortalama "
              "FAD'i 15.90'dir; Ingilizce turlardan biraz yuksek olmakla birlikte sinif "
              "bazinda FARKLI bir desen sergiler (acil_servis cok daha iyi: 10.2; "
              "insaat cok daha kotu: 21.1). Bu, prompt dilinin ortalama kaliteden cok "
              "per-sinif uretim karakterini etkiledigini gostermektedir.", st["govde"]),
            P("Donguler arasi yorum: FAD olceginde turler arasi ilerleme yoktur "
              "(Tur1≈Tur2≈Tur4). Ancak FAD'in bir KISITI oldugunu vurgulamak gerekir: "
              "asagida (Bolum 5.5) gosterilecegi uzere, FAD'a gore ozdes olan bu turlar, "
              "AKUSTIK OLARAK TUTARLI hedef siniflarda gorev faydasi acisindan belirgin "
              "sekilde AYRISMAKTADIR (en guclu kazanim Dongu2/Tur4'tendir). Yani FAD, "
              "gorev faydasinin guvenilir bir vekili degildir.", st["govde"]),
            PageBreak(),
        ]
    else:
        icerik += [PageBreak()]

    # ── BOLUM 5.6: HEDEF AKUSTIK TUTARLILIGI (ANA POZITIF BULGU) ──────────────
    coh_dir = ROOT / "outputs" / "report" / "coherence_figs"
    # Tezin son hali: secim olcutu + McNemar tablolari (makale ile uyumlu)
    secim_satirlar = [
        ["Hedef", "Secim olcutu", "Taban F1", "Ort. Delta F1"],
        ["insaat sesleri",  "salt dusuk-F1",   "0.388", "-0.007"],
        ["dini mekan",      "salt dusuk-F1",   "0.461", "-0.018"],
        ["acil servis",     "salt dusuk-F1",   "0.589", "-0.008"],
        ["diger hayvanlar", "salt dusuk-F1",   "0.582", "-0.016"],
        ["kopek sesleri",   "akustik tutarli", "0.665", "+0.019"],
        ["kus sesleri",     "akustik tutarli", "0.591", "+0.050"],
        ["hava olaylari",   "akustik tutarli", "0.553", "+0.034"],
    ]
    t_secim = Table(secim_satirlar, colWidths=[3.6*cm, 3.6*cm, 2.4*cm, 2.8*cm])
    t_secim.setStyle(tablo_stili_temel(secim_satirlar))
    for ri in range(1, 5):
        t_secim.setStyle(TableStyle([("TEXTCOLOR",(3,ri),(3,ri),colors.HexColor("#C62828")),
                                     ("FONTNAME",(3,ri),(3,ri),"Helvetica-Bold")]))
    for ri in range(5, 8):
        t_secim.setStyle(TableStyle([("TEXTCOLOR",(3,ri),(3,ri),colors.HexColor("#1B5E20")),
                                     ("FONTNAME",(3,ri),(3,ri),"Helvetica-Bold"),
                                     ("BACKGROUND",(0,ri),(-1,ri),colors.HexColor("#F1F8E9"))]))
    mcn2_satirlar = [
        ["Hedef / Kosu", "n", "chi2", "p", "Anlam"],
        ["Ilk-4 hedef (heterojen, Dongu1)", "48.144", "2.34", "0.126", "ns"],
        ["Tutarli-3 (H1, pretrained)",      "2.343", "25.5", "<1e-6",  "***"],
        ["Tutarli-3 (coh2, Dongu1 EN)",     "2.343", "20.6", "<1e-5",  "***"],
        ["Tutarli-3 (coh3, Dongu1 TR)",     "2.343", "23.1", "<1e-5",  "***"],
        ["Tutarli-3 (coh4, Dongu2 EN)",     "2.343", "66.4", "3.3e-16","***"],
        ["Catch-all (diger hayv., coh4)",   "754",   "1.27", "0.261",  "ns"],
    ]
    t_mcn = Table(mcn2_satirlar, colWidths=[6.2*cm, 1.7*cm, 1.5*cm, 2.0*cm, 1.3*cm])
    t_mcn.setStyle(tablo_stili_temel(mcn2_satirlar))
    for ri in range(1, len(mcn2_satirlar)):
        if "***" in mcn2_satirlar[ri][-1]:
            t_mcn.setStyle(TableStyle([("TEXTCOLOR",(-1,ri),(-1,ri),colors.HexColor("#1B5E20")),
                                       ("FONTNAME",(-1,ri),(-1,ri),"Helvetica-Bold"),
                                       ("BACKGROUND",(0,ri),(-1,ri),colors.HexColor("#F1F8E9"))]))
    icerik += [
        P("5.5 Hedef Akustik Tutarliligi — Ana Bulgu", st["alt_bolum"]),
        P("Yukaridaki bolumler, ILK secilen dort hedefte (insaat, dini mekan, acil servis, "
          "diger hayvanlar) augmentasyonun anlamli kazanim saglamadigini gostermistir. "
          "Ancak ayrintili coozumleme, bu basarisizligin augmentasyonun kendisinden degil "
          "HEDEF SECIMINDEN kaynaklandigini ortaya koymustur: ilk hedeflerden 'diger "
          "hayvanlar' akustik olarak HETEROJEN bir cop kutusu (catch-all) sinifidir ve "
          "digerleri de genis, ic-cesitliligi yuksek siniflardir. Sinanan hipotez: "
          "uretici artirma, yalnizca hedef sinif AKUSTIK OLARAK TUTARLI oldugunda fayda "
          "saglar.", st["govde"]),
        P("Tasarim: Hedefleri akustik olarak tutarli uc dusuk-kaynakli sinifla "
          "(kopek sesleri, kus sesleri, hava olaylari) degistirdik; heterojen 'diger "
          "hayvanlar' sinifini ayni egitim icinde KONTROL olarak tuttuk. Deney, uretici "
          "model/prompt ekseninde dort bagimsiz konfigurasyonda tekrarlandi: "
          "H1 (pretrained+EN), coh2 (Dongu1+EN), coh3 (Dongu1+TR), coh4 (Dongu2+EN).",
          st["govde"]),
        P("Tablo 17. Hedef secim olcutunun etkisi. Ort. Delta F1, dort uretim "
          "konfigurasyonu uzerinden ortalama CNN sinif-F1 degisimidir. Salt dusuk-F1 "
          "ile secilen ilk dort hedefin DORDU DE negatif; akustik tutarlilik gozetilerek "
          "secilen uc hedefin UCU DE pozitiftir.", st["sekil_altyazi"]),
        t_secim, Spacer(1, 0.35*cm),
        img(str(coh_dir / "k1_replikasyon_4konfig.png"), 15.5*cm),
        P("Sekil 14. Sinif bazli F1 degisimi, dort uretim konfigurasyonu. Akustik olarak "
          "tutarli hedefler (yesil alan) dort kosuda da iyilesmekte; heterojen catch-all "
          "kontrol (kirmizi alan) iyilesmemektedir. Ortalama kazanim: kus +0.050, "
          "hava +0.034, kopek +0.019; catch-all -0.011.", st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        img(str(coh_dir / "k3_mekanizma.png"), 13*cm),
        P("Sekil 15. Mekanizma: tutarli-3 ortalamasi (yesil) her kosuda pozitif, catch-all "
          "kontrol (kirmizi) her kosuda yaklasik 0 veya negatif. Ayni uretim hatti, ayni "
          "egitim protokolu ve ayni klip sayisiyla, belirleyici degisken HEDEFIN YAPISIDIR.",
          st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        img(str(coh_dir / "k2_mcnemar.png"), 16*cm),
        P("Sekil 16. Istatistiksel anlamlilik. Sol: tutarli-3 McNemar chi-kare, dort kosu "
          "da ***. Sag: sinif x kosu anlamlilik isi haritasi (-log10 p). Kus ve hava "
          "istikrarli biçimde anlamli; catch-all kontrol hicbir kosuda LEHTE anlamli degil.",
          st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        P("Tablo 18. McNemar testi (Yates duzeltmeli, df=1). Tutarli hedefler dort "
          "konfigurasyonda da anlamli; heterojen ilk hedefler ve catch-all kontrol anlamsiz.",
          st["sekil_altyazi"]),
        t_mcn,
        P("*** p<0.001; ns: anlamli degil. Tutarli-3: n=2.343 test segmenti.", st["kucuk"]),
        Spacer(1, 0.35*cm),
        img(str(coh_dir / "k7_tutarlilik_kavram.png"), 13.5*cm),
        P("Sekil 17. Akustik tutarlilik kavrami: gercek BBC kliplerinin VGGish "
          "gommelerinin 2B izdusumu (PCA, birim-normlu). Tutarli sinif (kopek, yesil) "
          "siki bir kume; catch-all (diger hayvanlar, kirmizi) ve daginik sinif "
          "(dini mekan, turuncu) genis bulut olusturur. Elipsler 2-sigma kovaryans; "
          "D_c degerleri sinif-ici kosinus dagilimidir (dusuk = tutarli).",
          st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        img(str(coh_dir / "k5_tutarlilik_26.png"), 13.5*cm),
        P("Sekil 18. 26 sinifin VGGish sinif-ici dagilima (akustik tutarlilik) gore "
          "siralamasi. Tutarli hedefler (yesil) en tutarli ucta; ilk heterojen hedefler "
          "(kirmizi) daha daginik taraftadir. Olcut, kazanimdan BAGIMSIZ olarak yalnizca "
          "gercek seslerden hesaplanmistir (cherry-pick savunmasi).", st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        img(str(coh_dir / "k4_korelasyon.png"), 16*cm),
        P("Sekil 19. Nicel dogrulama (ayni VGGish uzayinda): (sol) sinif-ici dagilim "
          "augmentasyon kazanimini anlamli ongorur (Pearson r=-0.76, p=0.049; Spearman "
          "rho=-0.79, p=0.036); (sag) FAD kazanimi ongormez (r=+0.30, ns). NOT: korelasyon "
          "gomme-uzayina bagimlidir (CLAP uzayinda anlamli degil); cekirdek kanit, "
          "metrikten bagimsiz McNemar replikasyonudur.", st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        img(str(coh_dir / "k6_specaug.png"), 13.5*cm),
        P("Sekil 20. Ilke augmentasyon turunden bagimsizdir: geleneksel SpecAugment "
          "(global maskeleme) de tutarli siniflari iyilestirir, catch-all'i iyilestirmez "
          "(kotulestirir). Iki yontem hedef siniflarda istatistiksel olarak ayirt "
          "edilemez (McNemar p=0.81); birlestirme hedefte fayda katmaz (p=0.017 kotulesme).",
          st["sekil_altyazi"]),
        Spacer(1, 0.25*cm),
        P("Bulgu ozeti: (1) Uc tutarli sinif dort konfigurasyonun da hepsinde pozitiftir; "
          "tutarli-3 birlesik McNemar dort kosuda da p<10^-5 (***), Holm-Bonferroni "
          "sonrasi da *** kalir; etki buyuklugu OR 1.75-2.69, Cohen g 0.14-0.23. Kus icin "
          "doruluk 0.495'ten 0.601'e (+10.6 puan) yukselmistir. (2) Akustik tutarlilik "
          "NICEL olarak dogrulanmistir: VGGish sinif-ici dagilim kazanimi ongorur "
          "(r=-0.76, p=0.049); ayni uzayda FAD ongormez (r=+0.30, ns). En guclu gorev "
          "kazanimi, FAD'a gore Tur1 ile OZDES olan Dongu2/coh4'tendir. (3) Catch-all "
          "kontrol hicbir kosuda fayda gormez. (4) Ilke augmentasyon turunden "
          "BAGIMSIZDIR: SpecAugment ayni deseni gosterir ve hedeflerde uretici artirmayla "
          "esdegerdir (p=0.81). Sonuc: hedef akustik tutarliligi, ses veri artirmanin "
          "fayda ON KOSULUDUR. Bu, literaturdeki celiskili (kimi olumlu, kimi olumsuz) "
          "sonuclara birlestirici bir aciklama sunar.", st["govde"]),
        PageBreak(),
    ]

    # ── BOLUM 6: TARTISMA ─────────────────────────────────────────────────────
    icerik += [
        P("6. Tartisma", st["bolum"]),
        P("6.1 CNN'in Ustunlugu", st["alt_bolum"]),
        P("ResNet-18 tabanli CNN, her model ailesinde acik ara lider konumundadir. "
          "Bu sonuc birkac gozlemle aciklanabilir: (1) konvolusyonel mimari, "
          "spektrogramdaki yerel zaman-frekans kaliplari icin induktif bir on-bilgi "
          "saglar; (2) ImageNet on-egitiminden aktarilan dusuk-seviye ozellik detektoru "
          "agirliklar, ses veriyle ince ayar icin guclü bir baslangic noktasi olusturur; "
          "(3) spektrogramlar goruntulerle yeterince benzer oldugunden transfer learning "
          "etkili bir sekilde calismaktadir.", st["govde"]),
        P("6.2 AST Transformerin Overfitting Davranisi", st["alt_bolum"]),
        P("85.5M parametreli AST modeli, 9 epoch sonrasinda val loss artisina bagli "
          "overfitting sergiledi. Ozellikle dikkat cekici olan, train accuracy'nin "
          "epoch 7'de >%98'e ulasmisken val F1'in 0.60'ta plateolar yapmasidir. "
          "Bu durum, modelin egitim kumesini ezberledigi fakat genelleme yapamadigini "
          "gostermektedir. AST'nin potansiyelinden tam olarak faydalanmak icin "
          "SpecAugment, dropout artisi (0.2'den 0.4'e) ve gradual layer unfreezing "
          "gibi regularizasyon stratejilerinin denenmesi onerilmektedir.", st["govde"]),
        P("6.3 LightGBM'in Rekabetci Performansi", st["alt_bolum"]),
        P("El-yapimi 467-boyutlu ozellikler uzerinde calisan LightGBM'in "
          "0.5857 test F1 ile derin ogrenme modellerine rekabetci yaklasabilmesi "
          "dikkat cekicidir. Bu sonuc; veri miktarinin sinirli oldugu, hesaplama "
          "butcesinin kisitlandigi ya da yorumlanabilirlik gereksinimlerinin on "
          "planda tutuldugu durumlarda geleneksel ML'nin gecerliliğini dogrular. "
          "Gelecekte bu iki paradigmanin hibridlenmesi — ornegin CNN'in gizli "
          "katman ciktilarinin LightGBM'e girdi olarak verilmesi — arastirma "
          "konusu olabilir.", st["govde"]),
        Spacer(1, 0.2*cm),

        P("7. Sonuc ve Sonraki Adimlar", st["bolum"]),
        P("Bu calismada BBC Ses Arsivi uzerinde kapsamli bir karsilastirmali analiz "
          "gerceklestirilmistir. Temel bulgular soyle ozetlenebilir:", st["govde"]),
    ]

    sonuc_maddeleri = [
        ["1", "CNN (ResNet-18), Test Macro-F1=0.6716 ile en yuksek bireysel performansi gostermistir."],
        ["2", "LightGBM+Optuna, klasik ML'de en iyi skor olan 0.5857'ye ulasmistir."],
        ["3", "Late fusion, CNN'e istatistiksel olarak anlamli bir katki saglamadi (delta=+0.005)."],
        ["4", "Oracle analizi (0.7812) model cesitliligine ihtiyac oldugunu ortaya koymustur."],
        ["5", "insaat_sesleri (F1=0.39) ve dini_mekan (F1=0.46), uretici artirma icin oncelikli hedeflerdir."],
        ["6", "Tur 1 (zero-shot AudioLDM2): insaat_sesleri CNN +0.027, LightGBM +0.026 F1 iyilesmesi. "
              "Genel makro-F1 neredeyse degismedi; domain-specific fine-tuning'in gerekliligi dogrulandi."],
        ["7", "CLAP BBC fine-tuning (5 epoch, 26 saat): NT-Xent kaybi stabilize oldu, kNN=0.056 "
              "ile anlamli metin-ses hizalamasi kazanildi. 15K+ ciftlik retrieval icin beklenen durum."],
        ["8", "Tur 2 (fine-tuned CLAP + EN) ve Tur 3 (fine-tuned CLAP + TR) uretimi tamamlandi: "
              "toplam 600 sentetik klip. Turkce prompt ile BBC uretimi calismada ilk kez denendi."],
        ["9", "FAD (DETERMINISTIK): Ingilizce turlar neredeyse ozdes — Tur1=13.64, Tur2=13.59, "
              "Tur4=14.00; Tur3 (TR)=15.90. CLAP ince ayar dongusu ortalama uretim kalitesini "
              "DEGISTIRMEMEKTEDIR. Onceki '%19 iyilesme' bulgusu, uretim cesitlilik hatasi ve "
              "FAD non-determinizmi duzeltilmeden olculmus bir artefaktti."],
        ["10", "ILK 4 hedefte (insaat/dini/acil/diger hayvanlar) augmentasyon anlamli kazanim "
               "saglamadi (McNemar ns). Ayrintili coozumleme, bunun augmentasyondan degil "
               "HEDEF SECIMINDEN kaynaklandigini gosterdi: hedeflerden biri akustik olarak "
               "heterojen bir 'cop kutusu' (catch-all) sinifiydi."],
        ["11", "ANA BULGU — Hedef akustik tutarliligi: Hedefler akustik olarak TUTARLI uc "
               "sinifa (kopek/kus/hava) yoneltildiginde augmentasyon ANLAMLI ve dort uretim "
               "konfigurasyonunda TEKRARLANABILIR bir kazanim saglar. Tutarli-3 birlesik "
               "McNemar dort kosuda da p<10^-5 (***); kus dorulugu +10.6 puan. Catch-all "
               "kontrol hicbir kosuda fayda gormez. Akustik tutarlilik, faydanin ON KOSULUDUR."],
        ["12", "FAD gorev faydasini ONGORMEZ: En guclu gorev kazanimi, FAD'a gore Tur1 ile "
               "OZDES olan Dongu2/coh4'tendir (kus +0.078, hava +0.052). FAD'in ayirt edemedigi "
               "iki uretici, gorev faydasinda belirgin ayrisir. Iteratif ince ayar dongusu, "
               "TUTARLI hedefler soz konusu oldugunda uretimi gercekten daha faydali kilar."],
    ]
    t = Table(sonuc_maddeleri, colWidths=[0.8*cm, 14.2*cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    icerik += [t, Spacer(1, 0.4*cm)]

    sonraki_satirlar = [
        ["Asama", "Gorev", "Durum"],
        ["1.1-1.3", "Klasik ML + LightGBM + HPO", "TAMAMLANDI"],
        ["1.4",     "Late Fusion (CNN + BiLSTM + AST)", "TAMAMLANDI"],
        ["2.1",     "BBC dosya adi cevirileri + ses_metin_eslestirme.parquet", "TAMAMLANDI"],
        ["2.2",     "26 sinif x 5 TR/EN prompt + AudioLDM2 kurulum", "TAMAMLANDI"],
        ["3.1",     "Tur 1: Zero-shot uretim — 4 sinif x 50 klip", "TAMAMLANDI"],
        ["3.2",     "Tur 1: CNN+LightGBM augmented egitim + F1 karsilastirma", "TAMAMLANDI"],
        ["3.3",     "FAD Tur 1 olcumu — VGGish (gercek vs sentetik)", "TAMAMLANDI"],
        ["4.1",     "CLAP fine-tuning — 15.356 BBC ses-metin cifti, 5 epoch, 26 saat", "TAMAMLANDI"],
        ["4.2",     "CLAP → AudioLDM2 checkpoint entegrasyonu (09_integrate_clap.py)", "TAMAMLANDI"],
        ["4.3",     "Tur 2: Fine-tuned CLAP + Ingilizce — 200 klip uretimi", "TAMAMLANDI"],
        ["4.4",     "Tur 3: Fine-tuned CLAP + Turkce — 200 klip uretimi (ilk!)", "TAMAMLANDI"],
        ["5.1",     "FAD Tur 2 + Tur 3 olcumu + Tur1 vs Tur2 vs Tur3 karsilastirma", "TAMAMLANDI"],
        ["5.2",     "Tur 2/3 CNN+LightGBM augmented egitim + siniflandirma etkisi", "TAMAMLANDI"],
        ["5.3",     "McNemar istatistiksel anlamlilik testleri", "TAMAMLANDI"],
        ["6.1",     "Iteratif Dongu 2 (CLAP dongu2): BBC + Tur2 sentetik ile CLAP yeniden fine-tune (8.4 saat, kNN=4.5%)", "TAMAMLANDI"],
        ["6.2",     "CLAP dongu2 → AudioLDM2 entegrasyonu (audioldm2_clap_dongu2_finetuned.pth, 8.50 GB)", "TAMAMLANDI"],
        ["6.3",     "Tur 4: dongu2 CLAP + Ingilizce — 4 sinif x 50 klip uretimi (200 klip)", "TAMAMLANDI"],
        ["6.4",     "FAD Tur 4 olcumu — VGGish (deterministik). NOT: FAD gorev faydasini ongormez — coh4/Dongu2 en yuksek gorev kazanimini saglar (kus +0.078)", "TAMAMLANDI"],
        ["7.1",     "Iteratif loop Tur4: CNN+LGBM augmented egitim (CNN +0.005, insaat +0.071, dini +0.080)", "TAMAMLANDI"],
        ["7.2",     "Tez yazimi — Bolum 3 (Yontem) ve Bolum 4 (Deneyler)", "BEKLIYOR"],
    ]
    t = Table(sonraki_satirlar, colWidths=[1.8*cm, 10.7*cm, 3*cm])
    t.setStyle(tablo_stili_temel(sonraki_satirlar))
    for row_idx, row in enumerate(sonraki_satirlar):
        if row[-1] == "TAMAMLANDI":
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (-1, row_idx), (-1, row_idx), colors.HexColor("#2E7D32")),
                ("FONTNAME",  (-1, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
        elif row[-1] == "DEVAM EDIYOR":
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#FFF9C4")),
                ("TEXTCOLOR",  (-1, row_idx), (-1, row_idx), colors.HexColor("#E65100")),
                ("FONTNAME",   (-1, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
        elif row[-1] == "BEKLIYOR":
            t.setStyle(TableStyle([
                ("TEXTCOLOR",  (-1, row_idx), (-1, row_idx), colors.HexColor("#1565C0")),
            ]))
        elif row[-1] == "ARASTIRMA":
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F3E5F5")),
                ("TEXTCOLOR",  (-1, row_idx), (-1, row_idx), colors.HexColor("#6A1B9A")),
                ("FONTNAME",   (-1, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
    icerik += [
        P("Tablo 19. Arastirma Ilerleme Tablosu (Tamamlanan + Planlanan Adimlar)", st["sekil_altyazi"]),
        t, Spacer(1, 0.5*cm),
        P("<b>Kod ve Veri Erisilebilirligi.</b> BBC Ses Efektleri Arsivi kamuya "
          "aciktir. Uretim ve egitim kodu, uretilen sentetik klipler, egitilmis "
          "model checkpoint'leri ve tum sonuc dosyalari (FAD, McNemar ve "
          "tutarlilik olcumleri) makul istek uzerine yazisma yazarindan temin "
          "edilebilir. Tum deneyler tek bir RTX 3060 (12 GB) uzerinde, sabit "
          "tohumla (seed=42) deterministik bicimde yurutulmustur.", st["govde"]),
        Spacer(1, 0.3*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#B0BEC5")),
        Spacer(1, 0.2*cm),
        P("Bu rapor otomatik olarak uretilmistir. "
          f"Uretim tarihi: {tarih}  |  Veri: BBC Sound Archive  |  "
          "26 sinif, 15.356 WAV, 331.785 segment",
          st["kucuk"]),
    ]

    return icerik


# ═════════════════════════════════════════════════════════════════════════════
# SAYFA NUMARASI (Header/Footer)
# ═════════════════════════════════════════════════════════════════════════════

class NumaralandirmaliSayfa(BaseDocTemplate):
    def __init__(self, dosya, **kw):
        super().__init__(dosya, **kw)

    def afterPage(self):
        pass

    def handle_pageEnd(self):
        c = self.canv
        c.saveState()
        # Header
        c.setStrokeColor(colors.HexColor("#0D47A1"))
        c.setLineWidth(0.8)
        c.line(2*cm, H - 1.5*cm, W - 2*cm, H - 1.5*cm)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#37474F"))
        c.drawString(2*cm, H - 1.2*cm, "BBC Ses Arsivi Siniflandirma Raporu")
        c.drawRightString(W - 2*cm, H - 1.2*cm, "Doktora Tezi Ara Raporu")
        # Footer
        c.line(2*cm, 1.5*cm, W - 2*cm, 1.5*cm)
        c.drawCentredString(W/2, 1.0*cm, f"Sayfa {self.page}")
        c.restoreState()
        super().handle_pageEnd()


# ═════════════════════════════════════════════════════════════════════════════
# ANA
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("\nFigurler uretiliyor...")
    figler = {
        "sinif_dagilimi":      fig_sinif_dagilimi(),
        "model_secim":         fig_model_secim(),
        "egitim_sureci":       fig_egitim_sureci(),
        "genel_karsilastirma": fig_genel_karsilastirma(),
        "sinif_heatmap":       fig_sinif_f1_heatmap(),
        "fusion":              fig_fusion(),
        "cnn_sinif_cubuk":     fig_cnn_sinif_cubuk(),
        "pipeline":            fig_pipeline(),
        "fad_tur1":            fig_fad_tur1(),
        "fad_uc_tur":          fig_fad_uc_tur(),
        "clap_egitim":         fig_clap_egitim(),
        "tur1_augmentasyon":   fig_tur_ozet(),
        "flywheel_sema":       fig_flywheel_sema(),
        "sinif_tur_karsilastirma": fig_siniflandirma_tur_karsilastirma(),
        "makro_f1_ozet":       fig_makro_f1_ozet(),
    }
    print(f"  {len(figler)} figur uretildi -> {FIG_DIR}")

    print("\nPDF derleniyor...")
    st = stiller_olustur()

    doc = NumaralandirmaliSayfa(
        str(PDF_YOLU),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
        title="BBC Sound Archive Classification Report",
        author="PhD Researcher",
    )
    cerceve = Frame(doc.leftMargin, doc.bottomMargin,
                    doc.width, doc.height, id="normal")
    sablon  = PageTemplate(id="main", frames=[cerceve])
    doc.addPageTemplates([sablon])

    icerik = belge_olustur(figler, st)
    doc.build(icerik)

    print(f"\nPDF olusturuldu: {PDF_YOLU}")
    print(f"Boyut: {PDF_YOLU.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
