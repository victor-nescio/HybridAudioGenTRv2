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

# ── Sabitler ──────────────────────────────────────────────────────────────────
OUT     = ROOT / "outputs" / "report"
FIG_DIR = OUT / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_YOLU = OUT / "BBC_SoundClassification_Report.pdf"

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
        "Bu rapor, BBC Ses Arsivi uzerinde gerceklestirilen kapsamli ses siniflandirma "
        "deneylerinin bulgularini sunmaktadir. 15.356 WAV dosyasi ve 26 sinifi kapsayan "
        "veri seti uzerinde; 6 geleneksel makine ogrenimi algoritmasi, 3 derin ogrenme "
        "modeli (CNN, BiLSTM, AST Transformer) ve 4 farkli geç füzyon yaklasimi "
        "sistematik olarak degerlendirilmistir. En yuksek bireysel performans, "
        "pretrained ResNet-18 tabanli CNN modeli ile Test Macro-F1 = 0.6716 olarak elde "
        "edilmistir. Geleneksel ML'de LightGBM, Optuna HPO ile Test Macro-F1 = 0.5857 "
        "degerine ulasarak en iyi klasik model olmustur. Geç füzyon analizinde Oracle "
        "ust sinirinin 0.7812 oldugu saptanmis; model ciktilarinin yuksek korelasyonu "
        "nedeniyle Val-F1 agirlikli füzyon bireysel CNN performansini gecememistir."
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
        ["Toplam Segment (5 sn)", "312.725"],
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
          "dilimlere bolunmus ve toplamda 312.725 egitim segmenti elde edilmistir.",
          st["govde"]),
        P("Siniflandirma problemi icin dosya-duzeyli stratified train/val/test bolumleme "
          "uygulanmistir: %70 egitim, %15 dogrulama, %15 test. Bu yaklasim, ayni dosyadan "
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
        P("Tablo 3. Derin Ogrenme Mimarileri", st["sekil_altyazi"]),
        t, Spacer(1, 0.3*cm),
        P("Tum derin ogrenme modelleri icin ortak egitim protokolu: AdamW optimizer "
          "(weight_decay=1e-4), CosineAnnealingLR (eta_min=lr/100), label_smoothing=0.1, "
          "patience=7 erken durdurma ve AMP (Automatic Mixed Precision) kullanilmistir. "
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
        P("Tablo 4. Tam Model Karsilastirma Tablosu — * alt-orneklem, † erken durduruldu",
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
        P("Tablo 5. Late Fusion Sonuc Tablosu", st["sekil_altyazi"]),
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

    # ── BOLUM 5: TARTISMA ─────────────────────────────────────────────────────
    icerik += [
        P("5. Tartisma", st["bolum"]),
        P("5.1 CNN'in Ustunlugu", st["alt_bolum"]),
        P("ResNet-18 tabanli CNN, her model ailesinde acik ara lider konumundadir. "
          "Bu sonuc birkac gozlemle aciklanabilir: (1) konvolusyonel mimari, "
          "spektrogramdaki yerel zaman-frekans kaliplari icin induktif bir on-bilgi "
          "saglar; (2) ImageNet on-egitiminden aktarilan dusuk-seviye ozellik detektoru "
          "agirliklar, ses veriyle ince ayar icin guclü bir baslangic noktasi olusturur; "
          "(3) spektrogramlar goruntulerle yeterince benzer oldugunden transfer learning "
          "etkili bir sekilde calismaktadir.", st["govde"]),
        P("5.2 AST Transformerin Overfitting Davranisi", st["alt_bolum"]),
        P("85.5M parametreli AST modeli, 9 epoch sonrasinda val loss artisina bagli "
          "overfitting sergiledi. Ozellikle dikkat cekici olan, train accuracy'nin "
          "epoch 7'de >%98'e ulasmisken val F1'in 0.60'ta plateolar yapmasidir. "
          "Bu durum, modelin egitim kumesini ezberledigi fakat genelleme yapamadigini "
          "gostermektedir. AST'nin potansiyelinden tam olarak faydalanmak icin "
          "SpecAugment, dropout artisi (0.2'den 0.4'e) ve gradual layer unfreezing "
          "gibi regularizasyon stratejilerinin denenmesi onerilmektedir.", st["govde"]),
        P("5.3 LightGBM'in Rekabetci Performansi", st["alt_bolum"]),
        P("El-yapimi 467-boyutlu ozellikler uzerinde calisan LightGBM'in "
          "0.5857 test F1 ile derin ogrenme modellerine rekabetci yaklasabilmesi "
          "dikkat cekicidir. Bu sonuc; veri miktarinin sinirli oldugu, hesaplama "
          "butcesinin kisitlandigi ya da yorumlanabilirlik gereksinimlerinin on "
          "planda tutuldugu durumlarda geleneksel ML'nin gecerliliğini dogrular. "
          "Gelecekte bu iki paradigmanin hibridlenmesi — ornegin CNN'in gizli "
          "katman ciktilarinin LightGBM'e girdi olarak verilmesi — arastirma "
          "konusu olabilir.", st["govde"]),
        Spacer(1, 0.2*cm),

        P("6. Sonuc ve Sonraki Adimlar", st["bolum"]),
        P("Bu calismada BBC Ses Arsivi uzerinde kapsamli bir karsilastirmali analiz "
          "gerceklestirilmistir. Temel bulgular soyle ozetlenebilir:", st["govde"]),
    ]

    sonuc_maddeleri = [
        ["1", "CNN (ResNet-18), Test Macro-F1=0.6716 ile en yuksek bireysel performansi gostermistir."],
        ["2", "LightGBM+Optuna, klasik ML'de en iyi skor olan 0.5857'ye ulasmistir."],
        ["3", "Late fusion, CNN'e istatistiksel olarak anlamli bir katki saglamadi (delta=+0.005)."],
        ["4", "Oracle analizi (0.7812) model cesitliligine ihtiyac oldugunu ortaya koymustur."],
        ["5", "insaat_sesleri (F1=0.39) ve dini_mekan (F1=0.46), uretici artirma icin oncelikli hedeflerdir."],
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
        ["Asama", "Gorev", "Oncelik"],
        ["1.4", "Late Fusion (tamamlandi)", "BITTI"],
        ["2.1", "26 sinif x 5 Turkce prompt listesi (tamamlandi)", "BITTI"],
        ["3.1", "AudioLDM kurulum + ilk uretim deneyi (50 klip/sinif)", "YUKSEK"],
        ["3.2", "FAD olcumu — VGGish + CLAP karsilastirmali", "YUKSEK"],
        ["3.3", "Iteratif artirma Tur 1: F1<0.5 siniflar", "YUKSEK"],
        ["3.4", "AudioLDM Turkce CLAP fine-tuning", "ORTA"],
        ["3.5", "Iteratif artirma Tur 2: fine-tuned model ile", "ORTA"],
        ["4.1", "McNemar istatistiksel anlamlilik testleri", "DUSUK"],
        ["4.2", "Grad-CAM ve attention gorselleştirmesi", "DUSUK"],
    ]
    t = Table(sonraki_satirlar, colWidths=[1.5*cm, 10*cm, 3.5*cm])
    t.setStyle(tablo_stili_temel(sonraki_satirlar))
    for row_idx, row in enumerate(sonraki_satirlar):
        if row[-1] == "BITTI":
            t.setStyle(TableStyle([
                ("TEXTCOLOR", (-1, row_idx), (-1, row_idx), colors.HexColor("#2E7D32")),
                ("FONTNAME",  (-1, row_idx), (-1, row_idx), "Helvetica-Bold"),
            ]))
    icerik += [
        P("Tablo 6. Sonraki Arastirma Adimlari", st["sekil_altyazi"]),
        t, Spacer(1, 0.5*cm),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#B0BEC5")),
        Spacer(1, 0.2*cm),
        P("Bu rapor otomatik olarak uretilmistir. "
          f"Uretim tarihi: {tarih}  |  Veri: BBC Sound Archive  |  "
          "26 sinif, 15.356 WAV, 312.725 segment",
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
