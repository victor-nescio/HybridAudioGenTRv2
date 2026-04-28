"""
Tum modeller icin PhD kalitesinde egitim raporu uretir.

Ciktilar (outputs/report/):
  fig1_training_curves.pdf/png   -- CNN / BiLSTM / AST egitim egrileri
  fig2_model_comparison.pdf/png  -- Model karsilastirma cubuk grafigi
  fig3_class_f1_heatmap.pdf/png  -- Sinif bazli F1 isi haritasi
  table1_model_summary.csv       -- Genel karsilastirma tablosu
  table2_class_report.csv        -- Sinif bazli ayrinti tablosu
  thesis_summary.txt             -- Tez icin yazilanabilir ozet

Kullanim:
  python scripts/10_training_report.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Cikti dizini ──────────────────────────────────────────────────────────────
OUT = ROOT / "outputs" / "report"
OUT.mkdir(parents=True, exist_ok=True)

# ── Sinif isimleri ────────────────────────────────────────────────────────────
SINIF_ISIMLERI = {
    0:  "acil_servis",
    1:  "at_sesleri",
    2:  "ayak_sesleri",
    3:  "cocuk_bebek",
    4:  "diger_hayvanlar",
    5:  "dini_mekan",
    6:  "doga_atmosferi",
    7:  "endustri_fabrika",
    8:  "ev_ici_sesler",
    9:  "gemi_tekne",
    10: "hava_olaylari",
    11: "insaat_sesleri",
    12: "kalabalik",
    13: "kapi_sesleri",
    14: "kara_tasitlari",
    15: "kopek_sesleri",
    16: "kus_sesleri",
    17: "ofis_iletisim",
    18: "saat_can",
    19: "sehir_atmosferi",
    20: "silah_patlama",
    21: "spor_eglence",
    22: "su_sesleri",
    23: "tarim_ciftlik",
    24: "tren_sesleri",
    25: "ucak_sesleri",
}

# ── Matplotlib tema (tez uyumlu) ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

MODEL_RENKLER = {
    "CNN (ResNet-18)":   "#2196F3",
    "BiLSTM":            "#FF5722",
    "AST (Transformer)": "#4CAF50",
}

# ── Veri yukle ────────────────────────────────────────────────────────────────
def json_yukle(yol):
    with open(yol, encoding="utf-8") as f:
        return json.load(f)

cnn_data  = json_yukle(ROOT / "results" / "cnn_20260427_0312_sonuclar.json")
lstm_data = json_yukle(ROOT / "results" / "lstm_20260427_0732_sonuclar.json")
ast_data  = json_yukle(ROOT / "results" / "ast_20260427_1430_sonuclar.json")

ms_df = pd.read_csv(ROOT / "outputs" / "model_selection_results.csv")
lgbm_metrics = json_yukle(ROOT / "outputs" / "lgbm_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1: Egitim egrileri
# ─────────────────────────────────────────────────────────────────────────────
def fig_egitim_egrileri():
    modeller = [
        ("CNN (ResNet-18)",   cnn_data),
        ("BiLSTM",            lstm_data),
        ("AST (Transformer)", ast_data),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "BBC Sound Archive — Derin Ogrenme Modelleri Egitim Süreci\n"
        "(26 Sinif, 233K Egitim Segmenti, 5 sn/segment)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col, (isim, veri) in enumerate(modeller):
        renk = MODEL_RENKLER[isim]
        tarih = veri["epoch_tarih"]
        epochlar = [t["epoch"] for t in tarih]

        # Train ve Val loss
        ax_loss = axes[0, col]
        ax_loss.plot(epochlar, [t["train_loss"] for t in tarih],
                     color=renk, linewidth=2, label="Train Loss")
        ax_loss.plot(epochlar, [t["val_loss"] for t in tarih],
                     color=renk, linewidth=2, linestyle="--",
                     alpha=0.7, label="Val Loss")
        ax_loss.set_title(isim, fontweight="bold", color=renk)
        ax_loss.set_ylabel("Loss (Cross-Entropy)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()

        # Val F1 ve en iyi noktayi isaretler
        ax_f1 = axes[1, col]
        val_f1s = [t["val_f1"] for t in tarih]
        ax_f1.plot(epochlar, val_f1s,
                   color=renk, linewidth=2.5, marker="o", markersize=4)

        en_iyi_idx = int(np.argmax(val_f1s))
        en_iyi_epoch = epochlar[en_iyi_idx]
        en_iyi_f1    = val_f1s[en_iyi_idx]
        ax_f1.axvline(en_iyi_epoch, color="gray", linestyle=":", alpha=0.7)
        ax_f1.scatter([en_iyi_epoch], [en_iyi_f1],
                      color="red", zorder=5, s=80, label=f"Best: {en_iyi_f1:.4f}")

        # Test F1 referans cizgisi
        if veri.get("test_macro_f1"):
            ax_f1.axhline(veri["test_macro_f1"], color="black",
                          linestyle="-.", linewidth=1.2,
                          label=f"Test F1: {veri['test_macro_f1']:.4f}")

        ax_f1.set_ylabel("Val Macro-F1")
        ax_f1.set_xlabel("Epoch")
        ax_f1.legend(loc="lower right")
        ax_f1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    kaydet(fig, "fig1_training_curves")
    print("  fig1_training_curves kaydedildi")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2: Model karsilastirma cubuk grafigi
# ─────────────────────────────────────────────────────────────────────────────
def fig_model_karsilastirma():
    modeller = [
        ("k-NN",                   ms_df[ms_df.ad=="k-NN (k=5)"].iloc[0]["val_macro_f1"],
                                   ms_df[ms_df.ad=="k-NN (k=5)"].iloc[0]["test_macro_f1"]),
        ("SVM (RBF)",              ms_df[ms_df.ad=="SVM (RBF)"].iloc[0]["val_macro_f1"],
                                   ms_df[ms_df.ad=="SVM (RBF)"].iloc[0]["test_macro_f1"]),
        ("Extra Trees",            ms_df[ms_df.ad=="Extra Trees"].iloc[0]["val_macro_f1"],
                                   ms_df[ms_df.ad=="Extra Trees"].iloc[0]["test_macro_f1"]),
        ("XGBoost",                ms_df[ms_df.ad=="XGBoost"].iloc[0]["val_macro_f1"],
                                   ms_df[ms_df.ad=="XGBoost"].iloc[0]["test_macro_f1"]),
        ("HistGB",                 ms_df[ms_df.ad=="Hist Gradient Boosting"].iloc[0]["val_macro_f1"],
                                   ms_df[ms_df.ad=="Hist Gradient Boosting"].iloc[0]["test_macro_f1"]),
        ("LightGBM",               ms_df[ms_df.ad=="LightGBM"].iloc[0]["val_macro_f1"],
                                   lgbm_metrics["test_macro_f1"]),
        ("BiLSTM",                 lstm_data["en_iyi_val_f1"],
                                   lstm_data["test_macro_f1"]),
        ("AST (Transformer)",      ast_data["en_iyi_val_f1"],
                                   None),
        ("CNN (ResNet-18)",        cnn_data["en_iyi_val_f1"],
                                   cnn_data["test_macro_f1"]),
    ]

    isimler  = [m[0] for m in modeller]
    val_f1s  = [m[1] for m in modeller]
    test_f1s = [m[2] if m[2] is not None else np.nan for m in modeller]

    renkler = []
    for ad in isimler:
        if ad in ("CNN (ResNet-18)", "BiLSTM", "AST (Transformer)"):
            renkler.append("#2196F3")
        elif ad == "LightGBM":
            renkler.append("#FF5722")
        else:
            renkler.append("#9E9E9E")

    x = np.arange(len(isimler))
    genislik = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_val  = ax.bar(x - genislik/2, val_f1s,  genislik, label="Val Macro-F1",
                       color=[c + "CC" for c in renkler], edgecolor="white")
    bars_test = ax.bar(x + genislik/2, test_f1s, genislik, label="Test Macro-F1",
                       color=renkler, edgecolor="white")

    # Deger etiketleri
    for bar in bars_val:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars_test:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(isimler, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0, 0.85)
    ax.set_title(
        "BBC Ses Arsivi — Model Performans Karsilastirmasi (26 Sinif)\n"
        "Gri: Klasik ML  |  Turuncu: LightGBM (Optuna HPO)  |  Mavi: Derin Ogrenme",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.4, linewidth=1)

    plt.tight_layout()
    kaydet(fig, "fig2_model_comparison")
    print("  fig2_model_comparison kaydedildi")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3: Sinif bazli F1 isi haritasi (CNN vs BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sinif_heatmap():
    def f1_dizi(rapor):
        return [rapor[str(i)]["f1-score"] for i in range(26)]

    cnn_f1s  = f1_dizi(cnn_data["sinif_raporu"])
    lstm_f1s = f1_dizi(lstm_data["sinif_raporu"])
    matris = np.array([cnn_f1s, lstm_f1s])

    sinif_etiketleri = [SINIF_ISIMLERI[i] for i in range(26)]
    model_etiketleri = ["CNN\n(ResNet-18)", "BiLSTM"]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(matris, aspect="auto", cmap="RdYlGn",
                   vmin=0.0, vmax=1.0)

    ax.set_xticks(range(26))
    ax.set_xticklabels(sinif_etiketleri, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(2))
    ax.set_yticklabels(model_etiketleri, fontsize=10)
    ax.set_title(
        "Sinif Bazli F1 Skorlari — CNN vs BiLSTM (Test Kumesi)",
        fontsize=12, fontweight="bold",
    )

    for i in range(2):
        for j in range(26):
            v = matris[i, j]
            renk = "white" if v < 0.35 or v > 0.75 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=renk)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("F1-Score", fontsize=10)

    plt.tight_layout()
    kaydet(fig, "fig3_class_f1_heatmap")
    print("  fig3_class_f1_heatmap kaydedildi")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4: Sinif bazli F1 cubuk grafigi (CNN — en iyi model)
# ─────────────────────────────────────────────────────────────────────────────
def fig_sinif_cubuk():
    rapor = cnn_data["sinif_raporu"]
    etiketler = [SINIF_ISIMLERI[i] for i in range(26)]
    f1s = [rapor[str(i)]["f1-score"] for i in range(26)]
    sira = np.argsort(f1s)

    renkler = ["#4CAF50" if f >= 0.75 else "#FF9800" if f >= 0.50 else "#F44336"
               for f in [f1s[i] for i in sira]]

    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(26)
    ax.barh(y, [f1s[i] for i in sira], color=renkler, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([etiketler[i] for i in sira], fontsize=9.5)
    ax.set_xlabel("F1-Score")
    ax.set_xlim(0, 1.0)
    ax.axvline(cnn_data["test_macro_f1"], color="navy", linestyle="--",
               linewidth=1.5, label=f"Macro-F1 = {cnn_data['test_macro_f1']:.4f}")
    ax.set_title(
        "CNN (ResNet-18) — Sinif Bazli F1 Skorlari\n"
        "Yesil ≥ 0.75 | Turuncu 0.50–0.75 | Kirmizi < 0.50",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    for i, (f1, idx) in enumerate(zip([f1s[s] for s in sira], sira)):
        ax.text(f1 + 0.01, i, f"{f1:.3f}", va="center", fontsize=8.5)

    plt.tight_layout()
    kaydet(fig, "fig4_cnn_class_f1")
    print("  fig4_cnn_class_f1 kaydedildi")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: Model ozet tablosu
# ─────────────────────────────────────────────────────────────────────────────
def tablo_model_ozet():
    satirlar = []

    # Klasik ML
    for _, row in ms_df.sort_values("val_macro_f1").iterrows():
        satirlar.append({
            "Kategori":   "Klasik ML",
            "Model":      row["ad"],
            "Val F1":     f"{row['val_macro_f1']:.4f}",
            "Test F1":    f"{row['test_macro_f1']:.4f}",
            "Egitim (s)": f"{row['egitim_sure_s']:.0f}",
            "Parametreler": "467 ozellik",
            "Notlar":     "Alt-orneklem" if row["alt_orneklem"] else "",
        })

    # LightGBM Optuna
    lgbm_row = ms_df[ms_df.ad == "LightGBM"].iloc[0]
    # Remove duplicate, add optimized
    satirlar = [s for s in satirlar if s["Model"] != "LightGBM"]
    satirlar.append({
        "Kategori":   "Klasik ML (Optuna HPO)",
        "Model":      "LightGBM (Optuna)",
        "Val F1":     f"{lgbm_row['val_macro_f1']:.4f}",
        "Test F1":    f"{lgbm_metrics['test_macro_f1']:.4f}",
        "Egitim (s)": "~1800",
        "Parametreler": "467 ozellik",
        "Notlar":     "5-fold CV",
    })

    # Derin ogrenme
    satirlar.append({
        "Kategori":   "Derin Ogrenme",
        "Model":      "BiLSTM",
        "Val F1":     f"{lstm_data['en_iyi_val_f1']:.4f}",
        "Test F1":    f"{lstm_data['test_macro_f1']:.4f}",
        "Egitim (s)": f"{lstm_data['egitim_dakika']*60:.0f}",
        "Parametreler": "~2.8M",
        "Notlar":     f"30 epoch, {len(lstm_data['epoch_tarih'])} epoch calistirildi",
    })
    satirlar.append({
        "Kategori":   "Derin Ogrenme",
        "Model":      "AST (Transformer)",
        "Val F1":     f"{ast_data['en_iyi_val_f1']:.4f}",
        "Test F1":    "N/A (durduruldu)",
        "Egitim (s)": f"{ast_data['egitim_dakika']*60:.0f}",
        "Parametreler": "85.5M",
        "Notlar":     f"{ast_data['toplam_epoch']}/{ast_data['planlanan_epoch']} epoch",
    })
    satirlar.append({
        "Kategori":   "Derin Ogrenme",
        "Model":      "CNN (ResNet-18)",
        "Val F1":     f"{cnn_data['en_iyi_val_f1']:.4f}",
        "Test F1":    f"{cnn_data['test_macro_f1']:.4f}",
        "Egitim (s)": f"{cnn_data['egitim_dakika']*60:.0f}",
        "Parametreler": "11.2M",
        "Notlar":     f"30 epoch, pretrained ResNet-18",
    })

    df = pd.DataFrame(satirlar)
    yol = OUT / "table1_model_summary.csv"
    df.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"  table1_model_summary kaydedildi: {yol}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: Sinif bazli karsilastirma
# ─────────────────────────────────────────────────────────────────────────────
def tablo_sinif_raporu():
    satirlar = []
    for i in range(26):
        cnn_r  = cnn_data["sinif_raporu"][str(i)]
        lstm_r = lstm_data["sinif_raporu"][str(i)]
        satirlar.append({
            "Sinif ID":    i,
            "Sinif Adi":   SINIF_ISIMLERI[i],
            "Test Support": int(cnn_r["support"]),
            "CNN F1":      round(cnn_r["f1-score"],  4),
            "CNN Prec":    round(cnn_r["precision"],  4),
            "CNN Recall":  round(cnn_r["recall"],     4),
            "LSTM F1":     round(lstm_r["f1-score"],  4),
            "LSTM Prec":   round(lstm_r["precision"], 4),
            "LSTM Recall": round(lstm_r["recall"],    4),
            "CNN-LSTM Δ":  round(cnn_r["f1-score"] - lstm_r["f1-score"], 4),
        })
    df = pd.DataFrame(satirlar)
    df = df.sort_values("CNN F1", ascending=False)
    yol = OUT / "table2_class_report.csv"
    df.to_csv(yol, index=False, encoding="utf-8-sig")
    print(f"  table2_class_report kaydedildi: {yol}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metin ozet
# ─────────────────────────────────────────────────────────────────────────────
def metin_ozet(tablo_df, sinif_df):
    cnn_f1s = [cnn_data["sinif_raporu"][str(i)]["f1-score"] for i in range(26)]
    en_iyi_siniflar  = sorted(range(26), key=lambda i: cnn_f1s[i], reverse=True)[:5]
    en_kotu_siniflar = sorted(range(26), key=lambda i: cnn_f1s[i])[:5]

    ozet = f"""
BBC SES ARSIVI — MODEL EGITIM OZETI
=====================================
Veri Seti   : 15,356 WAV dosyasi, 26 sinif, ~312K segment (5 sn/segment)
Ozellikler  : 467-boyutlu el-yapimi (MFCC, Chroma, Spectral, ZCR, vb.)
Spectrogram : 128-mel, 5 sn @ 16 kHz -> (1, 128, 313) tensor
GPU         : NVIDIA RTX 3060 12 GB, AMP + GradScaler, CosineAnnealingLR

SINIFLANDIRMA SONUCLARI (Macro-F1)
────────────────────────────────────────────────────────────────────
  Model                  Val Macro-F1   Test Macro-F1   Egitim
  ─────────────────────  ────────────   ─────────────   ──────
  k-NN (k=5)             0.3132         0.2960          <1 s
  SVM (RBF)              0.3938         0.3803          31 s
  Extra Trees            0.4652         0.4523          88 s
  XGBoost (GPU)          0.5417         0.5369          160 s
  HistGradBoost          0.5511         0.5480          1150 s
  LightGBM (Optuna HPO)  0.5573         0.5857          ~1800 s
  BiLSTM                 0.5138         0.5124          {lstm_data['egitim_dakika']:.0f} dk
  AST/Transformer        0.6064         N/A (durduruldu){ast_data['egitim_dakika']:.0f} dk
  CNN (ResNet-18)        0.6700         0.6716          {cnn_data['egitim_dakika']:.0f} dk
────────────────────────────────────────────────────────────────────

CNN (ResNet-18) DETAY
  Test Accuracy  : {cnn_data['sinif_raporu']['accuracy']:.4f}
  Test Macro-F1  : {cnn_data['test_macro_f1']:.4f}
  Test Macro-Prec: {cnn_data['sinif_raporu']['macro avg']['precision']:.4f}
  Test Macro-Rec : {cnn_data['sinif_raporu']['macro avg']['recall']:.4f}
  En Iyi Epoch   : {max(range(len(cnn_data['epoch_tarih'])), key=lambda i: cnn_data['epoch_tarih'][i]['val_f1'])+1} / 30
  Parametre      : 11,183,578

BiLSTM DETAY
  Test Accuracy  : {lstm_data['sinif_raporu']['accuracy']:.4f}
  Test Macro-F1  : {lstm_data['test_macro_f1']:.4f}
  En Iyi Epoch   : {max(range(len(lstm_data['epoch_tarih'])), key=lambda i: lstm_data['epoch_tarih'][i]['val_f1'])+1} / 30

AST (Transformer) DETAY
  Val Macro-F1   : {ast_data['en_iyi_val_f1']:.4f}  (epoch {ast_data['en_iyi_epoch']} / {ast_data['planlanan_epoch']})
  Not             : Egitim el ile durduruldu ({ast_data['toplam_epoch']} epoch)
  Parametre      : 85,500,000

CNN — EN IYI 5 SINIF (Test F1)
"""
    for i in en_iyi_siniflar:
        ozet += f"  {SINIF_ISIMLERI[i]:<25}: {cnn_f1s[i]:.4f}\n"

    ozet += "\nCNN — EN KOTU 5 SINIF (Test F1)\n"
    for i in en_kotu_siniflar:
        ozet += f"  {SINIF_ISIMLERI[i]:<25}: {cnn_f1s[i]:.4f}\n"

    yol = OUT / "thesis_summary.txt"
    with open(yol, "w", encoding="utf-8") as f:
        f.write(ozet)
    print(f"  thesis_summary.txt kaydedildi: {yol}")
    print(ozet.encode("ascii", "replace").decode())


# ─────────────────────────────────────────────────────────────────────────────
# Yardimci: PDF + PNG kaydet
# ─────────────────────────────────────────────────────────────────────────────
def kaydet(fig, isim):
    fig.savefig(OUT / f"{isim}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{isim}.pdf", bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Ana
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nRapor uretiliyor -> {OUT}\n")

    print("Sekiller olusturuluyor...")
    fig_egitim_egrileri()
    fig_model_karsilastirma()
    fig_sinif_heatmap()
    fig_sinif_cubuk()

    print("\nTablolar olusturuluyor...")
    t1 = tablo_model_ozet()
    t2 = tablo_sinif_raporu()

    print("\nMetin ozeti olusturuluyor...")
    metin_ozet(t1, t2)

    print(f"\nTum ciktilar: {OUT}")
