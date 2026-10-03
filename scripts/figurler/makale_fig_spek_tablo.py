# -*- coding: utf-8 -*-
"""Spektrogram (Şekil 6) + Tablo 3/4/5 — büyük font, tam genişlik."""
from pathlib import Path
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "makale_figurleri"
OUT.mkdir(parents=True, exist_ok=True)
SR, NFFT, HOP, NMEL, SURE = 16000, 1024, 256, 128, 5.0
INK = "#2C3E50"; GREEN = "#1E8E5A"; GRAY = "#5D6D7E"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.dpi": 220})

SINIF = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
ETIKET = ["dog", "bird", "weather"]
KOLON = ["Real BBC recording", "Synthetic — pretrained (H1)", "Synthetic — fine-tuned (C4)"]
GERCEK = {"kopek_sesleri": "Airedale barking angrily",
          "kus_sesleri": "Blackbird singing in quiet town garden",
          "hava_olaylari": "Heavy rain"}


def skor(y):
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-4:
        return -1
    return float(np.mean(librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP))) * np.sqrt(rms)


def en_iyi(kok, offsetler=(0.0,), ad_filtre=None):
    wavs = sorted(p for p in Path(kok).rglob("*.wav") if "convert" not in p.parts)
    if ad_filtre:
        wavs = [w for w in wavs if ad_filtre.lower() in w.name.lower()] or wavs
    best, best_s = None, -1
    for w in wavs[:10]:
        for off in offsetler:
            try:
                y, _ = librosa.load(w, sr=SR, mono=True, offset=off, duration=SURE)
            except Exception:
                continue
            if len(y) < SR * SURE * 0.9:
                continue
            s = skor(y)
            if s > best_s:
                best, best_s = (w, y), s
    return best


def mel_db(y):
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=NFFT, hop_length=HOP, n_mels=NMEL)
    return librosa.power_to_db(m, ref=np.max)


def spektrogram():
    ornek = {}
    for s in SINIF:
        g = en_iyi(ROOT / "data" / "raw" / s, (0.0, 3.0, 6.0, 10.0), GERCEK[s])
        h1 = en_iyi(ROOT / "data" / "synthetic_turcoh" / s)
        c4 = en_iyi(ROOT / "data" / "synthetic_turcoh4" / s)
        ornek[s] = [g, h1, c4]
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.2), sharex=True, sharey=True)
    for i, s in enumerate(SINIF):
        for j in range(3):
            ax = axes[i, j]; o = ornek[s][j]
            if o is None:
                ax.axis("off"); continue
            ax.imshow(mel_db(o[1]), aspect="auto", origin="lower", cmap="magma",
                      extent=[0, SURE, 0, NMEL], vmin=-80, vmax=0)
            ax.grid(False)
            if i == 0:
                ax.set_title(KOLON[j], fontsize=14, color=INK, pad=9)
            if j == 0:
                ax.set_ylabel(ETIKET[i], fontsize=15, fontweight="bold", color=INK)
            if i == 2:
                ax.set_xlabel("s", fontsize=13)
            ax.tick_params(labelsize=11.5)
    fig.text(0.02, 0.5, "Mel", va="center", rotation="vertical", fontsize=13, color=INK)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.savefig(OUT / "fig_spec_big.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig_spec_big")


def tablo(ad, bas, satirlar, dip, kolx, en_iyi_sutun=False, alt_kalin=False, fig_w=9.6):
    fig, ax = plt.subplots(figsize=(fig_w, 0.62 * len(satirlar) + 1.5))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n = len(satirlar)
    baslik_y = 0.93
    satir_h = 0.80 / (n + 0.5)
    cizgi_y = baslik_y - satir_h * 0.6
    for x, b in zip(kolx, bas):
        ax.text(x, baslik_y, b, fontsize=14, fontweight="bold", color=INK,
                ha="left" if x == kolx[0] else "center")
    ax.plot([0.01, 0.99], [cizgi_y] * 2, color=INK, lw=1.3)
    for i, sat in enumerate(satirlar):
        yy = cizgi_y - (i + 0.7) * satir_h
        son = alt_kalin and i == n - 1
        vurgu = "***" in sat[-1] if isinstance(sat[-1], str) else False
        if son:
            ax.plot([0.01, 0.99], [yy + satir_h * 0.5] * 2, color="#BDC3C7", lw=0.8)
        elif vurgu:
            ax.add_patch(plt.Rectangle((0.01, yy - satir_h * 0.42), 0.98, satir_h * 0.86,
                         facecolor="#E9F7EF", edgecolor="none"))
        elif i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, yy - satir_h * 0.42), 0.98, satir_h * 0.86,
                         facecolor="#F4F6F7", edgecolor="none"))
        if en_iyi_sutun:
            sayi = [float(str(v).replace(",", ".")) for v in sat[1:] if str(v).replace(",", ".").replace(".", "").isdigit()]
            mx = max(sayi) if sayi else None
        for j, (x, v) in enumerate(zip(kolx, sat)):
            iyi = en_iyi_sutun and j > 0 and str(v).replace(",", ".").replace(".", "").isdigit() and float(str(v).replace(",", ".")) == mx
            renk = GREEN if (iyi or (j == len(sat) - 1 and v == "***")) else INK
            ax.text(x, yy, v, fontsize=13, ha="left" if j == 0 else "center", va="center",
                    color=renk, fontweight="bold" if (iyi or son) else "normal")
    alt = cizgi_y - (n + 0.35) * satir_h
    ax.plot([0.01, 0.99], [alt] * 2, color="#BDC3C7", lw=0.8)
    ax.text(0.01, alt - satir_h * 0.7, dip, fontsize=9.5, color=GRAY, ha="left", va="top", wrap=True)
    fig.savefig(OUT / f"{ad}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {ad}")


def tablolar():
    # Tablo 3 — McNemar
    tablo("tablo3_big",
          ["Target / Run", "χ²", "p", ""],
          [("Combined targets (H1, pretrained · EN)", "25.5", "4.4×10⁻⁷", "***"),
           ("Combined targets (C2, Cycle 1 · EN)", "20.6", "5.8×10⁻⁶", "***"),
           ("Combined targets (C3, Cycle 1 · TR)", "23.1", "1.5×10⁻⁶", "***"),
           ("Combined targets (C4, Cycle 2 · EN)", "66.4", "3.3×10⁻¹⁶", "***"),
           ("Control (other animals, C4)", "1.27", "0.261", "ns")],
          "Yates-corrected McNemar test (df=1). Combined targets: dog+bird+weather, n=2,343 test "
          "segments. Holm–Bonferroni correction applied across classes. *** p<0.001; ns: not significant.",
          [0.03, 0.60, 0.75, 0.92])
    # Tablo 4 — FAD
    tablo("tablo4_big",
          ["Class", "H1", "C2", "C3 (TR)", "C4"],
          [("dog", "12.5", "11.3", "29.1", "11.2"),
           ("bird", "19.4", "17.3", "27.7", "19.2"),
           ("weather", "8.5", "8.6", "20.2", "8.1"),
           ("Mean", "13.5", "12.4", "25.7", "12.8")],
          "Deterministic FAD in the VGGish embedding space; reference distribution: all real recordings "
          "of each class (294 / 264 / 300 files for dog / bird / weather); 50 synthetic clips per class "
          "and configuration; fixed sampling seed.",
          [0.04, 0.40, 0.56, 0.72, 0.88], alt_kalin=True, fig_w=8.6)
    # Tablo 5 — SpecAugment (başlıklar kısaltıldı, kolonlar genişçe dağıtıldı)
    tablo("tablo5_big",
          ["Class", "Baseline", "+SpecAug", "+Gen. (C4)", "+Both"],
          [("dog", "0.665", "0.709", "0.705", "0.670"),
           ("bird", "0.591", "0.638", "0.669", "0.614"),
           ("weather", "0.553", "0.573", "0.605", "0.596"),
           ("other animals (control)", "0.582", "0.561", "0.578", "0.612")],
          "Class-wise F1, identical training protocol and split for all models. McNemar on the combined "
          "target classes: generative vs. SpecAugment χ²=0.06, p=0.81 (ns); combined vs. SpecAugment "
          "χ²=5.70, p=0.017 (degradation; Dog p<10⁻⁶). Full test set: SpecAugment vs. generative p<0.001.",
          [0.02, 0.42, 0.585, 0.75, 0.92], en_iyi_sutun=True, fig_w=11.0)


if __name__ == "__main__":
    import sys
    if "--tablo" in sys.argv:
        print("Sadece tablolar...")
        tablolar()
    else:
        print("Spektrogram + tablolar (buyuk font)...")
        spektrogram()
        tablolar()
    print("Tamam.")
