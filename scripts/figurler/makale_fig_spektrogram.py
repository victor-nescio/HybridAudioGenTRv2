# -*- coding: utf-8 -*-
"""Yeni Şekil 6: gerçek vs sentetik (H1 / C4) log-mel spektrogramları.
Örnek seçimi deterministik: ada göre sıralı ilk 6 aday içinden en yüksek RMS."""
from pathlib import Path
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "makale_figurleri"
SR, NFFT, HOP, NMEL, SURE = 16000, 1024, 256, 128, 5.0
INK = "#2C3E50"

SINIF = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
ETIKET = {"en": ["dog", "bird", "weather"], "tr": ["köpek", "kuş", "hava"]}
KOLON = {
 "en": ["Real BBC recording", "Synthetic — pretrained (H1)", "Synthetic — fine-tuned (C4)"],
 "tr": ["Gerçek BBC kaydı", "Sentetik — ön-eğitimli (H1)", "Sentetik — ince ayarlı (C4)"],
}


def adaylar(kok, n=10):
    wavs = sorted(p for p in Path(kok).rglob("*.wav") if "convert" not in p.parts)
    return wavs[:n]


def skor(y):
    """Olay zenginliği: onset gücü ortalaması × RMS (sessiz/düz kesitleri eler)."""
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-4:
        return -1
    on = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    return float(np.mean(on)) * np.sqrt(rms)


def en_iyi(kok, offsetler=(0.0,)):
    best, best_s = None, -1
    for w in adaylar(kok):
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


GERCEK_SEC = {
    "kopek_sesleri": "Airedale barking angrily",
    "kus_sesleri": "Blackbird singing in quiet town garden",
    "hava_olaylari": "Heavy rain",
}


def gercek_ornek(sinif):
    """Prompt içeriğiyle eşleşen temsilî gerçek kaydın en hareketli 5 sn'lik kesiti."""
    kok = ROOT / "data" / "raw" / sinif
    dosya = next(p for p in sorted(kok.rglob("*.wav"))
                 if GERCEK_SEC[sinif].lower() in p.name.lower())
    best, best_s = None, -1
    for off in (0.0, 3.0, 6.0, 10.0):
        try:
            y, _ = librosa.load(dosya, sr=SR, mono=True, offset=off, duration=SURE)
        except Exception:
            continue
        if len(y) < SR * SURE * 0.9:
            continue
        s = skor(y)
        if s > best_s:
            best, best_s = (dosya, y), s
    return best


ornekler = {}
for s in SINIF:
    gercek = gercek_ornek(s)
    h1 = en_iyi(ROOT / "data" / "synthetic_turcoh" / s)
    c4 = en_iyi(ROOT / "data" / "synthetic_turcoh4" / s)
    ornekler[s] = [gercek, h1, c4]
    print(s, "->", [str(o[0].name) if o else None for o in ornekler[s]])

for dil in ("en", "tr"):
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.6), sharex=True, sharey=True)
    for i, s in enumerate(SINIF):
        for j in range(3):
            ax = axes[i, j]
            o = ornekler[s][j]
            if o is None:
                ax.axis("off"); continue
            db = mel_db(o[1])
            ax.imshow(db, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, SURE, 0, NMEL], vmin=-80, vmax=0)
            ax.grid(False)
            if i == 0:
                ax.set_title(KOLON[dil][j], fontsize=10.5, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(ETIKET[dil][i], fontsize=11, fontweight="bold", color=INK)
            if i == 2:
                ax.set_xlabel("s" if dil == "en" else "sn", fontsize=9)
            ax.tick_params(labelsize=8)
    fig.text(0.015, 0.5, "Mel", va="center", rotation="vertical", fontsize=9, color=INK)
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.savefig(OUT / f"fig_spec_{dil}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
print("kaydedildi: fig_spec_en/tr.png")
