"""sema_pipeline — Sistem/pipeline genel bakis semasi (hakem revizyonu).

BBC arsivi -> on-isleme/bolme -> (A) siniflandirma temeli + (B) uretici
artirma hatti (CLAP ince-ayar -> AudioLDM2 -> sentetik klip -> augmente
egitim) -> degerlendirme (FAD + McNemar + tutarlilik) -> ANA BULGU.

Cikti: outputs/report/coherence_figs/sema_pipeline.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "report" / "coherence_figs"
OUT.mkdir(parents=True, exist_ok=True)

# Renk paleti
NAVY  = "#1D3E88"   # veri / ortak
BLUE  = "#1380A1"   # siniflandirma kolu
GREEN = "#27AE60"   # uretici kol / olumlu
ORANGE= "#E8891D"   # uretim
RED   = "#C0392B"   # ana bulgu vurgu
GRAY  = "#5D6D7E"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(9.2, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def kutu(x, y, w, h, baslik, govde, kenar, dolgu="white", baslik_renk=None,
         fs_b=9.2, fs_g=8.2):
    """Ortali baslik+govde kutusu. (x,y) sol-alt kose."""
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.4,rounding_size=1.6",
                        linewidth=1.6, edgecolor=kenar, facecolor=dolgu,
                        mutation_aspect=1)
    ax.add_patch(bb)
    cx = x + w / 2
    if baslik:
        ax.text(cx, y + h - h * 0.30, baslik, ha="center", va="center",
                fontsize=fs_b, fontweight="bold",
                color=baslik_renk or kenar)
        ax.text(cx, y + h * 0.34, govde, ha="center", va="center",
                fontsize=fs_g, color="#222222", linespacing=1.25)
    else:
        ax.text(cx, y + h / 2, govde, ha="center", va="center",
                fontsize=fs_g, color="#222222", linespacing=1.3)


def ok(x1, y1, x2, y2, renk=GRAY, lw=1.8, stil="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=stil, mutation_scale=16,
                 linewidth=lw, color=renk,
                 shrinkA=2, shrinkB=2))


# ── 1) Veri kaynagi ───────────────────────────────────────────────────────
kutu(27, 90, 46, 8.5, "BBC Ses Efektleri Arşivi",
     "15.356 WAV  ·  26 sınıf  ·  ~22× sınıf dengesizliği", NAVY,
     dolgu="#EAF0FB")

# ── 2) On-isleme + bolme ──────────────────────────────────────────────────
kutu(20, 78, 60, 9.5, "Ön-işleme ve bölme",
     "16 kHz · 5 sn örtüşmeyen segment · log-mel 128×128  →  331.785 segment\n"
     "Dosya-düzeyi katmanlı bölme (seed=42):  10.748 / 2.304 / 2.304", NAVY,
     dolgu="#EAF0FB", fs_g=8.0)

ok(50, 90, 50, 87.7, NAVY)        # arsiv -> on-isleme
ok(35, 78, 30, 70.5, BLUE)        # on-isleme -> A
ok(65, 78, 70, 70.5, GREEN)       # on-isleme -> B

# ── (A) Siniflandirma kolu (sol) ──────────────────────────────────────────
ax.text(8, 72.5, "(A) Sınıflandırma temeli", fontsize=9.4, fontweight="bold",
        color=BLUE)
kutu(4, 60, 40, 9.5, "Beş model ailesi",
     "LightGBM (Optuna) · BiLSTM+Dikkat\nAST · CNN/ResNet-18 · Geç Birleşim",
     BLUE, dolgu="#E6F2F6")
kutu(7, 48.5, 34, 8.5, "En iyi temel",
     "CNN/ResNet-18  ·  makro-F1 = 0,6716", BLUE, dolgu="#E6F2F6")
ok(24, 60, 24, 57.2, BLUE)

# ── (B) Uretici artirma kolu (sag) ────────────────────────────────────────
ax.text(58, 72.5, "(B) Üretici artırma hattı", fontsize=9.4,
        fontweight="bold", color=GREEN)
kutu(56, 60, 42, 9.5, "CLAP alan ince-ayarı",
     "HTSAT-base + RoBERTa-base · NT-Xent\nDöngü1 (BBC)  →  Döngü2 (BBC + sentetik)",
     GREEN, dolgu="#E7F6EC", fs_g=8.0)
kutu(56, 47.5, 42, 11, "AudioLDM2-full üretimi",
     "200 DDIM adımı · guidance 3,5 · TR/EN prompt\n"
     "50 benzersiz klip/sınıf\nHedef: köpek · kuş · hava  +  catch-all (kontrol)",
     ORANGE, dolgu="#FdF1E3", fs_g=7.9)
ok(77, 60, 77, 58.7, GREEN)

# ── Birlesme: augmente egitim ─────────────────────────────────────────────
kutu(28, 35, 44, 8.5, "Augmente eğitim",
     "Sentetik klipler eğitime eklenir\n(özdeş protokol · deterministik)", NAVY,
     dolgu="#EAF0FB", fs_g=8.0)
ok(24, 48.5, 35, 43.7, BLUE)      # temel -> augmente (paylasilan baseline)
ok(77, 47.5, 60, 43.7, ORANGE)   # uretim -> augmente

# ── Degerlendirme ─────────────────────────────────────────────────────────
kutu(20, 22.5, 60, 9, "Değerlendirme",
     "FAD (VGGish) · Yates-McNemar (Holm-Bonferroni)\n"
     "VGGish sınıf-içi dağılım  ↔  kazanım korelasyonu", GRAY,
     dolgu="#EEF1F4", fs_g=8.0)
ok(50, 35, 50, 31.7, NAVY)

# ── Ana bulgu ─────────────────────────────────────────────────────────────
kutu(17, 8.5, 66, 9.5, "ANA BULGU — Hedef akustik tutarlılığı",
     "Akustik olarak tutarlı sınıflarda augmentasyon anlamlı ve\n"
     "tekrarlanabilir kazanım sağlar; heterojen catch-all'da sağlamaz.",
     RED, dolgu="#FBEBEA", baslik_renk=RED, fs_g=8.1)
ok(50, 22.5, 50, 18.2, RED)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "sema_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("Kaydedildi:", OUT / "sema_pipeline.png")
