"""
Revizyon (v2) sekilleri — IEEE Access sutun genisligi, >= 8 pt yazi (R4.6). Ingilizce etiketler.

  fig_v2_is_akisi  : Sekil 1, is akisi (revize protokol)
  fig_v2_dagilim   : Sekil 2, sinif basina kayit sayisi (egitim / dogrulama + test)
  fig_v2_modeller  : Sekil 3, siniflandirici aileleri, dogrulama ve test macro-F1 (R1.1: secim dogrulamada)
  fig_v2_sinif_f1  : Sekil 4, CNN baseline sinif F1 (seed 42, test); Grup A / B / kontrol isaretli
  fig_v2_kopya     : Sekil 5, sabit uretim seed'inin yol actigi kopya (duzeltme oncesi dosyalardan MD5 sayimi)
  fig_v2_spectrogram, fig_v2_classwise_1c, fig_v2_dose, fig_v2_forest_1c: Sekil 6-9
  fig_v2_forest, fig_v2_classwise: iki sutun genisliginde surumler (makalede kullanilmiyor)

Boyut kurali: makaledeki sekiller tek sutuna (3.35 in) KUCULTULMEDEN, dogal genislikleriyle yerlestirilir
(olustur.py); bu yuzden her sekil yayin boyutunda cizilir, tum yazilar >= 8 pt'dir ve kaydet() dogal genisligin
sutunu asmadigini denetler (asarsa durur).
Renk: referans paletin dogrulanmis ilk iki kategorik slotu (mavi #2a78d6, turuncu #eb6834) + gri tonlu
baski icin ikincil kodlama (isaretci + cizgi turu); isi haritasi mavi <-> kirmizi, notr gri orta nokta.
Girdi: results/v2/ikincil_analiz_tam.json (+ sinif bazinda noktalar 20_ikincil_aile fonksiyonlariyla),
results/v1 (baseline aileleri), outputs/model_selection_results.csv ve outputs/lgbm_metrics.json (klasik ML),
data/_dup_backup_20260531 (duzeltme oncesi uretim; sayim results/v2/kopya_oncesi_sayim.json'a yazilir).
Kullanim: python scripts/23_sekiller_v2.py [--sadece ad1 ad2 ...]  ->  results/v2/sekiller/*.pdf, *.png
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parents[1]
CIKTI = ROOT / "results" / "v2" / "sekiller"
SUTUN = 3.35          # IEEE Access iki sutunlu duzende sutun genisligi (in)
TAM = 7.0             # iki sutun genisligi (in)
MAVI, TURUNCU, KIRMIZI, NOTR = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
MUREKKEP, IKINCIL, IZGARA = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "axes.edgecolor": IKINCIL, "axes.labelcolor": MUREKKEP, "xtick.color": IKINCIL,
                     "ytick.color": IKINCIL, "text.color": MUREKKEP, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "savefig.dpi": 300})
SINIF_EN = {"kopek_sesleri": "Dog", "kus_sesleri": "Bird", "hava_olaylari": "Weather",
            "insaat_sesleri": "Construction", "dini_mekan": "Religious place", "tarim_ciftlik": "Farm/livestock",
            "acil_servis": "Emergency", "diger_hayvanlar": "Other animals"}
KONFIG_EN = {"baseline": "baseline", "H1": "H1", "C2p": "C2′", "C3p": "C3′", "C4p": "C4′",
             "C4p_kaydir": "C4′-shift", "specaug": "SpecAug", "specaugC4p": "SpecAug+C4′"}
GRUP_EN = {"A_belirgin": "Group A", "B_dusukF1": "Group B", "kontrol": "control"}


EN26 = {  # ilk gonderimin Sekil 4'undeki adlar (24_ek_tablolar.py ile ayni)
    "ayak_sesleri": "footstep sounds", "at_sesleri": "horse sounds", "saat_can_sesleri": "clock/bell",
    "cocuk_bebek_sesleri": "children/baby", "kapi_sesleri": "door sounds", "ucak_sesleri": "aircraft sounds",
    "kara_tasitlari": "land vehicles", "gemi_tekne_sesleri": "ship/boat", "tren_sesleri": "train sounds",
    "ev_ici_sesler": "indoor sounds", "endustri_fabrika": "industry/factory", "kopek_sesleri": "dog sounds",
    "spor_eglence": "sports/entertainment", "ofis_iletisim": "office/communication", "kalabalik_sesleri": "crowd sounds",
    "silah_patlama": "weapon/explosion", "sehir_atmosferi": "city atmosphere", "kus_sesleri": "bird sounds",
    "acil_servis": "emergency service", "doga_atmosferi": "nature atmosphere", "diger_hayvanlar": "other animals",
    "su_sesleri": "water sounds", "hava_olaylari": "weather events", "tarim_ciftlik": "farm/livestock",
    "dini_mekan": "religious place", "insaat_sesleri": "construction",
}
GRUP_A = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
GRUP_B = ["insaat_sesleri", "dini_mekan", "tarim_ciftlik"]
ACIK_GRI, MAVI_ACIK = "#c9c8c4", "#a9c9ef"


def kaydet(fig, ad, azami=SUTUN):
    """Kirpilmis (tight) olarak kaydeder ve dogal genisligi denetler: makalede sekil kucultulmeden
    yerlestirildigi icin cizimdeki 8 pt yazi baskida da 8 pt olur."""
    CIKTI.mkdir(parents=True, exist_ok=True)
    for uz in ("pdf", "png"):
        fig.savefig(CIKTI / f"{ad}.{uz}", bbox_inches="tight", pad_inches=0.02, facecolor="white")
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    plt.close(fig)
    g, y = bb.width + 0.04, bb.height + 0.04
    print(f"kaydedildi: {CIKTI / ad}.pdf  ({g:.2f} x {y:.2f} in)")
    if g > azami + 0.005:
        raise SystemExit(f"HATA: {ad} genisligi {g:.2f} in > {azami} in (yazi baskida kuculur)")


def p_metni(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.1 else f"p = {p:.2f}"


def forest(d):
    satir = [("P1: " + "C4′ − baseline (Group A)", d["P1_hatirlatma"], None, True)]
    for x in d["aile"]:
        etiket = f"{KONFIG_EN[x['test']]} − {KONFIG_EN[x['referans']]}"
        if x["grup"] != "A_belirgin":
            etiket += f" ({GRUP_EN[x['grup']]})"
        satir.append((f"{x['no']}. {etiket}", x, x["p_holm"], False))
    n = len(satir)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    y = np.arange(n)[::-1].astype(float)
    y[1:] -= 0.5                                          # birincil ile aile arasinda bosluk
    for yi, (etiket, v, ph, birincil) in zip(y, satir):
        renk = MAVI if birincil else IKINCIL
        ax.plot(v["ga95"], [yi, yi], color=renk, lw=1.6 if birincil else 1.2, solid_capstyle="round", zorder=2)
        ax.plot(v["delta_f1"], yi, marker="o", ms=6 if birincil else 5, color=renk, mec="white", mew=1.0, zorder=3)
        sag = p_metni(v["p_bootstrap"]) + ("" if ph is None else f"   Holm {ph:.2f}")
        ax.text(1.02, yi, sag, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8,
                color=MUREKKEP if birincil else IKINCIL)
    ax.axvline(0, color=IKINCIL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline((y[0] + y[1]) / 2, color=IZGARA, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in satir])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("ΔF1 (difference in group macro-F1, 95% recording-level bootstrap CI)")
    ax.set_ylim(y[-1] - 0.7, y[0] + 0.7)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=IZGARA, lw=0.5)
    ax.set_axisbelow(True)
    ax.text(0.0, 1.02, "Primary (P1) and pre-registered secondary family (Holm, m = 11); final-epoch model, "
            "validation + test", transform=ax.transAxes, fontsize=8, color=IKINCIL)
    kaydet(fig, "fig_v2_forest", azami=8.0)


def forest_1c(d):
    """Tek sutun (3.35 in) surumu: makalede ilk gonderimin Sekil 9'unun yerine."""
    kisa = {"baseline": "base", "C4p_kaydir": "Shift", "specaug": "SA", "specaugC4p": "SA+C4′"}
    ad = lambda k: kisa.get(k, KONFIG_EN[k])
    satir = [("P1  C4′ − base", d["P1_hatirlatma"], None, True)]
    for x in d["aile"]:
        e = f"{x['no']}  {ad(x['test'])} − {ad(x['referans'])}"
        if x["grup"] != "A_belirgin":
            e += {"B_dusukF1": " (B)", "kontrol": " (ctrl)"}[x["grup"]]
        satir.append((e, x, x["p_holm"], False))
    fig = plt.figure(figsize=(SUTUN - 0.05, 3.6))
    ax = fig.add_axes([0.30, 0.13, 0.47, 0.80])         # sol: etiketler, sag: p ve Holm sutunlari (sekil icinde)
    sutun = blended_transform_factory(fig.transFigure, ax.transData)
    y = np.arange(len(satir))[::-1].astype(float)
    y[1:] -= 0.5
    for yi, (etiket, v, ph, birincil) in zip(y, satir):
        renk = MAVI if birincil else IKINCIL
        ax.plot(v["ga95"], [yi, yi], color=renk, lw=1.6 if birincil else 1.1, solid_capstyle="round", zorder=2)
        ax.plot(v["delta_f1"], yi, marker="o", ms=5.5 if birincil else 4.5, color=renk, mec="white", mew=0.8, zorder=3)
        p = v["p_bootstrap"]
        ax.text(0.79, yi, f"{p:.3f}" if p < 0.1 else f"{p:.2f}", transform=sutun, va="center",
                ha="left", fontsize=8, color=MUREKKEP if birincil else IKINCIL)
        ax.text(0.905, yi, "—" if ph is None else f"{ph:.2f}", transform=sutun, va="center",
                ha="left", fontsize=8, color=MUREKKEP if birincil else IKINCIL)
    ax.text(0.79, y[0] + 1.0, "p", transform=sutun, ha="left", fontsize=8, color=IKINCIL)
    ax.text(0.905, y[0] + 1.0, "Holm", transform=sutun, ha="left", fontsize=8, color=IKINCIL)
    ax.axvline(0, color=IKINCIL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline((y[0] + y[1]) / 2, color=IZGARA, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in satir])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("ΔF1 (group macro-F1), 95% CI")
    ax.set_xticks([-0.02, 0.0, 0.02])
    ax.set_xticklabels(["−0.02", "0", "+0.02"])
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.4)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=IZGARA, lw=0.5)
    ax.set_axisbelow(True)
    kaydet(fig, "fig_v2_forest_1c")


def classwise_1c():
    """Tek sutun surumu (satir = sinif, sutun = konfigurasyon); veri classwise() ciktisindan."""
    v = json.loads((CIKTI / "fig_v2_classwise_veri.json").read_text(encoding="utf-8"))
    M = np.array(v["delta_f1"]).T
    lim = float(np.ceil(np.abs(M).max() * 100) / 100)
    cmap = LinearSegmentedColormap.from_list("mavi_kirmizi", [KIRMIZI, NOTR, MAVI])
    fig = plt.figure(figsize=(SUTUN - 0.05, 3.15))
    ax = fig.add_axes([0.335, 0.19, 0.62, 0.725])
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=8, color=MUREKKEP)
    for yy in (2.5, 5.5, 6.5):
        ax.axhline(yy, color="white", lw=2.0)
    ax.set_yticks(range(len(v["siniflar"])))
    ax.set_yticklabels([SINIF_EN[c] for c in v["siniflar"]])
    ax.set_xticks(range(len(v["konfig"])))
    ax.set_xticklabels([KONFIG_EN[k] for k in v["konfig"]])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for k in ax.spines.values():
        k.set_visible(False)
    grup = blended_transform_factory(fig.transFigure, ax.transData)
    for y0, y1, ad in ((0, 2, "Group A"), (3, 5, "Group B")):
        ax.text(0.025, (y0 + y1) / 2, ad, transform=grup, ha="center", va="center", fontsize=8, color=IKINCIL,
                rotation=90)
    cax = fig.add_axes([0.335, 0.115, 0.62, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("ΔF1 vs. paired baseline")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    tik = np.linspace(-lim, lim, 5)
    cb.set_ticks(tik)
    cb.set_ticklabels([f"{t:+.2f}".replace("-", "−").replace("+0.00", "0") for t in tik])
    kaydet(fig, "fig_v2_classwise_1c")


def classwise():
    s = importlib.util.spec_from_file_location("ia", ROOT / "scripts" / "20_ikincil_aile.py")
    IA = importlib.util.module_from_spec(s)
    s.loader.exec_module(IA)
    konfig = ["H1", "C2p", "C3p", "C4p", "C4p_kaydir"]
    ctx = IA.hazirla(["baseline"] + konfig, "son_valtest")
    siniflar = list(SINIF_EN)
    M = np.zeros((len(konfig), len(siniflar)))
    for i, k in enumerate(konfig):
        noktalar = {}
        for g in ("A_belirgin", "B_dusukF1", "B_faz1_acil", "kontrol"):
            noktalar.update(IA.kars(ctx, k, "baseline", g)["sinif"])
        M[i] = [noktalar[c] for c in siniflar]
    json.dump({"konfig": konfig, "siniflar": siniflar, "delta_f1": M.round(4).tolist()},
              open(CIKTI / "fig_v2_classwise_veri.json", "w", encoding="utf-8"), indent=1)
    lim = float(np.ceil(np.abs(M).max() * 100) / 100)
    cmap = LinearSegmentedColormap.from_list("mavi_kirmizi", [KIRMIZI, NOTR, MAVI])
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=8, color=MUREKKEP)
    for x in (2.5, 5.5, 6.5):
        ax.axvline(x, color="white", lw=2.0)
    ax.set_xticks(range(len(siniflar)))
    ax.set_xticklabels([SINIF_EN[c].replace(" ", "\n").replace("/", "/\n") for c in siniflar])
    ax.set_yticks(range(len(konfig)))
    ax.set_yticklabels([KONFIG_EN[k] for k in konfig])
    ax.tick_params(length=0)
    for k in ax.spines.values():
        k.set_visible(False)
    for x0, x1, ad in ((0, 2, "Group A"), (3, 5, "Group B"), (7, 7, "Control")):
        ax.text((x0 + x1) / 2, -0.75, ad, ha="center", va="bottom", fontsize=8, color=IKINCIL)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("ΔF1 vs. baseline")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    kaydet(fig, "fig_v2_classwise", azami=8.0)


def doz(d):
    dz = d["doz_cevap"]["dozlar"]
    x = np.array([25, 50, 100, 200])
    fig = plt.figure(figsize=(SUTUN - 0.05, 2.6))
    ax = fig.add_axes([0.17, 0.30, 0.66, 0.66])          # sagda dogrudan etiketler icin yer
    for g, renk, isaret, cizgi, kay, ad in (("A_belirgin", MAVI, "o", "-", 0.96, "Group A (dog, bird, weather)"),
                                            ("kontrol", TURUNCU, "s", (0, (4, 2)), 1.04, "Control (other animals)")):
        v = [dz[str(k)][g] for k in x]
        nk = np.array([a["delta_f1"] for a in v])
        lo = nk - np.array([a["ga95"][0] for a in v])
        hi = np.array([a["ga95"][1] for a in v]) - nk
        xs = x * kay
        ax.errorbar(xs, nk, yerr=[lo, hi], color=renk, lw=0, elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=2)
        ax.plot(xs, nk, color=renk, lw=1.6, ls=cizgi, marker=isaret, ms=5.5, mec="white", mew=1.0, label=ad, zorder=3)
        ax.annotate(ad.split(" (")[0], (xs[-1], nk[-1]), xytext=(9, 0 if g == "A_belirgin" else -7),
                    textcoords="offset points", va="center",
                    fontsize=8, color=MUREKKEP)                          # dogrudan etiket (renk degil murekkep)
    ax.axhline(0, color=IKINCIL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.minorticks_off()
    ax.set_xlabel("Synthetic clips per class")
    ax.set_ylabel("ΔF1 vs. baseline")
    ax.grid(axis="y", color=IZGARA, lw=0.5)
    ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.set_xlim(20, 290)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{t:+.2f}".replace("-", "−").replace("+0.00", "0") for t in ax.get_yticks()])
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=False,
               ncol=1, handlelength=2.4)
    kaydet(fig, "fig_v2_dose")


def spektrogram():
    """Ilk gonderimin Sekil 6'sinin karsiligi: gercek / H1 / C4' log-Mel (C4' artik gercekten ince-ayarli).
    Ornekler deterministik: gercekte sinifin egitim listesindeki ilk kaydin orta 5 sn'si (mel onbellegiyle ayni on
    isleme), sentetikte egitime giren ilk klip (uretim seed sirasi)."""
    import sys
    import librosa
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    from src.data.dataset import _mel_hesapla

    def mod(ad, yol):
        s = importlib.util.spec_from_file_location(ad, yol)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        return m
    IL = mod("il", ROOT / "scripts" / "07_iterative_loop.py")
    S15 = mod("s15", ROOT / "scripts" / "15_siniflandirma_v2.py")
    tr, _, _ = IL.veri_bol(IL.PARQUET_LISTE, ROOT)
    siniflar = ["kopek_sesleri", "kus_sesleri", "hava_olaylari"]
    h1 = S15.sentetik_sec(["data/synthetic_turcoh:" + ",".join(siniflar)], 50)
    c4 = S15.sentetik_sec(["data/synthetic_v2/C4p:" + ",".join(siniflar)], 50)

    def hazirla(y, sr):
        y = y.mean(1) if y.ndim > 1 else y
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        y = y - y.mean()
        return (y * (0.99 / np.abs(y).max())).astype(np.float32)

    def mel(y):
        m = _mel_hesapla(y, 16000)[:, :313]
        return m - m.max()

    # Secim kurali (seffaf, deterministik): gercek = arsiv aciklamasi sinifin ilk promptundaki olayi iceren ilk
    # egitim kaydi (dosya adina gore sirali), bu kaydin en yuksek ortalama enerjili 5 sn segmenti (egitimde
    # kullanilan mel onbellegindeki dilim); sentetik = ayni ilk prompttan uretilmis ilk klip (uretim seed sirasi).
    anahtar = {"kopek_sesleri": "bark", "kus_sesleri": "dawn chorus", "hava_olaylari": "thunder"}
    satirlar = []
    for c in siniflar:
        aday = tr[(tr.sinif == c) & tr.dosya.str.lower().str.contains(anahtar[c])].sort_values("dosya")
        r = aday.iloc[0]
        tam = np.load(ROOT / "data" / "mel_cache" / Path(r.dosya).with_suffix(".npy"), mmap_mode="r")
        s = int(np.argmax([tam[:, k * 313:(k + 1) * 313].astype(np.float32).mean() for k in range(int(r.n_segs))]))
        gercek = tam[:, s * 313:(s + 1) * 313].astype(np.float32)
        gercek = gercek - gercek.max()
        sent = [mel(hazirla(*sf.read(str(ROOT / df[df.sinif == c].iloc[0].dosya), dtype="float32", always_2d=False)))
                for df in (h1, c4)]
        satirlar.append([gercek] + sent)
        print(c, "gercek:", Path(r.dosya).name[:70], "seg", s, "| sentetik:",
              Path(h1[h1.sinif == c].iloc[0].dosya).name, Path(c4[c4.sinif == c].iloc[0].dosya).name, flush=True)
    cmap = LinearSegmentedColormap.from_list("tek_ton", ["#ffffff", MAVI, "#0b2a4a"])
    fig, axs = plt.subplots(3, 3, figsize=(3.35, 3.4), sharex=True, sharey=True)
    for i, c in enumerate(siniflar):
        for j, m in enumerate(satirlar[i]):
            ax = axs[i, j]
            im = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, vmin=-80, vmax=0,
                           extent=[0, m.shape[1] * 256 / 16000, 0, m.shape[0]])
            ax.tick_params(length=2)
            if i == 0:
                ax.set_title(["Real", "H1 (pretrained)", "C4′ (fine-tuned)"][j], fontsize=8, color=MUREKKEP)
            if j == 0:
                ax.set_ylabel(f"{SINIF_EN[c]}\nMel band")
            if i == 2:
                ax.set_xlabel("Time (s)")
            ax.set_yticks([0, 64, 128])
            ax.set_xticks([0, 2.5, 5])
            ax.set_xticklabels(["0", "2.5", "5"])
    cb = fig.colorbar(im, ax=axs, orientation="horizontal", fraction=0.04, pad=0.2, aspect=35)
    cb.set_label("Level (dB re. max)")
    cb.outline.set_visible(False)
    kaydet(fig, "fig_v2_spectrogram")


def is_akisi():
    """Ilk gonderimin Sekil 1'inin (is akisi) guncel karsiligi: ayni kutu yapisi, revize protokol ve bulgu."""
    from matplotlib.patches import FancyBboxPatch
    MAVI_ACIK2, TURUNCU_ACIK, GRI_ACIK = "#e3eefb", "#fde8dd", "#f0efec"
    H = 5.55                                              # sekil yuksekligi (in); tum yazilar 8 pt
    fig = plt.figure(figsize=(SUTUN - 0.05, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    satir = 8.2 / 72 / H                                  # olculen satir araligi (8 pt Times, 1.18) eksen biriminde
    ust_bosluk, ok_boyu = 0.012, 0.018

    def yukseklik(metin):
        return ust_bosluk + satir * (1.2 + metin.count("\n") + 1) + 0.016

    def kutu(x, y_ust, w, baslik, metin, zemin, kenar=IKINCIL, kalin=0.8, h=None):
        h = h or yukseklik(metin)
        ax.add_patch(FancyBboxPatch((x, y_ust - h), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=zemin,
                                    ec=kenar, lw=kalin))
        ax.text(x + w / 2, y_ust - ust_bosluk, baslik, ha="center", va="top", fontsize=8, fontweight="bold",
                color=MUREKKEP)
        ax.text(x + w / 2, y_ust - ust_bosluk - satir * 1.2, metin, ha="center", va="top", fontsize=8,
                color=MUREKKEP, linespacing=1.18)
        return y_ust - h

    def ok(x, y1, y2):
        ax.annotate("", (x, y2), (x, y1), arrowprops=dict(arrowstyle="-|>", color=IKINCIL, lw=0.8, mutation_scale=7))

    y = kutu(0.04, 0.995, 0.92, "BBC Sound Effects Archive",
             "16,008 unique recordings; 15,356 in 26 classes\n(ordered keyword taxonomy)", GRI_ACIK)
    ok(0.5, y, y - ok_boyu); y -= ok_boyu
    y = kutu(0.04, y, 0.92, "Preprocessing and splitting",
             "16 kHz · 5-s segments · log-Mel 128 × 313\nfile-level split (seed 42): 10,748 / 2,304 / 2,304", GRI_ACIK)
    ok(0.255, y, y - ok_boyu); ok(0.745, y, y - ok_boyu); y -= ok_boyu
    sol = "LightGBM · BiLSTM · AST\nResNet-18 · late fusion\nselected on validation:\nResNet-18"
    sag = "training partition only\nfp32 · two cycles\nCycle 1: BBC\nCycle 2: BBC + synthetic"
    h = max(yukseklik(sol), yukseklik(sag))
    kutu(0.02, y, 0.47, "A) Classifier selection", sol, MAVI_ACIK2, h=h)
    y = kutu(0.51, y, 0.47, "B) CLAP fine-tuning", sag, TURUNCU_ACIK, h=h)
    ok(0.745, y, y - ok_boyu)
    y_sol = y
    y = kutu(0.51, y - ok_boyu, 0.47, "AudioLDM2 generation",
             "H1 · C2′ · C3′ (TR) · C4′\n50 clips per class\nGroups A, B and control", TURUNCU_ACIK)
    ok(0.255, y_sol, y - ok_boyu); ok(0.745, y, y - ok_boyu); y -= ok_boyu
    y = kutu(0.04, y, 0.92, "Augmented training",
             "ResNet-18 · paired seeds (5 or 3)\ncontent control (label shift) · SpecAugment\n"
             "SpecAugment + C4′ · dose 25–200 clips\nfrozen BEATs + MLP as second classifier", GRI_ACIK)
    ok(0.5, y, y - ok_boyu); y -= ok_boyu
    y = kutu(0.04, y, 0.92, "Evaluation",
             "final-epoch models · validation + test (4,608 rec.)\nrecording-level cluster bootstrap\n"
             "pre-registered P1 · Holm-corrected secondary family\nFAD · CLAP score · classifier validity · memorization",
             GRI_ACIK)
    ok(0.5, y, y - ok_boyu); y -= ok_boyu
    y = kutu(0.04, y, 0.92, "Main finding",
             "+0.018 macro-F1 for acoustically distinctive classes\n(95% CI [0.004, 0.033], p = 0.010; pre-registered)\n"
             "no gain for lowest-F1 classes, control or BEATs\nno added benefit of CLAP fine-tuning", "#ffffff",
             kenar=MAVI, kalin=1.4)
    if y < 0:
        raise SystemExit(f"HATA: is akisi sekle sigmiyor (alt kenar {y:.3f})")
    alt = y - 0.006                                       # bos alt kismi kirp; olcek (in / eksen birimi) degismez
    ax.set_ylim(alt, 1)
    fig.set_size_inches(SUTUN - 0.05, H * (1 - alt))
    kaydet(fig, "fig_v2_is_akisi")


def _yatay_eksen(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=IZGARA, lw=0.5)
    ax.set_axisbelow(True)


def dagilim():
    """Sekil 2: sinif basina kayit sayisi, egitim ve degerlendirme (dogrulama + test) bolumleri."""
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.dataset import veri_bol
    tr, va, te = veri_bol(ROOT / "data" / "processed" / "veri_listesi.parquet", ROOT)
    n_tr = tr.sinif.value_counts()
    n_ev = va.sinif.value_counts().add(te.sinif.value_counts(), fill_value=0)
    siniflar = sorted(EN26, key=lambda c: n_tr[c] + n_ev[c])          # en buyuk en ustte
    a = np.array([n_tr[c] for c in siniflar])
    b = np.array([n_ev[c] for c in siniflar])
    assert a.sum() + b.sum() == 15356, a.sum() + b.sum()
    fig = plt.figure(figsize=(SUTUN - 0.05, 3.9))
    ax = fig.add_axes([0.355, 0.115, 0.56, 0.875])
    yy = np.arange(len(siniflar))
    ax.barh(yy, a, 0.72, color=MAVI_ACIK, label="training")
    ax.barh(yy, b, 0.72, left=a, color=MAVI, label="validation + test")
    for yi, t in zip(yy, a + b):
        ax.text(t + 15, yi, f"{t:,}", va="center", fontsize=8, color=IKINCIL)
    ax.set_yticks(yy)
    ax.set_yticklabels([EN26[c] for c in siniflar])
    ax.set_ylim(-0.6, len(siniflar) - 0.4)
    ax.set_xlim(0, 2000)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xticklabels(["0", "500", "1,000", "1,500"])
    ax.set_xlabel("Recordings")
    _yatay_eksen(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, borderaxespad=0.2)
    kaydet(fig, "fig_v2_dagilim")
    veri = {"egitim": {c: int(n_tr[c]) for c in siniflar}, "degerlendirme": {c: int(n_ev[c]) for c in siniflar},
            "degerlendirme_min": int(b.min()), "degerlendirme_maks": int(b.max())}
    (CIKTI / "fig_v2_dagilim_veri.json").write_text(json.dumps(veri, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  en kucuk:", EN26[siniflar[0]], a[0] + b[0], "| en buyuk:", EN26[siniflar[-1]], a[-1] + b[-1],
          "| degerlendirme kaydi min-maks:", int(b.min()), int(b.max()))


def modeller():
    """Sekil 3: siniflandirici aileleri, dogrulama ve test macro-F1 (kayitli ciktilardan)."""
    def j(ad):
        return json.loads((ROOT / "results" / "v1" / ad).read_text(encoding="utf-8"))
    with open(ROOT / "outputs" / "model_selection_results.csv", encoding="utf-8") as f:
        klasik = {r["ad"]: (float(r["val_macro_f1"]), float(r["test_macro_f1"])) for r in csv.DictReader(f)}
    lgbm_hpo = json.loads((ROOT / "outputs" / "lgbm_metrics.json").read_text(encoding="utf-8"))
    cnn, lstm, ast = j("cnn_20260427_0312_sonuclar.json"), j("lstm_20260427_0732_sonuclar.json"), \
        j("ast_20260427_1430_sonuclar.json")
    fu = j("fusion_20260428_1924_sonuclar.json")
    assert fu["en_iyi_fusion"] == "val_f1_agirlik"
    satir = [("k-NN", *klasik["k-NN (k=5)"]), ("SVM (RBF)", *klasik["SVM (RBF)"]),
             ("Extra Trees", *klasik["Extra Trees"]), ("XGBoost", *klasik["XGBoost"]),
             ("HistGB", *klasik["Hist Gradient Boosting"]), ("LightGBM", *klasik["LightGBM"]),
             ("LightGBM (Optuna)", None, lgbm_hpo["test_macro_f1"]),
             ("BiLSTM", lstm["en_iyi_val_f1"], lstm["test_macro_f1"]),
             ("AST", ast["en_iyi_val_f1"], fu["bireysel_test"]["ast"]),
             ("CNN (ResNet-18)", cnn["en_iyi_val_f1"], cnn["test_macro_f1"]),
             ("Late fusion", fu["fusion_sonuclari"]["val_f1_agirlik"]["val"],
              fu["fusion_sonuclari"]["val_f1_agirlik"]["test"])]
    fig = plt.figure(figsize=(SUTUN - 0.05, 2.75))
    ax = fig.add_axes([0.30, 0.16, 0.44, 0.74])
    sutun = blended_transform_factory(fig.transFigure, ax.transData)
    yy = np.arange(len(satir))[::-1].astype(float)
    yy[6:] -= 0.4; yy[7:] -= 0.4; yy[10:] -= 0.4                   # aile araliklari
    for yi, (ad, v, t) in zip(yy, satir):
        en = ad.startswith("CNN")
        ax.plot([min(x for x in (v, t) if x is not None), max(x for x in (v, t) if x is not None)], [yi, yi],
                color=IZGARA, lw=1.2, zorder=1)
        if v is not None:
            ax.plot(v, yi, "o", ms=4.5, mfc="white", mec=TURUNCU, mew=1.2, zorder=3)
        ax.plot(t, yi, "o", ms=4.5, color=MAVI, mec=MAVI, zorder=2)
        for x, s in ((0.765, "—" if v is None else f"{v:.3f}"), (0.885, f"{t:.3f}")):
            ax.text(x, yi, s, transform=sutun, va="center", fontsize=8, color=MUREKKEP if en else IKINCIL,
                    fontweight="bold" if en else "normal")
    ust = yy[0] + 1.0
    ax.text(0.765, ust, "val.", transform=sutun, fontsize=8, color=IKINCIL)
    ax.text(0.885, ust, "test", transform=sutun, fontsize=8, color=IKINCIL)
    for yc in ((yy[5] + yy[6]) / 2, (yy[6] + yy[7]) / 2, (yy[9] + yy[10]) / 2):
        ax.axhline(yc, color=IZGARA, lw=0.5, ls=(0, (2, 2)))
    ax.set_yticks(yy)
    ax.set_yticklabels([s[0] for s in satir])
    ax.get_yticklabels()[9].set_fontweight("bold")
    ax.set_ylim(yy[-1] - 0.7, yy[0] + 0.7)
    ax.set_xlim(0.25, 0.70)
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("Macro-F1 (26 classes)")
    _yatay_eksen(ax)
    el = [Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=TURUNCU, mew=1.2, label="validation"),
          Line2D([], [], ls="", marker="o", ms=4.5, color=MAVI, label="test")]
    fig.legend(handles=el, loc="upper left", bbox_to_anchor=(0.29, 1.0), ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.5, borderaxespad=0.0)
    kaydet(fig, "fig_v2_modeller")
    print("  ", [(s[0], s[1], s[2]) for s in satir])


def sinif_f1():
    """Sekil 4: CNN baseline sinif F1 (seed 42, test; ilk gonderimle ayni kaynak); artirilan gruplar isaretli."""
    d = json.loads((ROOT / "results" / "v1" / "iteratif_loop_coherent_20260604_0320.json").read_text(encoding="utf-8"))
    f1 = d["cnn"]["baseline"]["sinif_f1"]
    makro = float(np.mean(list(f1.values())))
    assert len(f1) == 26 and abs(makro - 0.6716) < 5e-4, makro
    siniflar = sorted(f1, key=f1.get)                                    # en yuksek en ustte
    renk = {**{c: MAVI for c in GRUP_A}, **{c: TURUNCU for c in GRUP_B}, "diger_hayvanlar": IKINCIL,
            "acil_servis": "#8a8985"}
    fig = plt.figure(figsize=(SUTUN - 0.05, 4.4))
    ax = fig.add_axes([0.355, 0.20, 0.56, 0.79])
    yy = np.arange(len(siniflar))
    ax.barh(yy, [f1[c] for c in siniflar], 0.72, color=[renk.get(c, ACIK_GRI) for c in siniflar])
    for yi, c in zip(yy, siniflar):
        ax.text(f1[c] + 0.01, yi, f"{f1[c]:.3f}", va="center", fontsize=8, color=IKINCIL, zorder=4,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.axvline(makro, color=MUREKKEP, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.text(makro + 0.01, len(siniflar) - 0.2, f"macro-F1 {makro:.3f}", fontsize=8, color=MUREKKEP, va="bottom")
    ax.set_yticks(yy)
    ax.set_yticklabels([EN26[c] for c in siniflar])
    ax.set_ylim(-0.6, len(siniflar) + 0.6)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("F1 (CNN baseline, seed 42, test)")
    _yatay_eksen(ax)
    el = [Patch(color=MAVI, label="Group A"), Patch(color=TURUNCU, label="Group B"),
          Patch(color=IKINCIL, label="control"), Patch(color="#8a8985", label="emergency service"),
          Patch(color=ACIK_GRI, label="not augmented")]
    fig.legend(handles=el, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    kaydet(fig, "fig_v2_sinif_f1")


def kopya():
    """Sekil 5: sabit uretim seed'inin yol actigi kopyalar. Duzeltme oncesi dosyalar (data/_dup_backup_20260531)
    MD5 ile sayilir; sonuc results/v2/kopya_oncesi_sayim.json'a yazilir (klasor yoksa oradan okunur)."""
    siniflar = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]
    turlar = {"R1": "synthetic", "R2": "synthetic_tur2", "R3": "synthetic_tur3", "R4": "synthetic_tur4"}
    kayit = ROOT / "results" / "v2" / "kopya_oncesi_sayim.json"
    yedek = ROOT / "data" / "_dup_backup_20260531"
    if yedek.exists():
        say = {}
        for r, klasor in turlar.items():
            say[r] = {}
            for c in siniflar:
                dosyalar = sorted((yedek / klasor / c).glob("*.wav"))
                say[r][c] = {"dosya": len(dosyalar),
                             "benzersiz": len({hashlib.md5(f.read_bytes()).hexdigest() for f in dosyalar})}
        kayit.write_text(json.dumps({"kaynak": "data/_dup_backup_20260531 (sabit seed, duzeltme oncesi)",
                                     "olcut": "MD5", "sayim": say}, ensure_ascii=False, indent=1), encoding="utf-8")
    say = json.loads(kayit.read_text(encoding="utf-8"))["sayim"]
    assert all(say[r][c]["dosya"] == 50 for r in say for c in siniflar)
    fig = plt.figure(figsize=(SUTUN - 0.05, 2.3))
    ax = fig.add_axes([0.13, 0.25, 0.85, 0.62])
    tonlar = ["#cfe0f6", "#8fb8ec", "#2a78d6", "#14467f"]
    x = np.arange(len(siniflar)); w = 0.19
    for i, r in enumerate(turlar):
        v = [say[r][c]["benzersiz"] for c in siniflar]
        ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=tonlar[i], label=r)
    ax.axhline(50, color=MUREKKEP, lw=0.8, ls=(0, (3, 2)))
    ax.text(len(siniflar) - 0.5, 48, "50 clips generated per class; all 50 unique after per-clip seeding",
            ha="right", va="top", fontsize=8, color=MUREKKEP)
    ax.set_xticks(x)
    ax.set_xticklabels(["construction", "religious\nplace", "emergency\nservice", "other\nanimals"])
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("Unique clips (MD5)")
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    ax.grid(axis="y", color=IZGARA, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncol=4, frameon=False, handlelength=1.0,
              columnspacing=0.9, title="Fixed-seed generation round", title_fontsize=8, alignment="left")
    kaydet(fig, "fig_v2_kopya")
    print("  ", {r: [say[r][c]["benzersiz"] for c in siniflar] for r in say})


SEKILLER = {"is_akisi": lambda d: is_akisi(), "dagilim": lambda d: dagilim(), "modeller": lambda d: modeller(),
            "sinif_f1": lambda d: sinif_f1(), "kopya": lambda d: kopya(), "spektrogram": lambda d: spektrogram(),
            "classwise": lambda d: (classwise(), classwise_1c()), "classwise_1c": lambda d: classwise_1c(),
            "doz": doz, "forest": lambda d: (forest(d), forest_1c(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece", nargs="+", choices=list(SEKILLER))
    a = ap.parse_args()
    d = json.loads((ROOT / "results" / "v2" / "ikincil_analiz_tam.json").read_text(encoding="utf-8"))
    for ad in a.sadece or [k for k in SEKILLER if k != "classwise_1c"]:
        SEKILLER[ad](d)


if __name__ == "__main__":
    main()
