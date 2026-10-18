"""
Revizyon (v2) sekilleri — IEEE Access sutun genisligi, >= 8 pt yazi (R4.6). Ingilizce etiketler.

  fig_v2_forest   : on kayitli birincil (P1) + ikincil aile (11), dF1 ve %95 GA (ilk gonderimdeki
                    Sekil 9 sol panelinin yerine: McNemar yerine kayit duzeyi bootstrap)
  fig_v2_classwise: konfigurasyon x sinif dF1 isi haritasi (Sekil 9 sag panelinin yerine; R4.3: degerler
                    F1 degil F1 FARKI)
  fig_v2_dose     : doz-cevap (Grup A ve kontrol; seed 42-44)

Renk: referans paletin dogrulanmis ilk iki kategorik slotu (mavi #2a78d6, turuncu #eb6834) + gri tonlu
baski icin ikincil kodlama (isaretci + cizgi turu); isi haritasi mavi <-> kirmizi, notr gri orta nokta.
Girdi: results/v2/ikincil_analiz_tam.json (+ sinif bazinda noktalar 20_ikincil_aile fonksiyonlariyla).
Kullanim: python scripts/23_sekiller_v2.py  ->  results/v2/sekiller/*.pdf, *.png
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
CIKTI = ROOT / "results" / "v2" / "sekiller"
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


def kaydet(fig, ad):
    CIKTI.mkdir(parents=True, exist_ok=True)
    for uz in ("pdf", "png"):
        fig.savefig(CIKTI / f"{ad}.{uz}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("kaydedildi:", CIKTI / f"{ad}.pdf")


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
    kaydet(fig, "fig_v2_forest")


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
    fig, ax = plt.subplots(figsize=(3.35, 3.7))
    y = np.arange(len(satir))[::-1].astype(float)
    y[1:] -= 0.5
    for yi, (etiket, v, ph, birincil) in zip(y, satir):
        renk = MAVI if birincil else IKINCIL
        ax.plot(v["ga95"], [yi, yi], color=renk, lw=1.6 if birincil else 1.1, solid_capstyle="round", zorder=2)
        ax.plot(v["delta_f1"], yi, marker="o", ms=5.5 if birincil else 4.5, color=renk, mec="white", mew=0.8, zorder=3)
        p = v["p_bootstrap"]
        ax.text(1.03, yi, f"{p:.3f}" if p < 0.1 else f"{p:.2f}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=8, color=MUREKKEP if birincil else IKINCIL)
        ax.text(1.27, yi, "—" if ph is None else f"{ph:.2f}", transform=ax.get_yaxis_transform(), va="center",
                ha="left", fontsize=8, color=MUREKKEP if birincil else IKINCIL)
    ax.text(1.03, y[0] + 1.0, "p", transform=ax.get_yaxis_transform(), ha="left", fontsize=8, color=IKINCIL)
    ax.text(1.27, y[0] + 1.0, "Holm", transform=ax.get_yaxis_transform(), ha="left", fontsize=8, color=IKINCIL)
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
    fig, ax = plt.subplots(figsize=(3.35, 3.3))
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
    for y0, y1, ad in ((0, 2, "A"), (3, 5, "B"), (7, 7, "Ctrl")):
        ax.text(-0.5 - 1.95, (y0 + y1) / 2, ad, ha="center", va="center", fontsize=8, color=IKINCIL, rotation=90)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, aspect=30)
    cb.set_label("ΔF1 vs. paired baseline")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
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
    kaydet(fig, "fig_v2_classwise")


def doz(d):
    dz = d["doz_cevap"]["dozlar"]
    x = np.array([25, 50, 100, 200])
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), frameon=False, ncol=2, handlelength=2.4,
              columnspacing=1.2)
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
    MAVI_ACIK, TURUNCU_ACIK, GRI_ACIK = "#e3eefb", "#fde8dd", "#f0efec"
    fig, ax = plt.subplots(figsize=(3.35, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def kutu(x, y, w, h, baslik, metin, zemin, kenar=IKINCIL, kalin=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=zemin, ec=kenar,
                                    lw=kalin))
        ax.text(x + w / 2, y + h - 0.012, baslik, ha="center", va="top", fontsize=8, fontweight="bold", color=MUREKKEP)
        ax.text(x + w / 2, y + h - 0.042, metin, ha="center", va="top", fontsize=7, color=MUREKKEP, linespacing=1.15)

    def ok(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=IKINCIL, lw=0.8, mutation_scale=7))

    kutu(0.10, 0.915, 0.80, 0.078, "BBC Sound Effects Archive",
         "16,008 unique recordings; 15,356 in 26 classes\n(ordered keyword taxonomy)", GRI_ACIK)
    ok(0.5, 0.915, 0.5, 0.897)
    kutu(0.04, 0.807, 0.92, 0.09, "Preprocessing and splitting",
         "16 kHz · 5-s segments · log-Mel 128 × 313\nfile-level split (seed 42): 10,748 / 2,304 / 2,304", GRI_ACIK)
    ok(0.255, 0.807, 0.255, 0.787); ok(0.745, 0.807, 0.745, 0.787)
    kutu(0.02, 0.66, 0.47, 0.127, "A) Classifier selection",
         "LightGBM · BiLSTM · AST\nResNet-18 · late fusion\nselected on validation:\nResNet-18", MAVI_ACIK)
    kutu(0.51, 0.66, 0.47, 0.127, "B) CLAP fine-tuning",
         "training partition only · fp32\nCycle 1: BBC\nCycle 2: BBC + synthetic", TURUNCU_ACIK)
    ok(0.745, 0.66, 0.745, 0.64)
    kutu(0.51, 0.53, 0.47, 0.11, "AudioLDM2 generation",
         "H1 · C2′ · C3′ (TR) · C4′\n50 clips/class\nGroups A, B, control", TURUNCU_ACIK)
    ok(0.255, 0.66, 0.255, 0.475); ok(0.745, 0.53, 0.745, 0.475)
    kutu(0.04, 0.36, 0.92, 0.115, "Augmented training",
         "ResNet-18 (frozen BEATs check) · paired seeds (5 or 3)\ncontent control (label shift) · SpecAugment\n"
         "SpecAugment + C4′ · dose 25–200 clips", GRI_ACIK)
    ok(0.5, 0.36, 0.5, 0.34)
    kutu(0.04, 0.18, 0.92, 0.16, "Evaluation",
         "final-epoch models · validation + test (4,608 rec.)\nrecording-level cluster bootstrap\n"
         "pre-registered P1 · Holm-corrected secondary family\nFAD · CLAP score · classifier validity · memorization",
         GRI_ACIK)
    ok(0.5, 0.18, 0.5, 0.16)
    kutu(0.04, 0.005, 0.92, 0.155, "Main finding",
         "+0.018 macro-F1 for acoustically distinctive classes\n(95% CI 0.004–0.033, p = 0.010); grows with dose\n"
         "no gain for lowest-F1 classes, control or BEATs\nno added benefit of CLAP fine-tuning", "#ffffff",
         kenar=MAVI, kalin=1.4)
    kaydet(fig, "fig_v2_is_akisi")


def main():
    d = json.loads((ROOT / "results" / "v2" / "ikincil_analiz_tam.json").read_text(encoding="utf-8"))
    is_akisi()
    forest(d)
    doz(d)
    classwise()
    spektrogram()


if __name__ == "__main__":
    main()
