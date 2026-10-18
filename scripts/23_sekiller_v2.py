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
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
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


def main():
    d = json.loads((ROOT / "results" / "v2" / "ikincil_analiz_tam.json").read_text(encoding="utf-8"))
    forest(d)
    doz(d)
    classwise()


if __name__ == "__main__":
    main()
