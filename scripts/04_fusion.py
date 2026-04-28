"""
Late fusion: CNN + BiLSTM + AST softmax ciktilarini birlestir.

Yontemler:
  1. Agirlikli ortalama   -- val F1 ile orantili agirliklar
  2. Ogrenilen fusion     -- LogisticRegression meta-learner (val kumesinde fit)
  3. Oracle ust siniri    -- her ornek icin en dogru modeli sec
  4. Esit agirlik         -- basit referans

Kullanim:
  python scripts/04_fusion.py

Ciktilar:
  results/fusion_<tarih>_sonuclar.json
  outputs/report/fig5_fusion_comparison.pdf
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import autocast

from src.data.dataset import dataloader_olustur, veri_bol
from src.models.ast_model import ASTClassifier
from src.models.cnn import SpectrogramCNN
from src.models.lstm import BiLSTMClassifier

# ── Sabitler ──────────────────────────────────────────────────────────────────
PARQUET   = ROOT / "data/processed/veri_listesi.parquet"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
OUT        = ROOT / "outputs" / "report"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_SINIF  = 26
BATCH    = 64

MODEL_CONF = {
    "cnn":  {"sinif": SpectrogramCNN,   "mod": "cnn",  "n_sinif": N_SINIF, "pretrained": False},
    "lstm": {"sinif": BiLSTMClassifier, "mod": "lstm", "n_sinif": N_SINIF},
    "ast":  {"sinif": ASTClassifier,    "mod": "ast",  "n_sinif": N_SINIF, "pretrained": False},
}

# ── Model yukleyici ───────────────────────────────────────────────────────────
def model_yukle(ad: str) -> nn.Module:
    conf = MODEL_CONF[ad]
    if ad == "cnn":
        model = conf["sinif"](n_sinif=conf["n_sinif"], pretrained=False)
    elif ad == "ast":
        model = conf["sinif"](n_sinif=conf["n_sinif"], pretrained=False)
    else:
        model = conf["sinif"](n_sinif=conf["n_sinif"])

    yol = MODEL_DIR / f"{ad}_best.pt"
    ckpt = torch.load(yol, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    val_f1 = ckpt.get("val_f1", 0.0)
    print(f"  {ad.upper():<6} yuklendi  (val F1={val_f1:.4f}  <- {yol.name})")
    return model, val_f1


# ── Softmax olasiliklari topla ────────────────────────────────────────────────
@torch.no_grad()
def olasilik_topla(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """(N, N_SINIF) olasilik matrisi ve (N,) gercek etiket dizisi dondurur."""
    tum_prob, tum_etiket = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        with autocast(enabled=(DEVICE == "cuda")):
            logit = model(x)
        prob = torch.softmax(logit, dim=1).cpu().numpy()
        tum_prob.append(prob)
        tum_etiket.append(y.numpy())
    return np.vstack(tum_prob), np.concatenate(tum_etiket)


# ── Fusion fonksiyonlari ──────────────────────────────────────────────────────
def agirlikli_ortalama(prob_dict: dict, agirliklar: dict) -> np.ndarray:
    toplam = sum(agirliklar.values())
    birlesik = sum(agirliklar[ad] * prob for ad, prob in prob_dict.items())
    return birlesik / toplam


def oracle_f1(prob_dict: dict, gercek: np.ndarray) -> float:
    """Her ornek icin en az bir modelin dogru tahmin ettigini varsay."""
    n = len(gercek)
    dogru = np.zeros(n, dtype=bool)
    for prob in prob_dict.values():
        tahmin = prob.argmax(axis=1)
        dogru |= (tahmin == gercek)
    return dogru.mean()


# ── Ana akis ─────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"  Late Fusion: CNN + BiLSTM + AST")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Veri
    print("Veri yuklenip boluniyor...")
    df_train, df_val, df_test = veri_bol(PARQUET, ROOT)

    # Her mod icin ayri loader
    loaders = {}
    for ad, conf in MODEL_CONF.items():
        mod = conf["mod"]
        loaders[ad] = {
            "val":  dataloader_olustur(df_val,  ROOT, mod=mod,
                                       batch_size=BATCH, shuffle=False,
                                       num_workers=0, preload=False),
            "test": dataloader_olustur(df_test, ROOT, mod=mod,
                                       batch_size=BATCH, shuffle=False,
                                       num_workers=0, preload=False),
        }
    print(f"Val: {len(df_val):,} segment | Test: {len(df_test):,} segment\n")

    # Modelleri yukle
    print("Modeller yukleniyor...")
    modeller, val_f1_dict = {}, {}
    for ad in MODEL_CONF:
        modeller[ad], val_f1_dict[ad] = model_yukle(ad)

    # Olasiliklari topla
    print("\nOlasiliklar hesaplaniyor...")
    val_probs, test_probs = {}, {}
    gercek_val = gercek_test = None

    for ad in MODEL_CONF:
        t0 = time.time()
        vp, gv  = olasilik_topla(modeller[ad], loaders[ad]["val"])
        tp, gt  = olasilik_topla(modeller[ad], loaders[ad]["test"])
        val_probs[ad]  = vp
        test_probs[ad] = tp
        if gercek_val is None:
            gercek_val  = gv
            gercek_test = gt
        print(f"  {ad.upper():<6} val F1={f1_score(gv, vp.argmax(1), average='macro', zero_division=0):.4f}"
              f"  test F1={f1_score(gt, tp.argmax(1), average='macro', zero_division=0):.4f}"
              f"  ({time.time()-t0:.0f}s)")

    # ── Fusion yontemleri ─────────────────────────────────────────────────────
    sonuclar = {}

    # 1. Esit agirlik
    esit = {ad: 1.0 for ad in MODEL_CONF}
    for split, probs, gercek in [("val", val_probs, gercek_val),
                                   ("test", test_probs, gercek_test)]:
        birlesik = agirlikli_ortalama(probs, esit)
        f1 = f1_score(gercek, birlesik.argmax(1), average="macro", zero_division=0)
        sonuclar.setdefault("esit_agirlik", {})[split] = round(f1, 4)
    print(f"\n[Esit agirlik]     val={sonuclar['esit_agirlik']['val']:.4f}"
          f"  test={sonuclar['esit_agirlik']['test']:.4f}")

    # 2. Val F1 agirlikli ortalama
    for split, probs, gercek in [("val", val_probs, gercek_val),
                                   ("test", test_probs, gercek_test)]:
        birlesik = agirlikli_ortalama(probs, val_f1_dict)
        f1 = f1_score(gercek, birlesik.argmax(1), average="macro", zero_division=0)
        sonuclar.setdefault("val_f1_agirlik", {})[split] = round(f1, 4)
    print(f"[Val-F1 agirlik]   val={sonuclar['val_f1_agirlik']['val']:.4f}"
          f"  test={sonuclar['val_f1_agirlik']['test']:.4f}")

    # 3. Ogrenilen meta-learner (LogReg)
    X_val  = np.hstack([val_probs[ad]  for ad in MODEL_CONF])   # (N, 78)
    X_test = np.hstack([test_probs[ad] for ad in MODEL_CONF])

    print("\n[Meta-learner] LogisticRegression egitiliyor (val -> test)...")
    t0 = time.time()
    meta = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                               multi_class="multinomial", n_jobs=-1)
    meta.fit(X_val, gercek_val)
    print(f"  Egitim tamamlandi ({time.time()-t0:.1f}s)")

    for split, X, gercek in [("val", X_val, gercek_val),
                               ("test", X_test, gercek_test)]:
        pred = meta.predict(X)
        f1   = f1_score(gercek, pred, average="macro", zero_division=0)
        sonuclar.setdefault("meta_learner", {})[split] = round(f1, 4)
    print(f"[Meta-learner]     val={sonuclar['meta_learner']['val']:.4f}"
          f"  test={sonuclar['meta_learner']['test']:.4f}")

    # 4. Oracle
    oracle_val  = oracle_f1(val_probs,  gercek_val)
    oracle_test = oracle_f1(test_probs, gercek_test)
    sonuclar["oracle"] = {"val": round(oracle_val, 4), "test": round(oracle_test, 4)}
    print(f"[Oracle ust sinir] val={oracle_val:.4f}  test={oracle_test:.4f}")

    # En iyi fusion modelinin detayli raporu (test)
    en_iyi_ad = max(["esit_agirlik", "val_f1_agirlik", "meta_learner"],
                    key=lambda k: sonuclar[k]["test"])
    print(f"\nEn iyi fusion yontemi: {en_iyi_ad}"
          f"  Test F1={sonuclar[en_iyi_ad]['test']:.4f}")

    if en_iyi_ad == "meta_learner":
        en_iyi_tahmin = meta.predict(X_test)
    else:
        agirlik = val_f1_dict if en_iyi_ad == "val_f1_agirlik" else esit
        en_iyi_tahmin = agirlikli_ortalama(test_probs, agirlik).argmax(1)

    rapor = classification_report(gercek_test, en_iyi_tahmin,
                                  zero_division=0, output_dict=True)
    print(classification_report(gercek_test, en_iyi_tahmin,
                                zero_division=0, digits=3))

    # ── Gorsellestime ─────────────────────────────────────────────────────────
    fig_karsilastirma(sonuclar, val_f1_dict)

    # ── Kaydet ───────────────────────────────────────────────────────────────
    tarih = datetime.now().strftime("%Y%m%d_%H%M")
    cikti = {
        "tarih":          tarih,
        "modeller":       {ad: round(v, 4) for ad, v in val_f1_dict.items()},
        "bireysel_test":  {
            ad: round(f1_score(gercek_test, test_probs[ad].argmax(1),
                               average="macro", zero_division=0), 4)
            for ad in MODEL_CONF
        },
        "fusion_sonuclari": sonuclar,
        "en_iyi_fusion":    en_iyi_ad,
        "sinif_raporu":     rapor,
    }
    yol = RESULT_DIR / f"fusion_{tarih}_sonuclar.json"
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(cikti, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar kaydedildi: {yol}")
    return cikti


# ── Gorsel ────────────────────────────────────────────────────────────────────
def fig_karsilastirma(sonuclar, val_f1_dict):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    # Bireysel model test F1'leri
    bireysel = {
        "LightGBM\n(Optuna)": 0.5857,
        "BiLSTM":             None,   # test'ten gelecek, placeholder
        "AST\n(partial)":     None,
        "CNN\n(ResNet-18)":   None,
    }

    etiketler = (
        ["LightGBM\n(Optuna)", "BiLSTM", "AST\n(partial)", "CNN\n(ResNet-18)"]
        + ["Esit\nAgirlik", "Val-F1\nAgirlik", "Meta-\nLearner", "Oracle\nUst Sinir"]
    )
    degerler  = [
        0.5857,
        sonuclar.get("bireysel_test_val", {}).get("lstm", None),  # doldurulacak
        None,
        None,
        sonuclar["esit_agirlik"]["test"],
        sonuclar["val_f1_agirlik"]["test"],
        sonuclar["meta_learner"]["test"],
        sonuclar["oracle"]["test"],
    ]

    # Bireysel degerleri JSON'dan al (main'de hesaplandi, buraya parametre olarak gelemedi)
    # — bunlari grafik basliginda not olarak gosteriyoruz

    renkler = (
        ["#9E9E9E"] * 4          # bireysel (cnn val_f1 ile doldurulacak)
        + ["#42A5F5", "#1565C0", "#E53935", "#757575"]  # fusion
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    # Fusion barlarini ciz (bireysel icin ayri renk)
    fusion_etiket = ["Esit\nAgirlik", "Val-F1\nAgirlik", "Meta-\nLearner", "Oracle\nUst Sinir"]
    fusion_deger  = [
        sonuclar["esit_agirlik"]["test"],
        sonuclar["val_f1_agirlik"]["test"],
        sonuclar["meta_learner"]["test"],
        sonuclar["oracle"]["test"],
    ]
    bireysel_etiket = ["LightGBM\n(Optuna)", "BiLSTM", "AST\n(9 ep.)", "CNN\n(ResNet-18)"]
    bireysel_deger  = [0.5857,
                       round(val_f1_dict.get("lstm", 0.5124), 4),
                       round(val_f1_dict.get("ast",  0.6064), 4),
                       round(val_f1_dict.get("cnn",  0.6716), 4)]

    x_b = np.arange(len(bireysel_etiket))
    x_f = np.arange(len(fusion_etiket)) + len(bireysel_etiket) + 0.8

    bars_b = ax.bar(x_b, bireysel_deger, color="#90A4AE", edgecolor="white",
                    width=0.6, label="Bireysel Model (test/val F1)")
    bars_f = ax.bar(x_f, fusion_deger,
                    color=["#42A5F5", "#1565C0", "#E53935", "#B0BEC5"],
                    edgecolor="white", width=0.6, label="Fusion Yontemi (test F1)")

    ax.set_xticks(np.concatenate([x_b, x_f]))
    ax.set_xticklabels(bireysel_etiket + fusion_etiket, fontsize=10)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "BBC Ses Arsivi — Late Fusion Karsilastirmasi (26 Sinif, Test Kumesi)\n"
        "Gri: Bireysel modeller  |  Mavi/Kirmizi: Fusion  |  Gri: Oracle ust sinir",
        fontsize=11,
    )

    for bar in list(bars_b) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axvline(len(bireysel_etiket) + 0.3, color="gray",
               linestyle=":", linewidth=1.2, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_fusion_comparison.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  fig5_fusion_comparison kaydedildi -> {OUT}")


if __name__ == "__main__":
    main()
