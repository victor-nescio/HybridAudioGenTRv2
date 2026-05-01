"""
Iteratif veri artirma dongusu — sentetik ses + CNN yeniden egitimi.

Akis:
  1. Orijinal test bolunmesini (seed=42) sabitle — kiyaslama adil olmali
  2. data/synthetic/<sinif>/ altindaki WAV'lari egitim setine ekle
  3. CNN'i sifirdan egit (pretrained ResNet-18 + finetuning)
  4. Orijinal test seti uzerinde F1 karsilastir (oncesi / sonrasi)
  5. Sonuclari results/iteratif_loop_<tarih>.json'a kaydet

Kullanim:
    python scripts/07_iterative_loop.py
    python scripts/07_iterative_loop.py --siniflar insaat_sesleri dini_mekan
    python scripts/07_iterative_loop.py --epochs 20 --patience 5
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import (BBCSegmentDataset, dataloader_olustur,
                               veri_bol, SR, T_MAX)
from src.models.cnn import SpectrogramCNN

# ── Sabitler ──────────────────────────────────────────────────────────────────
PARQUET      = ROOT / "data" / "processed" / "veri_listesi.parquet"
SENTETIK_DIR = ROOT / "data" / "synthetic"
MODEL_DIR    = ROOT / "models"
RESULT_DIR   = ROOT / "results"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
N_SINIF      = 26
SEED         = 42

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
SINIF_IDX = {s: i for i, s in enumerate(SINIF_SIRASI)}


# ── Sentetik DataFrame olustur ────────────────────────────────────────────────
def sentetik_df_olustur(siniflar: list[str]) -> pd.DataFrame:
    """
    data/synthetic/<sinif>/*.wav dosyalarindan veri_listesi formatinda
    DataFrame olusturur. Her sentetik dosya = 1 segment.
    """
    satirlar = []
    for sinif in siniflar:
        sinif_dir = SENTETIK_DIR / sinif
        if not sinif_dir.exists():
            continue
        sinif_idx = SINIF_IDX.get(sinif)
        if sinif_idx is None:
            continue
        for wav in sorted(sinif_dir.glob("*.wav")):
            if wav.stat().st_size < 1000:   # 0-byte yarim dosyalari atla
                continue
            # dosya yolunu ROOT'a gore relative tut (dataset.py ile uyumlu)
            try:
                rel = wav.relative_to(ROOT)
            except ValueError:
                rel = wav
            satirlar.append({
                "dosya":     str(rel),
                "sinif":     sinif,
                "sinif_idx": sinif_idx,
                "n_segs":    1,
                "sentetik":  True,
            })
    df = pd.DataFrame(satirlar)
    print(f"  Sentetik: {len(df)} satir ({', '.join(f'{s}={len(df[df.sinif==s])}' for s in siniflar)})")
    return df


# ── Egitim / Degerlendirme (03_train_cnn_lstm.py'den alindi) ──────────────────
def egit_epoch(model, loader, optimizer, kayip_fn, scaler):
    model.train()
    toplam_kayip = 0.0
    dogru = toplam = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(DEVICE == "cuda")):
            kayip = kayip_fn(model(x), y)
        scaler.scale(kayip).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        toplam_kayip += kayip.item() * x.size(0)
        toplam += x.size(0)
    return toplam_kayip / toplam


def degerlendir(model, loader, kayip_fn):
    model.eval()
    toplam_kayip = 0.0
    tahminler, gercekler = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with autocast(enabled=(DEVICE == "cuda")):
                cikis  = model(x)
                kayip  = kayip_fn(cikis, y)
            toplam_kayip += kayip.item() * x.size(0)
            tahminler.extend(cikis.argmax(1).cpu().tolist())
            gercekler.extend(y.cpu().tolist())
    n  = len(gercekler)
    f1 = f1_score(gercekler, tahminler, average="macro", zero_division=0)
    return toplam_kayip / n, f1, tahminler, gercekler


def egit_cnn(df_train, df_val, df_test, tag: str, epochs: int,
             batch_size: int, lr: float, patience: int) -> dict:
    """
    SpectrogramCNN'i verilen bolunme uzerinde egitir.
    Returns: {'val_f1', 'test_f1', 'sinif_f1': {...}, 'tarih': [...], 'sure_dk'}
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_yolu = MODEL_DIR / f"cnn_{tag}_best.pt"

    dl_train = dataloader_olustur(df_train, ROOT, mod="cnn",
                                  batch_size=batch_size, num_workers=4,
                                  preload=False)
    dl_val   = dataloader_olustur(df_val,   ROOT, mod="cnn",
                                  batch_size=batch_size * 2, shuffle=False,
                                  num_workers=0, preload=False)
    dl_test  = dataloader_olustur(df_test,  ROOT, mod="cnn",
                                  batch_size=batch_size * 2, shuffle=False,
                                  num_workers=0, preload=False)

    print(f"  Train={len(dl_train.dataset):,}  Val={len(dl_val.dataset):,}  "
          f"Test={len(dl_test.dataset):,}")

    model     = SpectrogramCNN(n_sinif=N_SINIF, pretrained=True).to(DEVICE)
    kayip_fn  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    scaler    = GradScaler(enabled=(DEVICE == "cuda"))

    en_iyi_val_f1  = 0.0
    patience_sayac = 0
    tarih          = []
    basla          = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = egit_epoch(model, dl_train, optimizer, kayip_fn, scaler)
        val_loss, val_f1, _, _ = degerlendir(model, dl_val, kayip_fn)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_f1={val_f1:.4f}  "
              f"{time.time()-t0:.0f}s")

        tarih.append({"epoch": epoch,
                      "train_loss": round(train_loss, 5),
                      "val_f1":     round(val_f1, 4)})

        if val_f1 > en_iyi_val_f1:
            en_iyi_val_f1  = val_f1
            patience_sayac = 0
            torch.save({"model_state": model.state_dict(),
                        "val_f1": val_f1, "epoch": epoch}, model_yolu)
        else:
            patience_sayac += 1
            if patience_sayac >= patience:
                print(f"  Erken durdurma (epoch {epoch})")
                break

    # En iyi checkpoint yukle
    ckpt = torch.load(model_yolu, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    _, test_f1, tahmin, gercek = degerlendir(model, dl_test, kayip_fn)
    rapor = classification_report(gercek, tahmin,
                                  target_names=SINIF_SIRASI,
                                  output_dict=True, zero_division=0)

    sinif_f1 = {SINIF_SIRASI[i]: round(rapor[SINIF_SIRASI[i]]["f1-score"], 4)
                for i in range(N_SINIF)}

    sure_dk = (time.time() - basla) / 60
    print(f"  [{tag}] val_F1={en_iyi_val_f1:.4f}  test_F1={test_f1:.4f}  "
          f"({sure_dk:.1f} dk)")

    return {
        "val_f1":   round(en_iyi_val_f1, 4),
        "test_f1":  round(test_f1,       4),
        "sinif_f1": sinif_f1,
        "tarih":    tarih,
        "sure_dk":  round(sure_dk, 1),
    }


# ── Ana ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siniflar", nargs="+", default=None,
                        help="Sentetik veri eklenecek siniflar. "
                             "Varsayilan: data/synthetic/ altindaki tum siniflar")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=7)
    parser.add_argument("--sadece_aug", action="store_true",
                        help="Baseline CNN'i atla (zaten varsa), sadece augmented egit")
    args = parser.parse_args()

    # Sentetik siniflar
    if args.siniflar:
        hedef_siniflar = args.siniflar
    else:
        hedef_siniflar = [
            d.name for d in SENTETIK_DIR.iterdir()
            if d.is_dir() and any(
                f.stat().st_size > 1000
                for f in d.glob("*.wav")
            )
        ]

    if not hedef_siniflar:
        print("HATA: data/synthetic/ altinda gecerli WAV bulunamadi.")
        print("Once scripts/05_generate_audio.py calistirin.")
        sys.exit(1)

    print("=" * 60)
    print("  Iteratif Veri Artirma Dongusu — Tur 1")
    print(f"  Cihaz   : {DEVICE}")
    print(f"  Siniflar: {hedef_siniflar}")
    print(f"  Epochs  : {args.epochs}  Patience: {args.patience}")
    print("=" * 60)

    # ── Orijinal bolunme (seed=42 ile her zaman ayni) ─────────────────────────
    print("\nOrijinal veri bolunuyor...")
    df_train_org, df_val, df_test = veri_bol(PARQUET, ROOT)
    print(f"  Train={len(df_train_org):,}  Val={len(df_val):,}  "
          f"Test={len(df_test):,} (segment sayisi)")

    # ── 1. Baseline CNN (sentetik veri YOK) ───────────────────────────────────
    if not args.sadece_aug:
        print("\n[1/2] Baseline CNN egitiliyor (sentetik YOK)...")
        sonuc_baseline = egit_cnn(
            df_train_org, df_val, df_test,
            tag="baseline_iter",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
    else:
        # Mevcut cnn_best.pt'yi baseline olarak kullan
        print("\n[1/2] Baseline atlanıyor (--sadece_aug), mevcut cnn_best.pt kullanilacak...")
        kayip_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        model = SpectrogramCNN(n_sinif=N_SINIF, pretrained=False).to(DEVICE)
        ckpt_yolu = MODEL_DIR / "cnn_best.pt"
        if not ckpt_yolu.exists():
            print(f"HATA: {ckpt_yolu} bulunamadi. --sadece_aug olmadan calistirin.")
            sys.exit(1)
        ckpt = torch.load(ckpt_yolu, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        dl_test  = dataloader_olustur(df_test, ROOT, mod="cnn",
                                      batch_size=64, shuffle=False,
                                      num_workers=0, preload=False)
        _, test_f1, tahmin, gercek = degerlendir(model, dl_test, kayip_fn)
        rapor = classification_report(gercek, tahmin,
                                      target_names=SINIF_SIRASI,
                                      output_dict=True, zero_division=0)
        sinif_f1 = {SINIF_SIRASI[i]: round(rapor[SINIF_SIRASI[i]]["f1-score"], 4)
                    for i in range(N_SINIF)}
        sonuc_baseline = {
            "val_f1":   ckpt.get("val_f1", 0.0),
            "test_f1":  round(test_f1, 4),
            "sinif_f1": sinif_f1,
            "tarih":    [],
            "sure_dk":  0.0,
        }
        print(f"  Baseline test_F1={test_f1:.4f}")

    # ── 2. Augmented CNN (sentetik veri VAR) ──────────────────────────────────
    print("\n[2/2] Augmented CNN egitiliyor (sentetik VAR)...")
    df_sentetik = sentetik_df_olustur(hedef_siniflar)

    if len(df_sentetik) == 0:
        print("HATA: Sentetik DataFrame bos. "
              "scripts/05_generate_audio.py ciktilarini kontrol edin.")
        sys.exit(1)

    # Sentetik satirlari egitim setine ekle (val/test degismez)
    df_train_aug = pd.concat(
        [df_train_org, df_sentetik[["dosya", "sinif", "sinif_idx", "n_segs"]]],
        ignore_index=True,
    )
    print(f"  Augmented train: {len(df_train_org):,} + {len(df_sentetik):,} "
          f"= {len(df_train_aug):,} satir")

    sonuc_aug = egit_cnn(
        df_train_aug, df_val, df_test,
        tag="augmented_iter",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    # ── Karsilastirma ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Sonuc Karsilastirmasi:")
    print(f"  {'':30s}  Baseline   Augmented  Delta")
    print(f"  {'-'*55}")
    print(f"  {'Genel test F1 (macro)':30s}  "
          f"{sonuc_baseline['test_f1']:.4f}     "
          f"{sonuc_aug['test_f1']:.4f}     "
          f"{sonuc_aug['test_f1']-sonuc_baseline['test_f1']:+.4f}")
    print()
    for sinif in hedef_siniflar:
        b = sonuc_baseline["sinif_f1"].get(sinif, 0.0)
        a = sonuc_aug["sinif_f1"].get(sinif, 0.0)
        delta = a - b
        isaretci = "**" if delta > 0.01 else ("  " if delta >= 0 else "!!")
        print(f"  {isaretci}{sinif:<28}  {b:.4f}     {a:.4f}     {delta:+.4f}")
    print("=" * 60)

    # ── Kaydet ────────────────────────────────────────────────────────────────
    tarih_str = datetime.now().strftime("%Y%m%d_%H%M")
    cikti_yolu = RESULT_DIR / f"iteratif_loop_{tarih_str}.json"
    RESULT_DIR.mkdir(exist_ok=True)

    kayit = {
        "tarih":          tarih_str,
        "tur":            1,
        "hedef_siniflar": hedef_siniflar,
        "n_sentetik":     len(df_sentetik),
        "baseline":       sonuc_baseline,
        "augmented":      sonuc_aug,
        "delta_test_f1":  round(sonuc_aug["test_f1"] - sonuc_baseline["test_f1"], 4),
    }

    with open(cikti_yolu, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar kaydedildi: {cikti_yolu}")


if __name__ == "__main__":
    main()
