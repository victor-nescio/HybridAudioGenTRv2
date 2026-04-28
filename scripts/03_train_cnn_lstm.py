"""
CNN ve BiLSTM modellerini egitir.

Kullanim ornekleri:
    python scripts/03_train_cnn_lstm.py --model cnn
    python scripts/03_train_cnn_lstm.py --model lstm
    python scripts/03_train_cnn_lstm.py --model simple_cnn --no_pretrained
    python scripts/03_train_cnn_lstm.py --model simple_lstm --epochs 50

Ciktilar:
    models/cnn_best.pt    (veya lstm_best.pt, simple_cnn_best.pt, ...)
    results/cnn_tarih_sonuclar.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import dataloader_olustur, veri_bol
from src.models.ast_model import ASTClassifier
from src.models.cnn import SimpleCNN, SpectrogramCNN
from src.models.lstm import BiLSTMClassifier, SimpleLSTM

# ── Sabitler ──────────────────────────────────────────────────────────────────
PARQUET = ROOT / "data/processed/veri_listesi.parquet"
MODEL_DIR  = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONF = {
    "cnn":        {"mod": "cnn",  "sinif": SpectrogramCNN,    "pretrained": True},
    "simple_cnn": {"mod": "cnn",  "sinif": SimpleCNN,         "pretrained": False},
    "lstm":       {"mod": "lstm", "sinif": BiLSTMClassifier,  "pretrained": False},
    "simple_lstm":{"mod": "lstm", "sinif": SimpleLSTM,        "pretrained": False},
    "ast":        {"mod": "ast",  "sinif": ASTClassifier,     "pretrained": True},
}


# ── Argüman ayrıştırıcı ───────────────────────────────────────────────────────
def arguman_ayristir():
    p = argparse.ArgumentParser(description="CNN / BiLSTM egitici")
    p.add_argument("--model",        default="cnn",
                   choices=list(MODEL_CONF.keys()),
                   # ast icin onerilen: --lr 1e-4 --epochs 20 --batch_size 16
                   help="Model turu (cnn | simple_cnn | lstm | simple_lstm)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=7,
                   help="Val loss iyilesmediyse erken durdurma epoch sayisi")
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--no_pretrained", action="store_true",
                   help="CNN icin pretrained agirliklari kullanma")
    p.add_argument("--n_sinif",      type=int,   default=26)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


# ── Model fabrikasi ───────────────────────────────────────────────────────────
def model_olustur(args) -> nn.Module:
    conf   = MODEL_CONF[args.model]
    sinif  = conf["sinif"]
    use_pt = conf["pretrained"] and not args.no_pretrained

    if args.model == "cnn":
        return sinif(n_sinif=args.n_sinif, pretrained=use_pt)
    elif args.model == "ast":
        return sinif(n_sinif=args.n_sinif, pretrained=use_pt)
    elif args.model == "simple_cnn":
        return sinif(n_sinif=args.n_sinif)
    else:
        return sinif(n_sinif=args.n_sinif)


# ── Tek epoch egitim ──────────────────────────────────────────────────────────
def egit_epoch(model, loader, optimizer, kayip_fn, scaler, device):
    model.train()
    toplam_kayip = 0.0
    dogru = toplam = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device == "cuda")):
            cikis = model(x)
            kayip = kayip_fn(cikis, y)

        scaler.scale(kayip).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        toplam_kayip += kayip.item() * x.size(0)
        tahmin = cikis.argmax(dim=1)
        dogru  += (tahmin == y).sum().item()
        toplam += x.size(0)

    return toplam_kayip / toplam, dogru / toplam


# ── Degerlendirme ─────────────────────────────────────────────────────────────
def degerlendir(model, loader, kayip_fn, device):
    model.eval()
    toplam_kayip = 0.0
    tahminler, gercekler = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast(enabled=(device == "cuda")):
                cikis = model(x)
                kayip = kayip_fn(cikis, y)
            toplam_kayip += kayip.item() * x.size(0)
            tahminler.extend(cikis.argmax(1).cpu().tolist())
            gercekler.extend(y.cpu().tolist())

    n = len(gercekler)
    f1 = f1_score(gercekler, tahminler, average="macro", zero_division=0)
    return toplam_kayip / n, f1, tahminler, gercekler


# ── Ana egitim dongüsü ────────────────────────────────────────────────────────
def egit(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Model  : {args.model.upper()}")
    print(f"  Epochs : {args.epochs}   Batch : {args.batch_size}")
    print(f"  LR     : {args.lr}       Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Veri ──────────────────────────────────────────────────────────────────
    mod = MODEL_CONF[args.model]["mod"]
    print("Veri yuklenip boluniyor...")
    df_train, df_val, df_test = veri_bol(PARQUET, ROOT)

    dl_train = dataloader_olustur(df_train, ROOT, mod=mod,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  preload=True)
    dl_val   = dataloader_olustur(df_val,   ROOT, mod=mod,
                                  batch_size=args.batch_size * 2,
                                  shuffle=False,
                                  num_workers=0,
                                  preload=False)
    dl_test  = dataloader_olustur(df_test,  ROOT, mod=mod,
                                  batch_size=args.batch_size * 2,
                                  shuffle=False,
                                  num_workers=0,
                                  preload=False)

    print(f"Train: {len(dl_train.dataset):,} segment | "
          f"Val: {len(dl_val.dataset):,} | Test: {len(dl_test.dataset):,}")

    # ── Model, optimizer, scheduler ───────────────────────────────────────────
    model = model_olustur(args).to(DEVICE)
    param_sayisi = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Egitilecek parametre: {param_sayisi:,}\n")

    sinif_agirliklari = None  # ihtiyac duyulursa hesaplanabilir
    kayip_fn  = nn.CrossEntropyLoss(weight=sinif_agirliklari, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)
    scaler    = GradScaler(enabled=(DEVICE == "cuda"))

    # ── Egitim dongüsü ────────────────────────────────────────────────────────
    en_iyi_val_f1  = 0.0
    patience_sayac = 0
    tarih          = []
    model_yolu     = MODEL_DIR / f"{args.model}_best.pt"

    basla = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_kayip, train_acc = egit_epoch(model, dl_train, optimizer,
                                            kayip_fn, scaler, DEVICE)
        val_kayip, val_f1, _, _ = degerlendir(model, dl_val, kayip_fn, DEVICE)
        scheduler.step()

        sure = time.time() - t0
        lr_simdi = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_kayip:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_kayip:.4f} val_f1={val_f1:.4f}  "
              f"lr={lr_simdi:.2e}  {sure:.0f}s")

        tarih.append({
            "epoch": epoch,
            "train_loss": round(train_kayip, 5),
            "train_acc":  round(train_acc,   4),
            "val_loss":   round(val_kayip,   5),
            "val_f1":     round(val_f1,      4),
        })

        # En iyi model kaydet
        if val_f1 > en_iyi_val_f1:
            en_iyi_val_f1 = val_f1
            patience_sayac = 0
            torch.save({
                "epoch":      epoch,
                "model_adi":  args.model,
                "model_state": model.state_dict(),
                "val_f1":     val_f1,
                "args":       vars(args),
            }, model_yolu)
            print(f"  -> Model kaydedildi (val F1={val_f1:.4f})")
        else:
            patience_sayac += 1
            if patience_sayac >= args.patience:
                print(f"\nErken durdurma: {args.patience} epoch iyilesme olmadi.")
                break

    toplam_sure = time.time() - basla
    print(f"\nEgitim tamamlandi: {toplam_sure/60:.1f} dakika")

    # ── Test degerlendirmesi ───────────────────────────────────────────────────
    print("\nEn iyi model test kumesinde degerlendiriliyor...")
    ckpt = torch.load(model_yolu, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    test_kayip, test_f1, test_tahmin, test_gercek = degerlendir(
        model, dl_test, kayip_fn, DEVICE)
    rapor = classification_report(test_gercek, test_tahmin,
                                  zero_division=0, output_dict=True)

    print(f"\nTest Macro F1 : {test_f1:.4f}")
    print(f"Test Loss     : {test_kayip:.4f}")
    print("\nSinif bazli rapor (ilk 10):")
    print(classification_report(test_gercek, test_tahmin,
                                zero_division=0, digits=3))

    # ── Sonuclari kaydet ──────────────────────────────────────────────────────
    from datetime import datetime
    tarih_damgasi = datetime.now().strftime("%Y%m%d_%H%M")
    sonuc = {
        "model":        args.model,
        "tarih":        tarih_damgasi,
        "egitim_dakika": round(toplam_sure / 60, 1),
        "en_iyi_val_f1": round(en_iyi_val_f1, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_loss":    round(test_kayip, 4),
        "args":         vars(args),
        "epoch_tarih":  tarih,
        "sinif_raporu": rapor,
    }
    sonuc_yolu = RESULT_DIR / f"{args.model}_{tarih_damgasi}_sonuclar.json"
    with open(sonuc_yolu, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)

    print(f"\nSonuclar kaydedildi: {sonuc_yolu}")
    print(f"Model kaydedildi   : {model_yolu}")
    return sonuc


# ── Giris noktasi ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = arguman_ayristir()
    egit(args)
