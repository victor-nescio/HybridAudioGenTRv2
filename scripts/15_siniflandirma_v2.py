"""
v2 siniflandirma kosusu — tek konfigurasyon x tek seed.

v1 ile ayni egitim proseduru (07_iterative_loop.egit_cnn: ResNet-18, AdamW, cosine,
label smoothing 0.1, AMP, 30 epoch, sabir 7) kullanilir; v2 farklari:
  - seed parametresi (agirlik baslatma, karistirma, SpecAugment maskeleri),
  - sentetik klipler birden fazla klasorden sinif bazinda secilebilir (--kaynak),
  - doz: sinif basina ilk N klip (uretim sirasina = seed numarasina gore; promptlar dengeli),
  - her kosuda test/val segment tahminleri + kayit kimlikleri kaydedilir
    (kayit duzeyi / karma etkili analiz icin),
  - epoch duzeyinde kesintiye dayanikli (tam durum; ayni komutla birebir devam),
  - sonuc dosyasi varsa kosu atlanir (gece kuyrugu tekrar calistirilabilir).

Ciktilar:
  results/v2/siniflandirma/<konfig>_s<seed>.json
  results/v2/tahminler/<konfig>_s<seed>.npz        (git disi)
  models/v2/cnn_<konfig>_s<seed>.pt                 (git disi)

Ornekler:
  # baseline (sentetik yok), seed 42 — v1 uzlastirma kosusu (test F1 = 0.6716 beklenir)
  python scripts/15_siniflandirma_v2.py --konfig baseline --seed 42

  # H1: Grup A + kontrol tutarli kosudan, Grup B ilk kosudan
  python scripts/15_siniflandirma_v2.py --konfig H1 --seed 42 \
      --kaynak data/synthetic_turcoh:kopek_sesleri,kus_sesleri,hava_olaylari,diger_hayvanlar \
      --kaynak data/synthetic:insaat_sesleri,dini_mekan,acil_servis --n_klip 50
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("iterloop", ROOT / "scripts" / "07_iterative_loop.py")
IL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IL)

SONUC_DIR  = ROOT / "results" / "v2" / "siniflandirma"
TAHMIN_DIR = ROOT / "results" / "v2" / "tahminler"
MODEL_DIR  = ROOT / "models" / "v2"
SEED_RE    = re.compile(r"_s(\d+)_")


def sentetik_sec(kaynaklar: list[str], n_klip: int | None) -> pd.DataFrame:
    """--kaynak 'klasor:sinif1,sinif2' girdilerinden sentetik egitim satirlari.
    Her sinif icin klipler uretim seed'ine gore siralanir, ilk n_klip alinir.
    Eksik klip varsa sessizce devam etmez, hata verir."""
    satirlar, gorulen = [], set()
    for k in kaynaklar:
        klasor, siniflar = k.split(":")
        for sinif in [s.strip() for s in siniflar.split(",") if s.strip()]:
            if sinif not in IL.SINIF_IDX:
                raise SystemExit(f"HATA: bilinmeyen sinif {sinif}")
            if sinif in gorulen:
                raise SystemExit(f"HATA: {sinif} birden fazla kaynakta")
            gorulen.add(sinif)
            wavlar = [w for w in (ROOT / klasor / sinif).glob("*.wav") if w.stat().st_size > 1000]
            wavlar.sort(key=lambda w: (int(SEED_RE.search(w.name).group(1)) if SEED_RE.search(w.name) else 10**9, w.name))
            if n_klip is not None:
                if len(wavlar) < n_klip:
                    raise SystemExit(f"HATA: {klasor}/{sinif} icinde {len(wavlar)} klip var, {n_klip} istendi")
                wavlar = wavlar[:n_klip]
            for w in wavlar:
                satirlar.append({"dosya": str(w.relative_to(ROOT)), "sinif": sinif,
                                 "sinif_idx": IL.SINIF_IDX[sinif], "n_segs": 1})
    return pd.DataFrame(satirlar, columns=["dosya", "sinif", "sinif_idx", "n_segs"])


def git_surumu() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "?"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--konfig", required=True, help="Etiket: baseline, H1, C2p, C3p, C4p, specaug, ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kaynak", action="append", default=[],
                   help="'klasor:sinif1,sinif2' (tekrarlanabilir). Verilmezse sentetik veri yok.")
    p.add_argument("--n_klip", type=int, default=None, help="Sinif basina klip sayisi (doz)")
    p.add_argument("--spec_augment", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)    # v1 kosulari 64 ile yapildi
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--zorla", action="store_true", help="Sonuc dosyasi olsa da yeniden kos")
    p.add_argument("--kesinti_epoch", type=int, default=None, help="(Yalniz test) kesinti taklidi")
    a = p.parse_args()

    ad = f"{a.konfig}_s{a.seed}"
    cikti = SONUC_DIR / f"{ad}.json"
    if cikti.exists() and not a.zorla:
        print(f"Atlaniyor: {cikti.name} zaten var (yeniden kosmak icin --zorla)")
        return
    for d in (SONUC_DIR, TAHMIN_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(f"  v2 siniflandirma: {ad}  (spec_augment={a.spec_augment}, n_klip={a.n_klip})")
    print("=" * 62)
    df_tr, df_vl, df_te = IL.veri_bol(IL.PARQUET_LISTE, ROOT)
    df_synt = sentetik_sec(a.kaynak, a.n_klip)
    if len(df_synt):
        print("  Sentetik: " + "  ".join(f"{s}={n}" for s, n in df_synt["sinif"].value_counts().sort_index().items()))
        df_tr = pd.concat([df_tr, df_synt], ignore_index=True)
    else:
        print("  Sentetik: yok (baseline)")

    sonuc = IL.egit_cnn(df_tr, df_vl, df_te, tag=ad, epochs=a.epochs, batch_size=a.batch_size,
                        lr=a.lr, patience=a.patience, spec_augment=a.spec_augment, seed=a.seed,
                        model_yolu=MODEL_DIR / f"cnn_{ad}.pt",
                        durum_yolu=MODEL_DIR / f"cnn_{ad}_durum.pt",
                        tahmin_dondur=True, kesinti_epoch=a.kesinti_epoch)

    tahminler = sonuc.pop("_tahminler")
    np.savez_compressed(TAHMIN_DIR / f"{ad}.npz", **tahminler)
    kayit = {
        "konfig": a.konfig, "seed": a.seed, "tarih": datetime.now().strftime("%Y%m%d_%H%M"),
        "git": git_surumu(), "argumanlar": vars(a),
        "sentetik_sayisi": df_synt["sinif"].value_counts().sort_index().to_dict(),
        "sentetik_dosyalar": df_synt["dosya"].tolist(),
        **sonuc,
    }
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=1)
    (MODEL_DIR / f"cnn_{ad}_durum.pt").unlink(missing_ok=True)   # tamamlandi; ara durum gereksiz
    print(f"\nKaydedildi: {cikti}  (test macro-F1={sonuc['test_f1']:.4f}, val={sonuc['val_f1']:.4f})")


if __name__ == "__main__":
    main()
