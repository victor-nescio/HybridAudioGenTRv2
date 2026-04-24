"""
data/raw/ altindaki WAV dosyalarindan ozellik cikarir.

Ciktilar:
  data/processed/lgbm_ozellikler.parquet  -- LGBM icin 467-dim vektorler
  data/processed/veri_listesi.parquet     -- CNN/AST icin dosya listesi

Kullanim:
  python scripts/01_extract_features.py              # her ikisi de
  python scripts/01_extract_features.py --mode lgbm  # sadece LGBM ozellikleri
  python scripts/01_extract_features.py --mode liste # sadece dosya listesi
  python scripts/01_extract_features.py --sr 22050   # farkli sample rate
  python scripts/01_extract_features.py --jobs 4     # CPU sayisi (varsayilan: hepsi)

Kaldigi yerden devam: --mode lgbm yeniden calistirildiginda tamamlanmis
batch'ler atlanir; yalnizca eksik parcalar hesaplanir.
"""
import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.labels import SINIFLAR
from src.data.preprocessing import onisle
from src.data.features import duz_ozellik_cikar, segmentle, SEGMENT_SN

ISLENMIS    = ROOT / "data" / "processed"
HAM         = ROOT / "data" / "raw"
BATCH_SIZE  = 500   # dosya/batch — batch bittikten sonra parquet'e yazilir

SINIF_INDEX = {sinif: i for i, sinif in enumerate(sorted(SINIFLAR.keys()))}


# ── Bu fonksiyon top-level olmali (joblib pickle icin) ───────────────────────
def _dosya_isle(wav_yolu_str: str, sinif: str, sr: int) -> tuple[list, int]:
    """
    Tek WAV dosyasini isle: yukle -> segmentle -> ozellik cikar.
    (satirlar_listesi, hata_sayisi) dondurur.
    """
    wav_yolu = Path(wav_yolu_str)
    satirlar = []
    try:
        y, sr_out = onisle(wav_yolu_str, sr=sr)
        segmentler = segmentle(y, sr_out, segment_sn=SEGMENT_SN)
        for seg_idx, seg in enumerate(segmentler):
            ozellikler = duz_ozellik_cikar(seg, sr_out)
            satir = {
                "sinif":       sinif,
                "sinif_idx":   SINIF_INDEX[sinif],
                "dosya":       str(wav_yolu.relative_to(ROOT)),
                "segment_idx": seg_idx,
            }
            for i, v in enumerate(ozellikler):
                satir[f"f{i}"] = v
            satirlar.append(satir)
        return satirlar, 0
    except Exception:
        return [], 1


# ── Ana fonksiyonlar ──────────────────────────────────────────────────────────
def wav_listesi_al() -> list[tuple[Path, str]]:
    dosyalar = []
    for sinif in sorted(SINIFLAR.keys()):
        sinif_yolu = HAM / sinif
        if not sinif_yolu.exists():
            continue
        for wav in sinif_yolu.rglob("*.wav"):
            dosyalar.append((wav, sinif))
    return dosyalar


def lgbm_ozelliklerini_cikar(dosyalar: list[tuple[Path, str]],
                              sr: int, n_jobs: int) -> pd.DataFrame:
    n       = len(dosyalar)
    temp_dir = ISLENMIS / "_lgbm_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    n_batch = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  {n} dosya | {n_jobs} is parcacigi | {n_batch} batch x {BATCH_SIZE}\n")

    toplam_segment = 0
    toplam_hata    = 0

    for batch_idx in range(n_batch):
        out_file = temp_dir / f"batch_{batch_idx:04d}.parquet"

        if out_file.exists():
            mevcut = len(pd.read_parquet(out_file, columns=["sinif"]))
            print(f"  Batch {batch_idx+1:>4}/{n_batch}  [atlandi — {mevcut} segment mevcut]")
            toplam_segment += mevcut
            continue

        bas  = batch_idx * BATCH_SIZE
        bitis = min(bas + BATCH_SIZE, n)
        batch = dosyalar[bas:bitis]

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_dosya_isle)(str(wav_yolu), sinif, sr)
            for wav_yolu, sinif in batch
        )

        batch_satirlar = []
        for satirlar, hata in results:
            batch_satirlar.extend(satirlar)
            toplam_hata += hata

        df_batch = pd.DataFrame(batch_satirlar)
        df_batch.to_parquet(out_file, index=False)
        toplam_segment += len(df_batch)

        pct = bitis / n * 100
        print(f"  Batch {batch_idx+1:>4}/{n_batch}  {pct:5.1f}%  "
              f"({toplam_segment} segment, {toplam_hata} hata)")

    # Tum batch'leri birlestir
    print("\n  Parcalar birlestiriliyor...")
    dfs = [pd.read_parquet(f) for f in sorted(temp_dir.glob("batch_*.parquet"))]
    df_final = pd.concat(dfs, ignore_index=True)
    shutil.rmtree(temp_dir)

    print(f"  Tamamlandi: {len(df_final)} segment, {toplam_hata} hata")
    return df_final


def veri_listesi_olustur(dosyalar: list[tuple[Path, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dosya":     str(wav_yolu.relative_to(ROOT)),
            "sinif":     sinif,
            "sinif_idx": SINIF_INDEX[sinif],
        }
        for wav_yolu, sinif in dosyalar
    ])


def main():
    parser = argparse.ArgumentParser(description="BBC ses arsivinden ozellik cikarimi")
    parser.add_argument("--mode", choices=["lgbm", "liste", "hepsi"], default="hepsi")
    parser.add_argument("--sr",   type=int, default=16000)
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Paralel is parcacigi sayisi (-1 = tum cekirdekler)")
    args = parser.parse_args()

    if not HAM.exists():
        print(f"HATA: {HAM} bulunamadi. Once 00_organize_data.py calistir.")
        sys.exit(1)

    ISLENMIS.mkdir(parents=True, exist_ok=True)

    dosyalar = wav_listesi_al()
    n_jobs   = args.jobs if args.jobs > 0 else os.cpu_count() or 4
    print(f"\n{len(dosyalar)} WAV dosyasi ({len(SINIFLAR)} sinif) | {n_jobs} CPU\n")

    if args.mode in ("liste", "hepsi"):
        print("-- Dosya listesi olusturuluyor...")
        df_liste = veri_listesi_olustur(dosyalar)
        cikti = ISLENMIS / "veri_listesi.parquet"
        df_liste.to_parquet(cikti, index=False)
        print(f"   Kaydedildi: {cikti.relative_to(ROOT)}  ({len(df_liste)} satir)\n")

    if args.mode in ("lgbm", "hepsi"):
        print("-- LGBM ozellikleri cikariliyor...")
        df_lgbm = lgbm_ozelliklerini_cikar(dosyalar, sr=args.sr, n_jobs=n_jobs)
        cikti = ISLENMIS / "lgbm_ozellikler.parquet"
        df_lgbm.to_parquet(cikti, index=False)
        print(f"\n   Kaydedildi: {cikti.relative_to(ROOT)}"
              f"  ({len(df_lgbm)} satir, {df_lgbm.shape[1]} sutun)\n")

        print("  Sinif bazli segment sayisi:")
        for sinif, sayac in df_lgbm.groupby("sinif").size().sort_values(ascending=False).items():
            print(f"    {sinif:<25} {sayac:>6}")


if __name__ == "__main__":
    main()
