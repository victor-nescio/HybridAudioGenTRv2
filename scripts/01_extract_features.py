"""
data/raw/ altındaki WAV dosyalarından özellik çıkarır.

Çıktılar:
  data/processed/lgbm_ozellikler.parquet  -- LGBM için 467-dim vektörler (segment başına 1 satır)
  data/processed/veri_listesi.parquet     -- CNN/AST için dosya listesi (yol, sinif, sinif_idx)

Kullanım:
  python scripts/01_extract_features.py              # her ikisi de
  python scripts/01_extract_features.py --mode lgbm  # sadece LGBM özellikleri
  python scripts/01_extract_features.py --mode liste # sadece dosya listesi
  python scripts/01_extract_features.py --sr 22050   # farklı sample rate
"""
import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.labels import SINIFLAR
from src.data.preprocessing import onisle
from src.data.features import duz_ozellik_cikar, segmentle, SEGMENT_SN

ISLENMIS = ROOT / "data" / "processed"
HAM      = ROOT / "data" / "raw"

SINIF_INDEX = {sinif: i for i, sinif in enumerate(sorted(SINIFLAR.keys()))}


def wav_listesi_al() -> list[tuple[Path, str]]:
    """data/raw/<sinif>/ altındaki tüm WAV dosyalarını döndürür."""
    dosyalar = []
    for sinif in sorted(SINIFLAR.keys()):
        sinif_yolu = HAM / sinif
        if not sinif_yolu.exists():
            continue
        for wav in sinif_yolu.rglob("*.wav"):
            dosyalar.append((wav, sinif))
    return dosyalar


def lgbm_ozelliklerini_cikar(dosyalar: list[tuple[Path, str]], sr: int) -> pd.DataFrame:
    satirlar = []
    hata_sayisi = 0
    n = len(dosyalar)

    for idx, (wav_yolu, sinif) in enumerate(dosyalar, 1):
        if idx % 500 == 0 or idx == n:
            print(f"  [{idx:>5}/{n}] {wav_yolu.name[:50]}")

        try:
            y, sr_out = onisle(str(wav_yolu), sr=sr)
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

        except Exception:
            hata_sayisi += 1
            print(f"  HATA: {wav_yolu.relative_to(ROOT)}")
            if hata_sayisi <= 5:
                traceback.print_exc()

    print(f"\n  Tamamlandi: {len(satirlar)} segment, {hata_sayisi} hata")
    return pd.DataFrame(satirlar)


def veri_listesi_olustur(dosyalar: list[tuple[Path, str]]) -> pd.DataFrame:
    """Her WAV için dosya yolu + etiket. CNN/AST DataLoader bu listeyi okur."""
    return pd.DataFrame([
        {
            "dosya":     str(wav_yolu.relative_to(ROOT)),
            "sinif":     sinif,
            "sinif_idx": SINIF_INDEX[sinif],
        }
        for wav_yolu, sinif in dosyalar
    ])


def main():
    parser = argparse.ArgumentParser(description="BBC ses arşivinden özellik çıkarımı")
    parser.add_argument("--mode", choices=["lgbm", "liste", "hepsi"], default="hepsi",
                        help="lgbm: 467-dim vektörler | liste: dosya listesi | hepsi (varsayilan)")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Hedef sample rate (varsayilan: 16000)")
    args = parser.parse_args()

    if not HAM.exists():
        print(f"HATA: {HAM} bulunamadi. Once 00_organize_data.py calistir.")
        sys.exit(1)

    ISLENMIS.mkdir(parents=True, exist_ok=True)

    dosyalar = wav_listesi_al()
    print(f"\n{len(dosyalar)} WAV dosyasi bulundu ({len(SINIFLAR)} sinif)\n")

    if args.mode in ("liste", "hepsi"):
        print("-- Dosya listesi olusturuluyor...")
        df_liste = veri_listesi_olustur(dosyalar)
        cikti = ISLENMIS / "veri_listesi.parquet"
        df_liste.to_parquet(cikti, index=False)
        print(f"   Kaydedildi: {cikti.relative_to(ROOT)}  ({len(df_liste)} satir)\n")

    if args.mode in ("lgbm", "hepsi"):
        print("-- LGBM ozellikleri cikariliyor (bu sure alabilir)...")
        df_lgbm = lgbm_ozelliklerini_cikar(dosyalar, sr=args.sr)
        cikti = ISLENMIS / "lgbm_ozellikler.parquet"
        df_lgbm.to_parquet(cikti, index=False)
        print(f"   Kaydedildi: {cikti.relative_to(ROOT)}  ({len(df_lgbm)} satir, {df_lgbm.shape[1]} sutun)\n")

        # Sinif bazli segment dagilimi
        print("  Sinif bazli segment sayisi:")
        for sinif, sayac in df_lgbm.groupby("sinif").size().sort_values(ascending=False).items():
            print(f"    {sinif:<25} {sayac:>6}")


if __name__ == "__main__":
    main()
