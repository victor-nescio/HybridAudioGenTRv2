"""
BBC ses arşivini 26 Türkçe sınıfa göre data/raw/ altında düzenler.

Kullanım:
  python scripts/00_organize_data.py            # kuru çalışma (sadece rapor)
  python scripts/00_organize_data.py --execute  # dosyaları taşı
"""
import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Proje kökü
ROOT = Path(__file__).resolve().parent.parent
KAYNAK = ROOT / "sounds"
HEDEF  = ROOT / "data" / "raw"

sys.path.insert(0, str(ROOT))
from src.data.labels import siniflandir, turkce_ad, SINIFLAR


def tara():
    """sounds/ dizinini tarar; her girişi (yol, tip) olarak döndürür."""
    girdi_klasorler = []
    girdi_dosyalar  = []

    for girdi in sorted(KAYNAK.iterdir()):
        if girdi.name == "readme_bbc":
            continue
        if girdi.is_dir():
            girdi_klasorler.append(girdi)
        elif girdi.is_file() and girdi.suffix.lower() == ".wav":
            girdi_dosyalar.append(girdi)

    return girdi_klasorler, girdi_dosyalar


def siniflandir_hepsini(klasorler, dosyalar):
    """Her girişi sınıflandırır; eşleşme_haritası ve eslesmeyenler listesi döner."""
    eslesmeler: dict[str, list[Path]] = defaultdict(list)
    eslesmeyenler: list[str] = []

    for klasor in klasorler:
        sinif = siniflandir(klasor.name)
        if sinif:
            eslesmeler[sinif].append(klasor)
        else:
            eslesmeyenler.append(klasor.name)

    for dosya in dosyalar:
        sinif = siniflandir(dosya.stem)
        if sinif:
            eslesmeler[sinif].append(dosya)
        else:
            eslesmeyenler.append(dosya.name)

    return eslesmeler, eslesmeyenler


def rapor_yazdir(eslesmeler, eslesmeyenler):
    print("\n" + "=" * 65)
    print(f"  BBC SES ARSIVI -- VERI ORGANIZASYON PLANI")
    print("=" * 65)

    toplam_klasor  = 0
    toplam_dosya   = 0

    for sinif in sorted(SINIFLAR.keys()):
        girdi_listesi = eslesmeler.get(sinif, [])
        if not girdi_listesi:
            continue

        wav_sayisi = 0
        for girdi in girdi_listesi:
            if girdi.is_dir():
                wav_sayisi += sum(1 for f in girdi.iterdir()
                                  if f.suffix.lower() == ".wav")
            else:
                wav_sayisi += 1

        print(f"\n  [{turkce_ad(sinif)}]  ->  data/raw/{sinif}/")
        for girdi in girdi_listesi:
            tip = "[D]" if girdi.is_dir() else "[F]"
            print(f"    {tip} {girdi.name}")
        print(f"    -- Toplam WAV: {wav_sayisi}")
        toplam_klasor += len([g for g in girdi_listesi if g.is_dir()])
        toplam_dosya  += wav_sayisi

    print("\n" + "-" * 65)
    print(f"  Sınıflandırılan toplam WAV : {toplam_dosya}")
    print(f"  Sınıflandırılan kaynak     : {toplam_klasor} klasör")

    if eslesmeyenler:
        print(f"\n  [!] Eslesmeyenler ({len(eslesmeyenler)} giris) -- tasinmayacak:")
        for isim in eslesmeyenler:
            print(f"    - {isim}")
    else:
        print("\n  [OK] Tum girdiler siniflandirildi.")

    print("=" * 65 + "\n")


def tasi(eslesmeler, kuru_calis: bool):
    if kuru_calis:
        print("  [KURu ÇALIŞMA] Dosyalar taşınmadı. --execute ile çalıştır.\n")
        return

    HEDEF.mkdir(parents=True, exist_ok=True)

    for sinif, girdi_listesi in eslesmeler.items():
        sinif_klasoru = HEDEF / sinif
        sinif_klasoru.mkdir(parents=True, exist_ok=True)

        for girdi in girdi_listesi:
            hedef_yol = sinif_klasoru / girdi.name

            if hedef_yol.exists():
                print(f"  ATLANDI (zaten var): {hedef_yol.relative_to(ROOT)}")
                continue

            shutil.move(str(girdi), str(hedef_yol))
            print(f"  TASINDI: {girdi.name}  ->  data/raw/{sinif}/")

    print("\n  Tasima tamamlandi.\n")


def main():
    parser = argparse.ArgumentParser(description="BBC ses arşivini sınıflara göre düzenler.")
    parser.add_argument("--execute", action="store_true",
                        help="Dosyaları gerçekten taşı (varsayılan: sadece rapor)")
    args = parser.parse_args()

    if not KAYNAK.exists():
        print(f"HATA: Kaynak klasör bulunamadı: {KAYNAK}")
        sys.exit(1)

    klasorler, dosyalar = tara()
    eslesmeler, eslesmeyenler = siniflandir_hepsini(klasorler, dosyalar)
    rapor_yazdir(eslesmeler, eslesmeyenler)
    tasi(eslesmeler, kuru_calis=not args.execute)


if __name__ == "__main__":
    main()
