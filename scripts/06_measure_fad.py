"""
FAD (Frechet Audio Distance) olcumu — sentetik vs gercek sesler.

VGGish ve CLAP embedding'leriyle FAD hesaplar.
Gereksinim: fadtk  (pip install fadtk)

Kullanim:
    python scripts/06_measure_fad.py
    python scripts/06_measure_fad.py --siniflar insaat_sesleri dini_mekan
    python scripts/06_measure_fad.py --embedding clap
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_DIR       = ROOT / "data" / "raw"
SENTETIK_DIR  = ROOT / "data" / "synthetic"
SONUCLAR_YOL  = ROOT / "results" / "fad_sonuclar.json"

VARSAYILAN_SINIFLAR = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]


def fad_hesapla(gercek_dir: Path, sentetik_dir: Path,
                embedding: str = "vggish") -> float:
    """fadtk ile tek sinif icin FAD hesapla."""
    try:
        from fadtk import FrechetAudioDistance
    except ImportError:
        print("  HATA: fadtk yuklu degil. Lutfen: pip install fadtk")
        return -1.0

    fad = FrechetAudioDistance(audio_load_worker=4)

    if embedding == "clap":
        fad.model = "clap"

    score = fad.score(str(gercek_dir), str(sentetik_dir))
    return float(score)


def sinif_dosya_sec(sinif_dir: Path, maks: int = 500) -> list[Path]:
    """Sinif dizininden rastgele en fazla maks WAV sec."""
    import random
    dosyalar = list(sinif_dir.glob("*.wav"))
    if len(dosyalar) > maks:
        random.shuffle(dosyalar)
        dosyalar = dosyalar[:maks]
    return dosyalar


def gecici_gercek_dir_olustur(sinif: str, tmp_root: Path) -> Path:
    """
    Buyuk raw dizin yerine kucuk bir tmp alt kumesi olustur.
    FAD hesaplama suresi kisalir.
    """
    import shutil
    gercek_kaynak = RAW_DIR / sinif
    tmp_dir = tmp_root / f"gercek_{sinif}"
    if tmp_dir.exists():
        return tmp_dir
    tmp_dir.mkdir(parents=True)
    dosyalar = sinif_dosya_sec(gercek_kaynak, maks=200)
    for d in dosyalar:
        shutil.copy2(d, tmp_dir / d.name)
    return tmp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siniflar",  nargs="+", default=None,
                        help="Olcum yapilacak siniflar (varsayilan: zayif siniflar)")
    parser.add_argument("--embedding", choices=["vggish", "clap"], default="vggish",
                        help="FAD embedding modeli (varsayilan: vggish)")
    parser.add_argument("--maks_gercek", type=int, default=200,
                        help="Her sinif icin kullanilacak gercek ses sayisi")
    args = parser.parse_args()

    hedef_siniflar = args.siniflar or VARSAYILAN_SINIFLAR

    # Sentetik dizin kontrolu
    eksik = [s for s in hedef_siniflar
             if not (SENTETIK_DIR / s).exists() or
                len(list((SENTETIK_DIR / s).glob("*.wav"))) == 0]
    if eksik:
        print(f"UYARI: Su siniflar icin sentetik ses yok: {eksik}")
        print("Once scripts/05_generate_audio.py calistirin.")
        hedef_siniflar = [s for s in hedef_siniflar if s not in eksik]

    if not hedef_siniflar:
        print("Olcum yapilacak sinif yok, cikiliyor.")
        sys.exit(1)

    print("=" * 60)
    print(f"  FAD Olcumu — embedding: {args.embedding}")
    print(f"  Siniflar : {hedef_siniflar}")
    print("=" * 60)

    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="fad_"))

    sonuclar = {}
    try:
        for sinif in hedef_siniflar:
            sentetik_dir = SENTETIK_DIR / sinif
            n_sentetik = len(list(sentetik_dir.glob("*.wav")))
            print(f"\n[{sinif}]  sentetik={n_sentetik}")

            gercek_tmp = gecici_gercek_dir_olustur(sinif, tmp_root)
            n_gercek = len(list(gercek_tmp.glob("*.wav")))
            print(f"  Gercek orneklem: {n_gercek}")

            fad = fad_hesapla(gercek_tmp, sentetik_dir, args.embedding)
            sonuclar[sinif] = {"fad": fad, "embedding": args.embedding,
                               "n_gercek": n_gercek, "n_sentetik": n_sentetik}
            print(f"  FAD ({args.embedding}) = {fad:.4f}")
    finally:
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Mevcut sonuclari yukle ve birles
    if SONUCLAR_YOL.exists():
        with open(SONUCLAR_YOL, encoding="utf-8") as f:
            mevcut = json.load(f)
    else:
        mevcut = {}

    mevcut.update(sonuclar)
    SONUCLAR_YOL.parent.mkdir(exist_ok=True)
    with open(SONUCLAR_YOL, "w", encoding="utf-8") as f:
        json.dump(mevcut, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  Ozet FAD Sonuclari:")
    for sinif, v in sonuclar.items():
        print(f"    {sinif:<30} FAD={v['fad']:.4f}")
    print(f"\nSonuclar kaydedildi: {SONUCLAR_YOL}")


if __name__ == "__main__":
    main()
