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
SONUCLAR_YOL  = ROOT / "results" / "v1" / "fad_sonuclar.json"

VARSAYILAN_SINIFLAR = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]


def fad_hesapla(gercek_dir: Path, sentetik_dir: Path,
                embedding: str = "vggish") -> float:
    """fadtk ile tek sinif icin FAD hesapla."""
    try:
        from fadtk import FrechetAudioDistance
        from fadtk.utils import calculate_embd_statistics_online
        from fadtk.fad import calc_frechet_distance
    except ImportError:
        print("  HATA: fadtk yuklu degil. Lutfen: pip install fadtk")
        return -1.0

    from fadtk.model_loader import VGGishModel, CLAPLaionModel

    if embedding == "clap":
        ml = CLAPLaionModel("music_speech_audioset_epoch_15_esc_89.98")
    else:
        ml = VGGishModel()

    import tempfile, numpy as np

    fad = FrechetAudioDistance(ml=ml, audio_load_worker=1, load_model=True)

    def _istatistik_hesapla(audio_dir: Path):
        """Dizindeki WAV'lardan embedding cikar, mu/cov don."""
        wav_dosyalar = list(audio_dir.glob("*.wav"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            npy_dosyalar = []
            for wav in wav_dosyalar:
                wav_data = fad.load_audio(wav)
                embd = fad.ml.get_embedding(wav_data)   # (n_frames, dim)
                npy = tmp_path / (wav.stem + ".npy")
                np.save(npy, embd)
                npy_dosyalar.append(npy)
            return calculate_embd_statistics_online(npy_dosyalar)

    mu_g,  cov_g  = _istatistik_hesapla(gercek_dir)
    mu_s,  cov_s  = _istatistik_hesapla(sentetik_dir)
    return float(calc_frechet_distance(mu_g, cov_g, mu_s, cov_s))


def sinif_dosya_sec(sinif_dir: Path, maks: int = 500) -> list[Path]:
    """Sinif dizininden en fazla maks WAV sec — DETERMINISTIK.
    Onceki surum seed'siz random.shuffle kullaniyordu; bu, FAD'i
    kosumdan kosuma degisken (tekrar uretilemez) yapiyordu. Artik
    sabit seed (42) + sirali liste ile ayni alt kume her zaman secilir."""
    import random
    dosyalar = sorted(sinif_dir.glob("**/*.wav"))   # deterministik siralama
    if len(dosyalar) > maks:
        rng = random.Random(42)                      # sabit seed
        rng.shuffle(dosyalar)
        dosyalar = sorted(dosyalar[:maks])           # secimi de sirala
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
    # maks=300: 4 hedef sinifin tum gercek klipleri kullanilir (en fazla 243),
    # boylece secim deterministik ve tam (alt-orneklem gurultusu yok).
    dosyalar = sinif_dosya_sec(gercek_kaynak, maks=300)
    for i, d in enumerate(dosyalar):
        # Kisa sayisal ad: Windows MAX_PATH sorununu onler
        guvenli = f"{i:04d}.wav"
        shutil.copy2(d, tmp_dir / guvenli)
    return tmp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--siniflar",    nargs="+", default=None,
                        help="Olcum yapilacak siniflar (varsayilan: zayif siniflar)")
    parser.add_argument("--embedding",   choices=["vggish", "clap"], default="vggish",
                        help="FAD embedding modeli (varsayilan: vggish)")
    parser.add_argument("--maks_gercek", type=int, default=200,
                        help="Her sinif icin kullanilacak gercek ses sayisi")
    parser.add_argument("--sentetik_dir", type=str, default=None,
                        help="Ozel sentetik dizin (Tur2/Tur3 icin, orn: data/synthetic_tur2)")
    parser.add_argument("--tur",          type=str, default=None,
                        help="Sonuc anahtari etiketi (orn: tur2, tur3)")
    args = parser.parse_args()

    global SENTETIK_DIR, SONUCLAR_YOL
    if args.sentetik_dir:
        SENTETIK_DIR = ROOT / args.sentetik_dir
    if args.tur:
        SONUCLAR_YOL = ROOT / "results" / "v1" / f"fad_sonuclar_{args.tur}.json"

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

            # Per-sinif hataya dayanikli: bir sinif coozse digerleri + JSON korunur
            try:
                fad = fad_hesapla(gercek_tmp, sentetik_dir, args.embedding)
            except Exception as ex:
                print(f"  HATA ({sinif}): {ex} -> fad=-1, devam")
                fad = -1.0
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
