"""
Tum BBC dosyalarinin log-mel spektrogramlarini onceden hesaplar ve
float16 .npy dosyalari olarak cache'ler.

Cikti: data/mel_cache/<sinif>/<dosyaadi>.npy
       Boyut: (N_MELS, toplam_frame)  float16

Faydasi: Dataset.__getitem__ artik WAV okumaz, sadece .npy dilimleri
         yukler -> epoch suresi ~10-20x hizlanir.

Kullanim:
  python scripts/00b_precompute_mel.py
  python scripts/00b_precompute_mel.py --workers 8   # kac CPU
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.transforms as T
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.features import HOP_LENGTH, N_FFT, N_MELS, WIN_LENGTH

CACHE_DIR = ROOT / "data" / "mel_cache"
SR = 16_000

_transform = T.MelSpectrogram(
    sample_rate=SR, n_mels=N_MELS,
    n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, power=2.0,
)
_amp_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)


def mel_hesapla_ve_kaydet(row, root: Path, cache_dir: Path) -> str | None:
    dosya_yolu = root / row["dosya"]
    rel        = Path(row["dosya"])
    cache_yolu = cache_dir / rel.with_suffix(".npy")

    if cache_yolu.exists():
        return None   # zaten hesaplanmis

    try:
        y, sr = sf.read(str(dosya_yolu), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)

        y = y - y.mean()
        tepe = np.abs(y).max()
        if tepe > 0:
            y *= 0.99 / tepe

        y_t = torch.from_numpy(y).unsqueeze(0)
        mel = _transform(y_t)
        mel = _amp_to_db(mel)
        mel_np = mel.squeeze(0).numpy().astype(np.float16)  # (N_MELS, T) float16

        cache_yolu.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_yolu), mel_np)
        return str(cache_yolu)
    except Exception as e:
        return f"HATA:{dosya_yolu}:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    df = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")
    print(f"{len(df):,} dosya icin mel cache hesaplanacak")
    print(f"Cikti: {CACHE_DIR}  |  Workers: {args.workers}")

    # Kac tanesi zaten var?
    mevcut = sum(1 for r in df.itertuples()
                 if (CACHE_DIR / Path(r.dosya).with_suffix(".npy")).exists())
    print(f"Mevcut: {mevcut:,}  Eksik: {len(df)-mevcut:,}\n")

    if mevcut == len(df):
        print("Hepsi zaten hesaplanmis.")
        return

    rows = [row for _, row in df.iterrows()]
    sonuclar = Parallel(n_jobs=args.workers, prefer="threads")(
        delayed(mel_hesapla_ve_kaydet)(row, ROOT, CACHE_DIR)
        for row in tqdm(rows, desc="Mel cache", ncols=70)
    )

    hatalar = [s for s in sonuclar if s and s.startswith("HATA")]
    print(f"\nTamamlandi. Hata: {len(hatalar)}")
    if hatalar:
        for h in hatalar[:5]:
            print(" ", h)

    # Boyut raporu
    toplam_mb = sum(
        (CACHE_DIR / Path(r.dosya).with_suffix(".npy")).stat().st_size
        for r in df.itertuples()
        if (CACHE_DIR / Path(r.dosya).with_suffix(".npy")).exists()
    ) / 1e6
    print(f"Toplam cache boyutu: {toplam_mb/1024:.1f} GB")


if __name__ == "__main__":
    main()
