"""
BBC ses arsivi icin PyTorch Dataset / DataLoader.

Her ornek: on-the-fly WAV yukle -> 5 sn segment -> log-mel spectrogram -> tensor

Desteklenen modlar:
  'cnn'  -> (1, 128, T_max)  kanal-birinci 2D log-mel  (CNN / ResNet)
  'lstm' -> (T_max, 128)     zaman-birinci mel dizi     (BiLSTM / GRU)
  'ast'  -> (1, 128, T_max)  AudioSet normalize edilmis (AST / ViT)

Kullanim:
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.data.dataset import BBCSegmentDataset, dataloader_olustur, veri_bol

    df_train, df_val, df_test = veri_bol(ROOT / "data/processed/veri_listesi.parquet", ROOT)

    dl_train = dataloader_olustur(df_train, ROOT, mod='cnn', batch_size=32)
    dl_val   = dataloader_olustur(df_val,   ROOT, mod='cnn', batch_size=64, shuffle=False)

    for mel_batch, etiket_batch in dl_train:
        # mel_batch : (B, 1, 128, T_max)  float32
        # etiket_batch: (B,)              int64
        pass
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.features import HOP_LENGTH, N_FFT, N_MELS, SEGMENT_SN, WIN_LENGTH

warnings.filterwarnings("ignore")

def _mel_hesapla(y: np.ndarray, sr: int) -> np.ndarray:
    """torchaudio ile log-mel spectrogram hesapla (librosa'dan ~10x hizli)."""
    transform = T.MelSpectrogram(
        sample_rate=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        power=2.0,
    )
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)
    y_t  = torch.from_numpy(y).unsqueeze(0)   # (1, T)
    mel  = transform(y_t)                      # (1, N_MELS, frames)
    mel  = amp_to_db(mel)
    return mel.squeeze(0).numpy()              # (N_MELS, frames)

# ── Sabitler ──────────────────────────────────────────────────────────────────
SEED    = 42
SR      = 16_000
T_MAX   = 313   # 5 sn x 16000 Hz / hop=256 ≈ 313 frame

# AST AudioSet pretrained normalize parametreleri (Gong et al. 2021)
AST_MEAN = -4.2677393
AST_STD  = 4.5689974


# ── Dataset ───────────────────────────────────────────────────────────────────
class BBCSegmentDataset(Dataset):
    """
    veri_listesi.parquet'teki WAV dosyalarindan on-the-fly segment uretir.

    Args:
        df       : ['dosya', 'sinif', 'sinif_idx'] sutunlarini iceren DataFrame
        root     : proje kok dizini (df['dosya'] buraya gore relatiftir)
        mod      : 'cnn' | 'lstm' | 'ast'
        sr       : hedef sample rate
        segment_sn: segment suresi (saniye)
        T_max    : cikti zaman boyutu (pad veya truncate)
    """

    def __init__(self, df: pd.DataFrame, root,
                 mod: str = "cnn", sr: int = SR,
                 segment_sn: float = SEGMENT_SN, T_max: int = T_MAX,
                 preload: bool = True, spec_augment: bool = False):
        self.root       = Path(root)
        self.mod        = mod
        self.sr         = sr
        self.segment_sn = segment_sn
        self.T_max      = T_max
        self.cache_dir  = self.root / "data" / "mel_cache"
        # SpecAugment (yalnizca egitimde aciktir): normalize sonrasi mel uzerinde
        # zaman/frekans maskeleme. Geleneksel augmentasyon temeli (M2) icin.
        self.spec_augment = spec_augment
        self._sa_rng = np.random.default_rng(0)

        self.indeks = self._indeks_olustur(df)

        # Tum mel dosyalarini RAM'e yukle (disk I/O tamamen ortadan kalkar)
        self._mel_ram: dict | None = None
        if preload:
            self._mel_ram = self._ram_yukle()

    def _cache_yolu(self, wav_yolu: str) -> Path | None:
        try:
            rel = Path(wav_yolu).relative_to(self.root)
            return self.cache_dir / rel.with_suffix(".npy")
        except ValueError:
            return None

    def _ram_yukle(self) -> dict:
        """Benzersiz WAV dosyalarinin mel cache'lerini RAM'e yukler."""
        benzersiz = {wav for wav, _, _ in self.indeks}
        mel_ram   = {}
        for wav in benzersiz:
            cache = self._cache_yolu(wav)
            if cache and cache.exists():
                mel_ram[wav] = np.load(str(cache))   # float16, RAM'de
        return mel_ram

    # ── Init ──────────────────────────────────────────────────────────────────
    def _indeks_olustur(self, df: pd.DataFrame) -> list[tuple]:
        indeks = []
        has_n_segs = "n_segs" in df.columns
        for _, satir in df.iterrows():
            wav_yolu  = str(self.root / satir["dosya"])
            sinif_idx = int(satir["sinif_idx"])
            if has_n_segs:
                n_seg = max(1, int(satir["n_segs"]))
            else:
                try:
                    bilgi = sf.info(wav_yolu)
                    n_seg = max(1, int(bilgi.duration / self.segment_sn))
                except Exception:
                    n_seg = 1
            for s in range(n_seg):
                indeks.append((wav_yolu, sinif_idx, s))
        return indeks

    # ── Dataset arayuzu ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indeks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav_yolu, sinif_idx, seg_no = self.indeks[idx]

        try:
            # 1. RAM cache (en hizli) — sadece ilgili dilimi float32'ye cevir
            if self._mel_ram is not None and wav_yolu in self._mel_ram:
                baslangic = seg_no * self.T_max
                mel = self._mel_ram[wav_yolu][:, baslangic: baslangic + self.T_max].astype(np.float32)
            # 2. Disk cache
            elif (cache_yolu := self._cache_yolu(wav_yolu)) and cache_yolu.exists():
                mel_tam   = np.load(str(cache_yolu))
                baslangic = seg_no * self.T_max
                mel       = mel_tam[:, baslangic: baslangic + self.T_max].astype(np.float32)
            else:
                # Cache yoksa WAV'dan hesapla
                seg_ornekler = int(self.segment_sn * self.sr)
                y, sr_dosya  = sf.read(wav_yolu,
                                       start=seg_no * seg_ornekler,
                                       stop=(seg_no + 1) * seg_ornekler,
                                       dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr_dosya != self.sr:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr_dosya, target_sr=self.sr)
                y = y - y.mean()
                tepe = np.abs(y).max()
                if tepe > 0:
                    y *= 0.99 / tepe
                if len(y) < seg_ornekler:
                    y = np.pad(y, (0, seg_ornekler - len(y)))
                mel = _mel_hesapla(y, self.sr)
        except Exception:
            mel = np.full((N_MELS, self.T_max), fill_value=-80.0, dtype=np.float32)

        mel = self._isle(mel)
        return (
            torch.tensor(mel, dtype=torch.float32),
            torch.tensor(sinif_idx, dtype=torch.long),
        )

    # ── Log-mel on isleme ─────────────────────────────────────────────────────
    def _isle(self, mel: np.ndarray) -> np.ndarray:
        """Pad/truncate + normalize + moda gore yeniden boyutlandir."""
        # Zaman eksenini T_max'a esitle
        T = mel.shape[1]
        if T < self.T_max:
            mel = np.pad(mel, ((0, 0), (0, self.T_max - T)),
                         mode="constant", constant_values=-80.0)
        else:
            mel = mel[:, :self.T_max]

        if self.mod == "ast":
            # AST: AudioSet istatistikleriyle global normalize
            mel = (mel - AST_MEAN) / (AST_STD * 2)
            return mel[np.newaxis]          # (1, 128, T_max)

        # CNN / LSTM icin: ornek bazli normalize (sifir ortalama, birim varyas)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # SpecAugment: normalize sonrasi (sifir = ortalama) maskeleme
        if self.spec_augment:
            mel = self._spec_augment(mel)

        if self.mod == "lstm":
            return mel.T                    # (T_max, 128) zaman-birinci

        return mel[np.newaxis]              # (1, 128, T_max) cnn

    def _spec_augment(self, mel: np.ndarray,
                      n_freq: int = 2, max_freq: int = 15,
                      n_time: int = 2, max_time: int = 20) -> np.ndarray:
        """SpecAugment (Park ve ark. 2019): 2 frekans + 2 zaman maskesi.
        Normalize edilmis mel uzerinde 0 (=ortalama) ile maskeler."""
        mel = mel.copy()
        F, T = mel.shape
        rng = self._sa_rng
        for _ in range(n_freq):
            f = int(rng.integers(0, max_freq + 1))
            if f > 0 and F - f > 0:
                f0 = int(rng.integers(0, F - f))
                mel[f0:f0 + f, :] = 0.0
        for _ in range(n_time):
            t = int(rng.integers(0, max_time + 1))
            if t > 0 and T - t > 0:
                t0 = int(rng.integers(0, T - t))
                mel[:, t0:t0 + t] = 0.0
        return mel


# ── Yardimci: DataLoader fabrikasi ────────────────────────────────────────────
def dataloader_olustur(
    df: pd.DataFrame,
    root,
    mod: str = "cnn",
    sr: int = SR,
    T_max: int = T_MAX,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    preload: bool = False,
    spec_augment: bool = False,
) -> DataLoader:
    """BBCSegmentDataset + DataLoader'u tek satirda olusturur."""
    dataset = BBCSegmentDataset(
        df, root, mod=mod, sr=sr, T_max=T_max, preload=preload,
        spec_augment=spec_augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=(shuffle),
    )


# ── Yardimci: Dosya duzeyinde stratified split ────────────────────────────────
def veri_bol(
    parquet_yolu,
    root,
    test_oran: float = 0.15,
    val_oran: float  = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    veri_listesi.parquet'i dosya duzeyinde stratified olarak boler.

    Returns:
        (df_train, df_val, df_test)  — her biri Dataset'e dogrudan gecirilebilir
    """
    df = pd.read_parquet(Path(parquet_yolu))

    dosya_sinif = (df.groupby("dosya")["sinif_idx"]
                   .first().reset_index())
    dosyalar = dosya_sinif["dosya"].values
    siniflar = dosya_sinif["sinif_idx"].values

    d_trainval, d_test, s_trainval, _ = train_test_split(
        dosyalar, siniflar,
        test_size=test_oran, stratify=siniflar, random_state=seed)

    val_gercek = val_oran / (1 - test_oran)
    d_train, d_val = train_test_split(
        d_trainval, test_size=val_gercek,
        stratify=s_trainval, random_state=seed)

    def filtrele(dosya_arr):
        return df[df["dosya"].isin(dosya_arr)].reset_index(drop=True)

    return filtrele(d_train), filtrele(d_val), filtrele(d_test)
