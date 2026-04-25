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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.features import HOP_LENGTH, N_MELS, SEGMENT_SN, log_mel_cikar, segmentle
from src.data.preprocessing import onisle

warnings.filterwarnings("ignore")

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
                 segment_sn: float = SEGMENT_SN, T_max: int = T_MAX):
        self.root       = Path(root)
        self.mod        = mod
        self.sr         = sr
        self.segment_sn = segment_sn
        self.T_max      = T_max

        # Segment indeksi: [(wav_yolu_str, sinif_idx, seg_no), ...]
        # soundfile.info() ile sadece baslik okunarak segment sayisi tahmin edilir
        self.indeks = self._indeks_olustur(df)

    # ── Init ──────────────────────────────────────────────────────────────────
    def _indeks_olustur(self, df: pd.DataFrame) -> list[tuple]:
        indeks = []
        for _, satir in df.iterrows():
            wav_yolu = str(self.root / satir["dosya"])
            sinif_idx = int(satir["sinif_idx"])
            try:
                bilgi  = sf.info(wav_yolu)
                n_seg  = max(1, int(bilgi.duration / self.segment_sn))
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
            y, sr_out = onisle(wav_yolu, sr=self.sr)
            segmentler = segmentle(y, sr_out, self.segment_sn)

            if seg_no < len(segmentler):
                seg = segmentler[seg_no]
            elif segmentler:
                seg = segmentler[-1]      # son segment kullan
            else:
                seg = y

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mel = log_mel_cikar(seg, sr_out)  # (N_MELS, T)
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

        if self.mod == "lstm":
            return mel.T                    # (T_max, 128) zaman-birinci

        return mel[np.newaxis]              # (1, 128, T_max) cnn


# ── Yardimci: DataLoader fabrikasi ────────────────────────────────────────────
def dataloader_olustur(
    df: pd.DataFrame,
    root,
    mod: str = "cnn",
    sr: int = SR,
    T_max: int = T_MAX,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """BBCSegmentDataset + DataLoader'u tek satirda olusturur."""
    dataset = BBCSegmentDataset(
        df, root, mod=mod, sr=sr, T_max=T_max)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=(shuffle),   # train'de son eksik batch'i at
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
