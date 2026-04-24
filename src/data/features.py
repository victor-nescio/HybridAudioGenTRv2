"""
Ses özellik çıkarım modülü — BBC 26-sınıf hybrid sınıflandırma/üretim sistemi.

Özellik grupları ve toplam boyutlar:
  1. MFCC genişletilmiş    : 40×{mean,std,skew,kurt} + Δ×{m,s} + ΔΔ×{m,s} = 320-dim
  2. Mel filterbank enerjisi: 40 band × {mean,std}                           =  80-dim
  3. Spektral doku          : centroid,rolloff,bw,ZCR,RMS,contrast(7),
                              flatness,flux × {mean,std}                     =  28-dim
  4. Tonal/harmonik         : Chroma STFT(12) + Tonnetz(6) × {mean,std}      =  36-dim
  5. Zamansal dinamik       : onset {mean,std} + tempo                       =   3-dim
  ─────────────────────────────────────────────────────────────────────────────────────
  LGBM düz vektör (duz_ozellik_cikar)                                        = 467-dim

  CNN / AST için log-mel spectrogram (log_mel_cikar): (N_MELS=128, T) 2D dizi
"""

import warnings
import numpy as np
import librosa
from scipy.stats import kurtosis, skew

# ── Sabitler ──────────────────────────────────────────────────────────────────
N_MFCC     = 40    # MFCC katsayı sayısı
N_MELS     = 128   # CNN/AST log-mel yüksekliği
MEL_BANDS  = 40    # LGBM mel filterbank band sayısı
N_CHROMA   = 12    # Kroma bin sayısı
N_FFT      = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SEGMENT_SN = 5.0


# ── 1. MFCC genişletilmiş (320-dim) ──────────────────────────────────────────
def mfcc_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    40 MFCC × {mean, std, skewness, kurtosis}    → 160-dim
    Δ MFCC  × {mean, std}                        →  80-dim
    ΔΔ MFCC × {mean, std}                        →  80-dim
    Toplam: 320-dim

    Üst-derece istatistikler (skew, kurt) dağılım asimetrisini ve kuyruk
    ağırlığını yakalar; Piczak (2015) ve Salamon & Bello (2015)'de kullanılmıştır.
    """
    mfcc   = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            skew(mfcc, axis=1),
            kurtosis(mfcc, axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
        ])  # 320-dim


# ── 2. Mel filterbank enerjisi (80-dim) ───────────────────────────────────────
def mel_enerji_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    DCT uygulanmadan önceki 40 mel band log-enerjisi × {mean, std} → 80-dim.

    MFCC'nin dekorelasyon (DCT) adımını atlar; frekans bantları arasındaki
    mutlak enerji dağılımını korur. ESC-50 ve UrbanSound8K benchmark'larında
    MFCC ile birlikte sınıflandırma başarımını artırdığı gösterilmiştir.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=MEL_BANDS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    log_mel = np.log(mel + 1e-9)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])  # 80-dim


# ── 3. Spektral doku özellikleri (28-dim) ────────────────────────────────────
def spektral_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Temel spektral özellikler (10-dim):
      centroid, rolloff, bandwidth, ZCR, RMS × {mean, std}

    Spektral kontrast (14-dim):
      7 frekans bandı × {mean, std} — harmonik/gürültü oranını ölçer;
      ses doku sınıflandırmasında kritik (Hifny & Owen, 2019).

    Spektral düzlük (2-dim):
      Tonal ses ↔ gürültü ayrımı için; yüksek değer = gürültü benzeri.

    Spektral akış (2-dim):
      Ardışık spektrum kareleri arasındaki L2 farkı; geçici olayları (patlama,
      kapı çarpma) yakalar.

    Toplam: 28-dim
    """
    centroid  = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff   = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr       = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms       = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    contrast  = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6)
    flatness  = librosa.feature.spectral_flatness(
        y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft_mag  = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    flux_vals = np.sqrt(np.sum(np.diff(stft_mag, axis=1) ** 2, axis=0))
    flux      = flux_vals.reshape(1, -1)

    def ms(feat):
        return np.array([float(feat.mean()), float(feat.std())])

    return np.concatenate([
        ms(centroid), ms(rolloff), ms(bandwidth), ms(zcr), ms(rms),  # 10
        contrast.mean(axis=1), contrast.std(axis=1),                  # 14
        ms(flatness), ms(flux),                                        #  4
    ])  # 28-dim


# ── 4. Tonal ve harmonik özellikler (36-dim) ─────────────────────────────────
def tonal_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Chroma STFT (24-dim):
      12 yarım-ton bin × {mean, std} — perdeli sesler (zil, motor tonu,
      müzik) ile perdeli olmayan sesler (gürültü, hava) arasında ayrım sağlar.

    Tonnetz (12-dim):
      6 tonal merkez koordinatı × {mean, std} — beşli/majör-minör üçlü
      ilişkilerini temsil eden harmonik uzay; Harte et al. (2006).

    Toplam: 36-dim
    """
    # Chroma için ön koşul: harmonik bileşeni izole et
    y_harm = librosa.effects.harmonic(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chroma  = librosa.feature.chroma_stft(
            y=y_harm, sr=sr, n_chroma=N_CHROMA,
            n_fft=N_FFT, hop_length=HOP_LENGTH)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)

    return np.concatenate([
        chroma.mean(axis=1),  chroma.std(axis=1),    # 24
        tonnetz.mean(axis=1), tonnetz.std(axis=1),   # 12
    ])  # 36-dim


# ── 5. Zamansal ve dinamik özellikler (3-dim) ────────────────────────────────
def zamansal_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Onset gücü × {mean, std}:
      Spektral enerjideki ani artışlar — ayak sesleri, kapı çarpma,
      patlama gibi geçici olayların yoğunluğunu ve düzenliliğini ölçer.

    Tempo (BPM):
      Ritmik tekrarı olan seslerde (fabrika, ayak, tren) anlamlı;
      ritmik olmayan seslerde 0'a yakın, ayırt edici özellik görevi görür.

    Toplam: 3-dim
    """
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                   hop_length=HOP_LENGTH)

    return np.array([
        float(onset_env.mean()),
        float(onset_env.std()),
        float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo),
    ])  # 3-dim


# ── CNN / AST için 2D log-mel spectrogram ─────────────────────────────────────
def log_mel_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """(N_MELS=128, T) — CNN ve Audio Spectrogram Transformer girişi."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max)


# ── Segmentasyon ──────────────────────────────────────────────────────────────
def segmentle(y: np.ndarray, sr: int,
              segment_sn: float = SEGMENT_SN) -> list[np.ndarray]:
    """
    Sesi sabit uzunlukta segmentlere böler.
    Yarıdan kısa kalan son segment atılır.
    """
    uzunluk = int(segment_sn * sr)
    return [
        y[bas: bas + uzunluk]
        for bas in range(0, len(y), uzunluk)
        if len(y[bas: bas + uzunluk]) >= uzunluk // 2
    ]


# ── LGBM için 467-dim düz vektör ─────────────────────────────────────────────
def duz_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Tüm özellik gruplarını birleştirir → 467-dim düz vektör.

      Grup 1 — MFCC genişletilmiş : 320-dim
      Grup 2 — Mel filterbank     :  80-dim
      Grup 3 — Spektral doku      :  28-dim
      Grup 4 — Tonal/harmonik     :  36-dim
      Grup 5 — Zamansal dinamik   :   3-dim
      ─────────────────────────────────────
      Toplam                      : 467-dim
    """
    if len(y) == 0:
        return np.zeros(467)

    return np.concatenate([
        mfcc_ozellik_cikar(y, sr),       # 320
        mel_enerji_cikar(y, sr),         #  80
        spektral_ozellik_cikar(y, sr),   #  28
        tonal_ozellik_cikar(y, sr),      #  36
        zamansal_ozellik_cikar(y, sr),   #   3
    ])  # 467-dim
