import numpy as np
import librosa

N_MFCC = 40
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
SEGMENT_SN = 5.0


def mfcc_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    40 MFCC + delta + delta-delta, her biri mean+std → 240 boyutlu vektör.
    Literatürde zamansal dinamiği yakalamak için delta ve delta2 standarttır.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.concatenate([
        mfcc.mean(axis=1),   mfcc.std(axis=1),
        delta.mean(axis=1),  delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ])  # 240-dim


def spektral_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Spectral centroid, rolloff, bandwidth, ZCR, RMS — her biri mean+std → 10 boyut.
    """
    centroid   = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff    = librosa.feature.spectral_rolloff(y=y,  sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    bandwidth  = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    zcr        = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    rms        = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)

    return np.array([
        float(centroid.mean()),  float(centroid.std()),
        float(rolloff.mean()),   float(rolloff.std()),
        float(bandwidth.mean()), float(bandwidth.std()),
        float(zcr.mean()),       float(zcr.std()),
        float(rms.mean()),       float(rms.std()),
    ])  # 10-dim


def log_mel_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """(N_MELS, T) boyutunda log-mel spectrogram — CNN / AST pipeline için."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    return librosa.power_to_db(mel, ref=np.max)


def segmentle(y: np.ndarray, sr: int, segment_sn: float = SEGMENT_SN) -> list[np.ndarray]:
    """
    Uzun dosyaları sabit uzunlukta segmentlere böler.
    Yarıdan kısa olan son segment atılır.
    """
    segment_uzunluk = int(segment_sn * sr)
    min_uzunluk = segment_uzunluk // 2
    return [
        y[bas: bas + segment_uzunluk]
        for bas in range(0, len(y), segment_uzunluk)
        if len(y[bas: bas + segment_uzunluk]) >= min_uzunluk
    ]


def duz_ozellik_cikar(y: np.ndarray, sr: int) -> np.ndarray:
    """
    LGBM / klasik ML için 250 boyutlu düz özellik vektörü.
    mfcc_ozellik_cikar (240) + spektral_ozellik_cikar (10)
    """
    if len(y) == 0:
        return np.zeros(250)
    return np.concatenate([
        mfcc_ozellik_cikar(y, sr),
        spektral_ozellik_cikar(y, sr),
    ])
