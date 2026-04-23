import numpy as np
import librosa


def ses_yukle(dosya_yolu: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    y, sr_out = librosa.load(dosya_yolu, sr=sr, mono=True)
    return y, sr_out


def dc_offset_kaldir(y: np.ndarray) -> np.ndarray:
    return y - np.mean(y)


def on_vurgu_uygula(y: np.ndarray, katsayi: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - katsayi * y[:-1])


def sessizlik_kirp(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    kirpilmis, _ = librosa.effects.trim(y, top_db=top_db)
    return kirpilmis if len(kirpilmis) > 0 else y


def tepe_normalize(y: np.ndarray, hedef: float = 0.99) -> np.ndarray:
    tepe = np.max(np.abs(y))
    return y * (hedef / tepe) if tepe > 0 else y


def onisle(dosya_yolu: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Tam preprocessing pipeline:
    yükle → DC offset kaldır → ön vurgu → sessizlik kırp → tepe normalize
    """
    y, sr_out = ses_yukle(dosya_yolu, sr)
    y = dc_offset_kaldir(y)
    y = on_vurgu_uygula(y)
    y = sessizlik_kirp(y)
    y = tepe_normalize(y)
    return y, sr_out
