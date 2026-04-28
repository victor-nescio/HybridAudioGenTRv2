"""
BBC ses siniflandirma icin BiLSTM modelleri.

BiLSTMClassifier : Temporal attention ile cift yonlu LSTM (onerilen)
SimpleLSTM       : Vanilla BiLSTM (ablasyon / hiz testi)

Giris : (B, T_max, 128) — zaman-birinci log-mel dizi
Cikis : (B, N_SINIF) — ham logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-stili) temporal attention.
    Girdi: (B, T, H) -> Cikti: (B, H) agirlikli toplam
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        scores = self.attn(h).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=1)      # (B, T)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, H)
        return context


class BiLSTMClassifier(nn.Module):
    """
    Temporal attention ile cift yonlu LSTM siniflandirici.

    Mimari:
        Embedding projection (opsiyonel) ->
        N katman BiLSTM ->
        Temporal Attention ->
        FC siniflandirici

    Args:
        n_sinif    : cikti sinif sayisi
        n_mels     : giris ozellik boyutu (log-mel bant sayisi)
        hidden_dim : her yonun gizli boyutu (toplam 2*hidden_dim)
        n_katman   : LSTM katman sayisi
        dropout    : LSTM arasi + siniflandirici dropout
    """

    def __init__(
        self,
        n_sinif: int = 26,
        n_mels: int = 128,
        hidden_dim: int = 256,
        n_katman: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=n_katman,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_katman > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim * 2)

        self.siniflandirici = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, n_sinif),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 128)
        h, _ = self.lstm(x)              # (B, T, 2*hidden_dim)
        context = self.attention(h)      # (B, 2*hidden_dim)
        return self.siniflandirici(context)


class SimpleLSTM(nn.Module):
    """
    Son adim ciktisini kullanan basit BiLSTM.
    Ablasyon ve attention karsilastirmasi icin kullanilir.
    """

    def __init__(
        self,
        n_sinif: int = 26,
        n_mels: int = 128,
        hidden_dim: int = 128,
        n_katman: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=n_katman,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_katman > 1 else 0.0,
        )
        self.siniflandirici = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, n_sinif),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 128)
        h, _ = self.lstm(x)        # (B, T, 2*hidden_dim)
        return self.siniflandirici(h[:, -1, :])  # son adim
