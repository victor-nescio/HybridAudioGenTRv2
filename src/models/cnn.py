"""
BBC ses siniflandirma icin CNN modelleri.

SpectrogramCNN : ResNet18 tabanli transfer learning (onerilen)
SimpleCNN      : Sifirdan 3-blok CNN (ablasyon / hiz testi)

Giris : (B, 1, 128, T_max) — kanal-birinci log-mel spectrogram
Cikis : (B, N_SINIF) — ham logit
"""

import torch
import torch.nn as nn
from torchvision import models


class SpectrogramCNN(nn.Module):
    """
    ResNet18 transfer learning ile log-mel spectrogram siniflandirici.

    ImageNet agirlikliyla baslayip ilk conv katmani tek kanala adapte edilir.
    Son FC katmani hedef sinif sayisina gore yeniden yapilandirilir.

    Referans: Hershey et al. (2017) — CNN Architectures for Large-Scale Audio Classification
    """

    def __init__(self, n_sinif: int = 26, pretrained: bool = True):
        super().__init__()

        # ResNet18 yukle
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=pretrained)

        # Ilk conv: 3 kanal -> 1 kanal (geri kalan agirliklar korunur)
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Pretrained agirlik: 3 kanali ortalamasini tek kanala aktar
        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(
                    old_conv.weight.mean(dim=1, keepdim=True))

        # Son FC: 512 -> n_sinif
        in_features = backbone.fc.in_features
        backbone.fc  = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, n_sinif),
        )

        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, T_max)
        return self.model(x)


class SimpleCNN(nn.Module):
    """
    Sifirdan egitilen 3-bloklu kucuk CNN.
    Ablasyon calismasi ve pretrained karsilastirmasi icin kullanilir.

    Mimari: Conv-BN-ReLU-Pool x3 -> AdaptiveAvgPool -> FC
    """

    def __init__(self, n_sinif: int = 26, n_mels: int = 128):
        super().__init__()

        def blok(c_in, c_out, pool=True):
            katmanlar = [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            if pool:
                katmanlar.append(nn.MaxPool2d(2))
            return nn.Sequential(*katmanlar)

        self.ozellik = nn.Sequential(
            blok(1,   32),   # (1,128,T) -> (32,64,T/2)
            blok(32,  64),   # -> (64,32,T/4)
            blok(64, 128),   # -> (128,16,T/8)
            blok(128,256, pool=False),  # -> (256,16,T/8)
        )
        self.havuz = nn.AdaptiveAvgPool2d((4, 4))  # (256,4,4)
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_sinif),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ozellik(x)
        x = self.havuz(x)
        return self.siniflandirici(x)
