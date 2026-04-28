"""
BBC ses siniflandirma icin Audio Spectrogram Transformer (AST).

ASTClassifier : HuggingFace pretrained AST, son katman 26 sinifa adapte edilir.

Referans: Gong et al. (2021) — AST: Audio Spectrogram Transformer
HuggingFace: MIT/ast-finetuned-audioset-10-10-0.4593

Giris : (B, 1, 128, 313) — dataset.py mod='ast' ciktisi
        (AudioSet mean=-4.2677 / std=4.5690*2 normalizasyonu uygulanmis olmali)
Cikis : (B, N_SINIF) — ham logit
"""

import torch
import torch.nn as nn
from transformers import ASTConfig, ASTForAudioClassification


class ASTClassifier(nn.Module):
    """
    Pretrained AST fine-tuning ile log-mel spectrogram siniflandirici.

    ImageNet/AudioSet agirliklariyla baslayip son siniflandirici katmani
    hedef sinif sayisina gore yeniden yapilandirilir.

    Args:
        n_sinif     : cikti sinif sayisi (varsayilan 26)
        pretrained  : HuggingFace hub'dan pretrained agirlik yukle
        model_adi   : HuggingFace model kimlik (varsayilan AudioSet pretrained)
        freeze_enc  : True ise encoder dondurulur, sadece siniflandirici egitilir
    """

    MODEL_ADI = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(
        self,
        n_sinif: int = 26,
        pretrained: bool = True,
        model_adi: str | None = None,
        freeze_enc: bool = False,
    ):
        super().__init__()

        model_adi = model_adi or self.MODEL_ADI

        # Config: bizim giris boyutuna (128 mel, 313 frame) gore ayarla
        config = ASTConfig.from_pretrained(model_adi)
        config.num_mel_bins = 128
        config.max_length   = 313   # T_MAX: 5sn × 16kHz / hop256

        if pretrained:
            # ignore_mismatched_sizes: position embedding yeniden baslatilir,
            # diger tum agirliklar (attention, FFN) korunur
            self.ast = ASTForAudioClassification.from_pretrained(
                model_adi,
                config=config,
                ignore_mismatched_sizes=True,
            )
        else:
            self.ast = ASTForAudioClassification(config)

        # Son siniflandirici katmani hedef sinif sayisina adapte et
        in_features = self.ast.classifier.dense.in_features
        self.ast.classifier.dense = nn.Linear(in_features, n_sinif)

        if freeze_enc:
            for isim, param in self.ast.named_parameters():
                if "classifier" not in isim:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 128, 313) — kanal boyutunu kaldir, AST (B, 128, 313) bekliyor
        if x.dim() == 4:
            x = x.squeeze(1)                 # (B, 128, 313)

        # AST icin (B, zaman, frekans) -> (B, frekans, zaman) transpose gerekebilir
        # HuggingFace AST: input_values sekli (B, freq_bins, time_frames)
        cikis = self.ast(input_values=x)
        return cikis.logits                  # (B, n_sinif)

    def unfreeze_encoder(self):
        """Encoder'i coz — freeze_enc=True ile baslatilip sonra serbest birakmak icin."""
        for param in self.ast.parameters():
            param.requires_grad = True
