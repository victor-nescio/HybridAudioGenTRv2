"""
CLAP Fine-Tuning — BBC Ses Arsivi ile Turkce/Ingilizce Kontrastif Ogrenme

Arastirma motivasyonu:
    AudioLDM2'nin CLAP encoder'i LAION-Audio-630K ve AudioSet uzerinde egitilmistir.
    BBC Ses Arsivi'nin kendine ozgu akustik karakterleri (1930-1990 arasi kayitlar,
    Ingiliz radyo efektleri, spesifik Ingilizce yer/nesne isimleri) ile Turkce
    aciklamalar arasindaki semantik kopuklugu kapatmak icin contrastive fine-tuning
    uygulanmaktadir.

Yontem (Wu et al. 2023 — CLAP; Elizalde et al. 2022 — CLAP-LAION):
    - NT-Xent kayip fonksiyonu (InfoNCE / SimCLR loss)
    - Soft negative mining: ayni siniftan farklı promptlar negative degil, hard
      negative olarak isaretlenir (beta agirlik ile yumusatilir)
    - Selective layer freezing: audio_branch ve text_branch'in alt katmanlari
      dondurulur; sadece projeksiyon katmanlari + ust transformer bloklari guncellenir
    - Dil dengesi: her batch'te %50 TR + %50 EN prompt (dil-dengeli ornekleme)
    - Gradient checkpointing: VRAM tasarrufu icin
    - EMA (Exponential Moving Average): stabil inference modeli icin

Dondurulan katmanlar:
    - audio_branch: ilk N_FROZEN_AUDIO transformer blogu donuk
    - text_branch: ilk N_FROZEN_TEXT transformer katmani donuk
    - Projeksiyon + transform katmanlari her zaman egitilir

Degerlendirme metrikleri:
    - R@1, R@5, R@10 (audio-to-text ve text-to-audio recall)
    - mAP@10
    - Sinif bazinda embedding uzayinda kNN dogrulugu

Kullanim:
    python scripts/08_clap_finetune.py --config             # varsayilan konfig
    python scripts/08_clap_finetune.py --epochs 20 --lr 3e-5
    python scripts/08_clap_finetune.py --sadece_ingilizce   # ablasyon: TR olmadan
    python scripts/08_clap_finetune.py --kuru_calistirma    # egitim baslatmadan kontrol
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────
PARQUET      = ROOT / "data" / "processed" / "ses_metin_eslestirme.parquet"
CHECKPT_DIR  = ROOT / "checkpoints"
RESULT_DIR   = ROOT / "results" / "v1"
SR           = 16_000       # CLAP beklenen ornek hizi
CLIP_SN      = 10.0         # CLAP beklenen klip suresi (saniye)
CLIP_SAMPLES = int(SR * CLIP_SN)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Hangi transformer bloklari dondurulacak (0-indexed)
N_FROZEN_AUDIO = 6    # HTSAT-base: toplam 12 blok → ilk 6 donuk
N_FROZEN_TEXT  = 8    # RoBERTa-base: toplam 12 katman → ilk 8 donuk


# ════════════════════════════════════════════════════════════════════════════
# Konfigurasyon
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FinetuneConfig:
    # Egitim
    epochs: int             = 30
    batch_size: int         = 16      # RTX 3060 12GB icin guvenli
    lr: float               = 3e-5
    weight_decay: float     = 1e-4
    warmup_orani: float     = 0.1     # toplam adimin %10'u warmup
    patience: int           = 7
    grad_clip: float        = 1.0

    # Kayip fonksiyonu
    sicaklik: float         = 0.07    # NT-Xent sicaklik parametresi (tau)
    # Hard negative azaltma: ayni siniftan negative'lere bu agirlik uygulanir
    # 0.0 = tamamen yok say, 1.0 = tam negative muamelesi
    hard_neg_beta: float    = 0.3

    # Dil dengesi
    tr_orani: float         = 0.5     # batch'te Turkce prompt orani
    sadece_ingilizce: bool  = False   # ablasyon

    # Dondurma stratejisi
    n_frozen_audio: int     = N_FROZEN_AUDIO
    n_frozen_text: int      = N_FROZEN_TEXT

    # Veri
    val_orani: float        = 0.1
    test_orani: float       = 0.1
    maks_saniye: float      = CLIP_SN
    num_workers: int        = 0       # Windows'ta multiprocessing sorunu yok

    # Diger
    seed: int               = 42
    fp16: bool              = False   # CLAP spectrogram ops fp32 gerektirir
    ema_decay: float        = 0.999
    log_adim: int           = 50      # kac adimda bir log

    # v2 (revizyon) ayarlari — v1 davranisi: surum="v1", hassasiyet="fp16",
    # secim_metrigi="A2T_R@1", parcali_okuma=False
    surum: str              = "v2"
    hassasiyet: str         = "fp32"  # fp32 | bf16 | fp16 (autocast)
    secim_metrigi: str      = "zs_ort"
    parcali_okuma: bool     = True


# ════════════════════════════════════════════════════════════════════════════
# Veri Seti
# ════════════════════════════════════════════════════════════════════════════

class BBCSesMetinVeriSeti(Dataset):
    """
    BBC ses-metin cifti veri seti.

    Her ornek:
        wav_tensor : (CLIP_SAMPLES,) float32 — normalize edilmis mono dalga formu
        prompt     : str — Turkce veya Ingilizce aciklama
        sinif_idx  : int — sinif etiketi (hard negative icin)
        dosya      : str — kayit icin
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: Path,
        sr: int = SR,
        clip_samples: int = CLIP_SAMPLES,
        tr_orani: float = 0.5,
        sadece_ingilizce: bool = False,
        augment: bool = False,
        parcali_okuma: bool = False,
    ):
        self.df           = df.reset_index(drop=True)
        self.root         = root
        self.sr           = sr
        self.clip_samples = clip_samples
        self.tr_orani     = 0.0 if sadece_ingilizce else tr_orani
        self.augment      = augment
        self.parcali_okuma = parcali_okuma

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        satir = self.df.iloc[idx]

        # ── Ses yukle — basarisizsa rastgele gecerli bir ornek sec ────────
        wav = self._wav_yukle(satir["dosya"])
        # Sifir tensoru (yuklenemedi) yerine rastgele gecerli ornek kullan
        if wav.abs().max() < 1e-6:
            yeni_idx = np.random.randint(0, len(self.df))
            satir    = self.df.iloc[yeni_idx]
            wav      = self._wav_yukle(satir["dosya"])

        # ── Prompt sec (dil dengesi) ──────────────────────────────────────
        if self.tr_orani > 0 and np.random.rand() < self.tr_orani:
            prompt = str(satir["prompt_tr"])
        else:
            prompt = str(satir["prompt_en"])

        return {
            "wav":       wav,
            "prompt":    prompt,
            "sinif_idx": int(satir["sinif_idx"]),
            "dosya":     str(satir["dosya"]),
        }

    def _wav_yukle(self, dosya_yolu: str) -> torch.Tensor:
        """WAV dosyasini yukle, CLIP_SAMPLES uzunluguna getir."""
        import soundfile as sf
        import resampy

        tam_yol = self.root / dosya_yolu
        try:
            if self.parcali_okuma:
                # Yalnizca gereken pencereyi oku (v2): tum dosyayi okuyup yeniden
                # orneklemek yerine; kirpma dagilimi ayni (dosya icinde duzgun rastgele baslangic)
                bilgi = sf.info(str(tam_yol))
                sr_oku = bilgi.samplerate
                pencere = int(math.ceil(self.clip_samples * sr_oku / self.sr))
                baslangic = 0
                if self.augment and bilgi.frames > pencere:
                    baslangic = np.random.randint(0, bilgi.frames - pencere)
                data, sr_oku = sf.read(str(tam_yol), dtype="float32", always_2d=False,
                                       start=baslangic, frames=pencere)
            else:
                data, sr_oku = sf.read(str(tam_yol), dtype="float32", always_2d=False)
        except Exception as e:
            log.warning(f"Ses yuklenemedi: {tam_yol} — {e}")
            return torch.zeros(self.clip_samples)

        # Stereo → mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Yeniden ornekleme
        if sr_oku != self.sr:
            data = resampy.resample(data, sr_oku, self.sr)

        # Kirp veya dolgula
        if len(data) >= self.clip_samples:
            # Rastgele baslangic noktasindan kirp (egitim augmentasyonu)
            if self.augment and len(data) > self.clip_samples and not self.parcali_okuma:
                baslangic = np.random.randint(0, len(data) - self.clip_samples)
                data = data[baslangic : baslangic + self.clip_samples]
            else:
                data = data[: self.clip_samples]
        else:
            # Sifir ile dolgula (sag taraf)
            data = np.pad(data, (0, self.clip_samples - len(data)))

        # Normalize: peak normalization
        maks = np.abs(data).max()
        if maks > 0:
            data = data / maks

        return torch.from_numpy(data.astype(np.float32))


def collate_fn(batch: list[dict]) -> dict:
    """Değişken uzunluklu WAV'lari batch'e topla."""
    return {
        "wav":       torch.stack([b["wav"] for b in batch]),
        "prompt":    [b["prompt"] for b in batch],
        "sinif_idx": torch.tensor([b["sinif_idx"] for b in batch], dtype=torch.long),
        "dosya":     [b["dosya"] for b in batch],
    }


def sinif_dengeli_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Az ornekli siniflar icin agirlikli ornekleyici."""
    sinif_sayilari = df["sinif_idx"].value_counts()
    agirliklar = df["sinif_idx"].map(lambda x: 1.0 / sinif_sayilari[x]).values
    return WeightedRandomSampler(
        weights=agirliklar,
        num_samples=len(agirliklar),
        replacement=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# CLAP Model Sargici
# ════════════════════════════════════════════════════════════════════════════

class CLAPFinetune(nn.Module):
    """
    CLAP'i BBC verisine fine-tune etmek icin sargi sinifi.

    AudioLDM2'nin kendi CLAPAudioEmbeddingClassifierFreev2 modeli kullanilir
    (audioldm2.clap.open_clip.factory.create_model — HTSAT-base + RoBERTa-base).
    Bu, AudioLDM2 checkpoint'indeki CLAP ile tam mimarı uyum saglar.

    Dondurma stratejisi:
        - audio_branch.layers[:n_frozen]: donuk (HTSAT Swin Transformer bloklari)
        - text_branch.encoder.layer[:n_frozen]: donuk (RoBERTa katmanlari)
        - Projeksiyon katmanlari: her zaman egitilir
    """

    def __init__(self, cfg: FinetuneConfig):
        super().__init__()
        from audioldm2.latent_diffusion.modules.encoders.modules import (
            CLAPAudioEmbeddingClassifierFreev2,
        )
        from transformers import RobertaTokenizer

        self.cfg = cfg
        self.son_nan_sayisi = 0

        log.info("AudioLDM2 CLAP modeli yukleniyor (HTSAT-base + RoBERTa-base)...")
        # training_mode=True: dropout aktif, gradient akisi acik
        clap_enc = CLAPAudioEmbeddingClassifierFreev2(
            sampling_rate=SR,
            embed_mode="audio",
            amodel="HTSAT-base",
            training_mode=True,
        )
        self.model   = clap_enc.model         # CLAP çekirdek modeli
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")

        self.audio_cfg = clap_enc.model_cfg["audio_cfg"]
        # HTSAT kendi mel cikarmini yapiyor — harici transform gerekmez

        # AudioLDM2 checkpoint'inden pretrained CLAP agirliklarini yukle
        self._audioldm2_agirliklarini_yukle()

        # Once tum parametreleri egitilabilir yap, sonra secici dondur
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._katmanlari_dondur()
        self._egitilecek_parametreler_yaz()

    def _audioldm2_agirliklarini_yukle(self):
        """
        AudioLDM2 tam checkpoint'inden CLAP agirliklarini cikar ve yukle.
        Checkpoint'teki 'clap.model.*' keyleri bu modelin 'model.*' keylerini oluşturur.
        """
        from audioldm2 import download_checkpoint
        ckpt_path = download_checkpoint("audioldm2-full")
        log.info(f"AudioLDM2 checkpoint'inden CLAP agirliklar yukleniyor: {ckpt_path}")
        # mmap: 8,5 GB'lik tam checkpoint RAM'e kopyalanmadan okunur (yalniz CLAP anahtarlari kullanilir)
        sd_tam = torch.load(ckpt_path, map_location="cpu", mmap=True)["state_dict"]

        # 'clap.model.' prefix'ini 'model.' ile degistir
        clap_sd = {}
        for k, v in sd_tam.items():
            if k.startswith("clap.model."):
                yeni_k = k[len("clap."):]   # 'clap.model.X' → 'model.X'
                clap_sd[yeni_k] = v

        eksik, beklenmeyen = self.load_state_dict(clap_sd, strict=False)
        log.info(
            f"CLAP agirlik yukleme: {len(clap_sd)} key yukleneli  "
            f"eksik={len(eksik)}  beklenmeyen={len(beklenmeyen)}"
        )

    def _katmanlari_dondur(self):
        """Secici katman dondurma."""
        # Audio branch — HTSAT Swin Transformer bloklari (4 stage)
        audio_frozen_n = 0
        for p in self.model.audio_branch.patch_embed.parameters():
            p.requires_grad_(False)

        # Spectrogram + logmel: sabit DSP filtreleri, egitilmez, NaN kaynagi olabilir
        for sub in ["spectrogram_extractor", "logmel_extractor", "bn0"]:
            if hasattr(self.model.audio_branch, sub):
                for p in getattr(self.model.audio_branch, sub).parameters():
                    p.requires_grad_(False)

        if hasattr(self.model.audio_branch, "layers"):
            for stage in self.model.audio_branch.layers:
                if audio_frozen_n >= self.cfg.n_frozen_audio:
                    break
                for blk in stage.blocks:
                    for p in blk.parameters():
                        p.requires_grad_(False)
                    audio_frozen_n += 1
                    if audio_frozen_n >= self.cfg.n_frozen_audio:
                        break

        # Text branch — RoBERTa encoder katmanlari
        text_frozen_n = min(
            self.cfg.n_frozen_text,
            len(self.model.text_branch.encoder.layer),
        )
        for p in self.model.text_branch.embeddings.parameters():
            p.requires_grad_(False)
        for layer in self.model.text_branch.encoder.layer[:text_frozen_n]:
            for p in layer.parameters():
                p.requires_grad_(False)

        log.info(f"Dondurma: audio {audio_frozen_n} blok, text {text_frozen_n} katman")

    def _egitilecek_parametreler_yaz(self):
        toplam     = sum(p.numel() for p in self.parameters())
        egitilecek = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"Parametreler: {egitilecek/1e6:.2f}M egitilecek / "
            f"{toplam/1e6:.2f}M toplam  ({100*egitilecek/toplam:.1f}%)"
        )

    def ses_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, CLIP_SAMPLES) → (B, D) normalize embedding."""
        import torchaudio
        wav = wav.to(DEVICE)
        # HTSAT 48kHz bekliyor; wav 16kHz geliyor
        target_sr = self.audio_cfg["sample_rate"]
        if SR != target_sr:
            wav = torchaudio.functional.resample(wav, SR, target_sr)
        # HTSAT input: dict{"waveform": (B, samples)} — kendi mel cikarmini yapiyor
        out = self.model.audio_branch({"waveform": wav}, mixup_lambda=None)
        emb = self.model.audio_projection(out["embedding"])
        # NaN sayaci: v1'de NaN gommeler sessizce gurultuyle degistiriliyordu;
        # v2'de sayilip loglanir (egitim dongusu ve degerlendirme raporlar)
        self.son_nan_sayisi = int(emb.isnan().any(dim=-1).sum().item())
        # NaN koruması: sifir vektoru yerine küçük gurultu
        emb = torch.where(emb.isnan(), torch.randn_like(emb) * 1e-6, emb)
        return F.normalize(emb, dim=-1)

    def metin_embedding(self, promptlar: list[str]) -> torch.Tensor:
        """[str, ...] → (B, D) normalize embedding."""
        tokenized = self.tokenize(
            promptlar,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(DEVICE)
        out = self.model.text_branch(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        )
        if self.cfg.surum == "v1":
            # v1 (hatali): ham [CLS] gizli durumu. LAION-CLAP/AudioLDM2 metin
            # kodlamasi RoBERTa "pooler_output" kullanir (open_clip/model.py
            # encode_text); bu nedenle v1 metin gommeleri CLAP'in ortak uzayiyla
            # hizasizdi. Yalnizca v1'i birebir yeniden uretmek icin korunuyor.
            havuz = out.last_hidden_state[:, 0, :]
        else:
            havuz = out.pooler_output   # AudioLDM2'nin uretimde kullandigi yol
        emb = self.model.text_projection(havuz)
        return F.normalize(emb, dim=-1)

    def forward(self, wav: torch.Tensor, promptlar: list[str]) -> tuple:
        audio_emb = self.ses_embedding(wav)
        metin_emb = self.metin_embedding(promptlar)
        return audio_emb, metin_emb


# ════════════════════════════════════════════════════════════════════════════
# Kayip Fonksiyonu — NT-Xent + Hard Negative
# ════════════════════════════════════════════════════════════════════════════

class NTXentHardNeg(nn.Module):
    """
    NT-Xent (InfoNCE) kayip fonksiyonu — soft hard negative destegiyle.

    Referans:
        Chen et al. 2020 — SimCLR
        Wu et al. 2023 — CLAP

    Hard negative mantigi:
        Ayni siniftan iki ornek birbirine "cok benzer" — bunlari tam negative
        saymak yaniltici olabilir. Beta agirligiyla yumusatilir:
            logit[i,j] = logit[i,j] * beta   (ayni sinif, farkli ornek)
    """

    def __init__(self, sicaklik: float = 0.07, hard_neg_beta: float = 0.3):
        super().__init__()
        self.tau  = sicaklik
        self.beta = hard_neg_beta

    def forward(
        self,
        audio_emb: torch.Tensor,    # (B, D)
        metin_emb: torch.Tensor,    # (B, D)
        sinif_idx: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        B = audio_emb.size(0)

        # Cosine benzerlik matrisi / tau
        logits = (audio_emb @ metin_emb.T) / self.tau  # (B, B)

        # Hard negative masking: ayni sinif, farkli ornek → yumusatilmis logit
        sinif_matrisi = sinif_idx.unsqueeze(0) == sinif_idx.unsqueeze(1)  # (B, B)
        kimlik        = torch.eye(B, dtype=torch.bool, device=DEVICE)
        hard_neg_mask = sinif_matrisi & ~kimlik   # ayni sinif ama kendisi degil

        # Hard negative'lerin logit'lerini azalt (onlari daha az cezbedici yap)
        logits = logits.clone()
        logits[hard_neg_mask] = logits[hard_neg_mask] * self.beta

        # Simetrik InfoNCE kaybi
        hedef = torch.arange(B, device=DEVICE)
        kayip_a2t = F.cross_entropy(logits,   hedef)   # audio → text
        kayip_t2a = F.cross_entropy(logits.T, hedef)   # text → audio
        return (kayip_a2t + kayip_t2a) / 2.0


# ════════════════════════════════════════════════════════════════════════════
# EMA (Exponential Moving Average)
# ════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Model agirliklarinin EMA'sini tutar.
    Inference'da EMA agirliklar kullanilarak daha stabil sonuclar alinir.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self.yedek: dict  = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def guncelle(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * p.data
                )

    def yukle(self):
        """EMA agirliklarini modele yukle (inference icin)."""
        for name, p in self.model.named_parameters():
            if name in self.shadow:
                self.yedek[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def geri_al(self):
        """Orijinal egitim agirliklarini geri yukle."""
        for name, p in self.model.named_parameters():
            if name in self.yedek:
                p.data.copy_(self.yedek[name])
        self.yedek.clear()


# ════════════════════════════════════════════════════════════════════════════
# Degerlendirme Metrikleri
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def recall_at_k(
    audio_emb: torch.Tensor,
    metin_emb: torch.Tensor,
    k_degerleri: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Audio-to-Text ve Text-to-Audio Recall@K hesapla.

    Referans: Doh et al. 2023 — LP-MusicCaps
    """
    sim = audio_emb @ metin_emb.T   # (N, N)
    N   = sim.size(0)
    hedef = torch.arange(N, device=sim.device)

    sonuclar = {}
    for k in k_degerleri:
        k_gercek = min(k, N)

        # Audio → Text
        _, top_idx = sim.topk(k_gercek, dim=1)
        a2t_isabet = (top_idx == hedef.unsqueeze(1)).any(dim=1).float().mean().item()

        # Text → Audio
        _, top_idx = sim.T.topk(k_gercek, dim=1)
        t2a_isabet = (top_idx == hedef.unsqueeze(1)).any(dim=1).float().mean().item()

        sonuclar[f"A2T_R@{k}"] = round(a2t_isabet, 4)
        sonuclar[f"T2A_R@{k}"] = round(t2a_isabet, 4)

    return sonuclar


@torch.no_grad()
def sinif_knn_dogrulugu(
    audio_emb: torch.Tensor,
    sinif_idx: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Audio embedding uzayinda k-NN sinif dogrulugu.
    Embedding kalitesinin sinif ayirt ediciligi acisindan olcusu.
    """
    sim = audio_emb @ audio_emb.T
    sim.fill_diagonal_(-1e9)          # kendini disla

    _, top_idx = sim.topk(k, dim=1)   # (N, k)
    komsu_siniflar = sinif_idx[top_idx]   # (N, k)
    tahmin = komsu_siniflar.mode(dim=1).values
    return (tahmin == sinif_idx).float().mean().item()


def autocast_ayari(hassasiyet: str) -> dict:
    """torch.autocast parametreleri: fp32 -> kapali; bf16/fp16 -> ilgili dtype."""
    if DEVICE != "cuda" or hassasiyet == "fp32":
        return {"device_type": "cuda", "enabled": False}
    dtype = torch.bfloat16 if hassasiyet == "bf16" else torch.float16
    return {"device_type": "cuda", "dtype": dtype, "enabled": True}


def sinif_promptlari_yukle() -> dict:
    """{sinif_adi: {"ingilizce": [...], "turkce": [...]}} — uretimde kullanilan sablonlar."""
    with open(ROOT / "data" / "processed" / "turkce_promptlar.json", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def sifir_atis_dogrulugu(model, audio_emb, sinif_idx, sinif_adlari, hassasiyet) -> dict:
    """Sinif duzeyinde capraz-kip uyum: her sesi, sinifin sablon promptlarinin
    ortalama metin gommesine en yakin sinifa ata (EN ve TR ayri).
    sinif_adlari: {sinif_idx: sinif_adi}"""
    promptlar = sinif_promptlari_yukle()
    idxler = sorted(sinif_adlari)
    sonuc = {}
    for dil, kisa in (("ingilizce", "en"), ("turkce", "tr")):
        with torch.autocast(**autocast_ayari(hassasiyet)):
            T = torch.stack([
                model.metin_embedding(promptlar[sinif_adlari[i]][dil]).float().mean(0)
                for i in idxler
            ])
        T = F.normalize(T.cpu(), dim=-1)
        tahmin = torch.tensor(idxler)[(audio_emb.float() @ T.T).argmax(1)]
        sonuc[f"zs_{kisa}"] = round((tahmin == sinif_idx).float().mean().item(), 4)
    sonuc["zs_ort"] = round((sonuc["zs_en"] + sonuc["zs_tr"]) / 2, 4)
    return sonuc


@torch.no_grad()
def val_degerlendirme(
    model: CLAPFinetune,
    loader: DataLoader,
    kayip_fn: NTXentHardNeg,
    hassasiyet: str = "fp16",
) -> dict:
    """Validasyon seti uzerinde tam degerlendirme."""
    model.eval()
    tum_audio, tum_metin, tum_sinif = [], [], []
    toplam_kayip, n_batch, n_nan = 0.0, 0, 0
    sinif_adlari = dict(zip(loader.dataset.df["sinif_idx"], loader.dataset.df["sinif"]))

    for batch in loader:
        wav       = batch["wav"].to(DEVICE)
        promptlar = batch["prompt"]
        siniflar  = batch["sinif_idx"].to(DEVICE)

        with torch.autocast(**autocast_ayari(hassasiyet)):
            a_emb, t_emb = model(wav, promptlar)
            kayip = kayip_fn(a_emb, t_emb, siniflar)
        n_nan += model.son_nan_sayisi

        toplam_kayip += kayip.item()
        n_batch      += 1
        tum_audio.append(a_emb.cpu())
        tum_metin.append(t_emb.cpu())
        tum_sinif.append(siniflar.cpu())

    audio_emb = torch.cat(tum_audio)
    metin_emb = torch.cat(tum_metin)
    sinif_idx = torch.cat(tum_sinif)

    metrikler = {"val_kayip": round(toplam_kayip / n_batch, 4)}
    metrikler.update(recall_at_k(audio_emb, metin_emb))
    metrikler["knn_acc"] = round(sinif_knn_dogrulugu(audio_emb, sinif_idx), 4)
    metrikler.update(sifir_atis_dogrulugu(model, audio_emb, sinif_idx, sinif_adlari, hassasiyet))
    metrikler["nan_ses_orani"] = round(n_nan / len(sinif_idx), 4)

    return metrikler


# ════════════════════════════════════════════════════════════════════════════
# Warmup + Cosine LR Scheduler
# ════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Lineer warmup + cosine annealing scheduler.
    torch.optim.lr_scheduler ile uyumlu __call__ arayuzu.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_adim: int,
        toplam_adim: int,
        min_lr_orani: float = 0.01,
    ):
        self.optimizer    = optimizer
        self.warmup_adim  = warmup_adim
        self.toplam_adim  = toplam_adim
        self.min_lr_orani = min_lr_orani
        self.mevcut_adim  = 0
        self.baz_lr       = [g["lr"] for g in optimizer.param_groups]

    def adim(self):
        self.mevcut_adim += 1
        t = self.mevcut_adim

        if t < self.warmup_adim:
            lr_orani = t / max(1, self.warmup_adim)
        else:
            ilerleme = (t - self.warmup_adim) / max(
                1, self.toplam_adim - self.warmup_adim
            )
            lr_orani = self.min_lr_orani + 0.5 * (1 - self.min_lr_orani) * (
                1 + math.cos(math.pi * ilerleme)
            )

        for g, baz in zip(self.optimizer.param_groups, self.baz_lr):
            g["lr"] = baz * lr_orani


# ════════════════════════════════════════════════════════════════════════════
# Veri Bolme
# ════════════════════════════════════════════════════════════════════════════

def veri_bol(
    df: pd.DataFrame,
    val_orani: float,
    test_orani: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dosya duzeyinde stratified split.
    Ayni dosyanin farkli promptlari ayni bolume gider (veri sizintisi yok).
    """
    from sklearn.model_selection import train_test_split

    dosya_sinif = (
        df.groupby("dosya")["sinif_idx"].first().reset_index()
    )

    d_tv, d_te = train_test_split(
        dosya_sinif,
        test_size=test_orani,
        stratify=dosya_sinif["sinif_idx"],
        random_state=seed,
    )
    val_orani_ayarli = val_orani / (1 - test_orani)
    d_tr, d_vl = train_test_split(
        d_tv,
        test_size=val_orani_ayarli,
        stratify=d_tv["sinif_idx"],
        random_state=seed,
    )

    def filtre(d_alt):
        return df[df["dosya"].isin(d_alt["dosya"])].reset_index(drop=True)

    return filtre(d_tr), filtre(d_vl), filtre(d_te)


def veri_bol_v2(
    df: pd.DataFrame,
    val_orani: float,
    test_orani: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    v2 (sizintisiz) bolme: CLAP yalnizca SINIFLANDIRICININ EGITIM bolumundeki
    kayitlari gorur. Siniflandiricinin bolmesi src.data.dataset.veri_bol ile
    (07_iterative_loop ile ayni fonksiyon, seed=42, 70/15/15) yeniden uretilir;
    CLAP'in kendi egitim/val/test bolmesi bu kumenin icinde yapilir.
    Sentetik satirlar (gercek arsivde olmayan dosyalar) yalnizca CLAP egitimine girer.
    Dondurur: (tr, vl, te, denetim) — denetim: bolme ozeti + ortusme kontrolu.
    """
    from src.data.dataset import veri_bol as siniflandirici_bol

    clf_tr, clf_vl, clf_te = siniflandirici_bol(
        ROOT / "data" / "processed" / "veri_listesi.parquet", ROOT)
    norm = lambda s: str(s).replace("/", "\\")
    clf_tr_set = set(clf_tr["dosya"].map(norm))
    clf_yasak  = set(clf_vl["dosya"].map(norm)) | set(clf_te["dosya"].map(norm))

    gercek_mi   = df["dosya"].map(norm).isin(clf_tr_set | clf_yasak)
    df_gercek   = df[gercek_mi & df["dosya"].map(norm).isin(clf_tr_set)]
    df_sentetik = df[~gercek_mi]

    tr, vl, te = veri_bol(df_gercek, val_orani, test_orani, seed)
    if len(df_sentetik):
        tr = pd.concat([tr, df_sentetik], ignore_index=True)

    clap_dosyalar = set(pd.concat([tr, vl, te])["dosya"].map(norm))
    ortusme = clap_dosyalar & clf_yasak
    assert not ortusme, f"SIZINTI: {len(ortusme)} CLAP dosyasi siniflandirici val/test'te"

    denetim = {
        "siniflandirici_dosya": {"train": len(clf_tr_set), "val": len(set(clf_vl["dosya"])),
                                 "test": len(set(clf_te["dosya"]))},
        "clap_dosya": {"train_gercek": int(tr["dosya"].map(norm).isin(clf_tr_set).sum()),
                       "train_sentetik": int(len(df_sentetik)),
                       "val": int(vl["dosya"].nunique()), "test": int(te["dosya"].nunique())},
        "clap_ile_siniflandirici_val_test_ortusmesi": 0,
        "clap_val_dosyalari": sorted(vl["dosya"].unique().tolist()),
        "clap_test_dosyalari": sorted(te["dosya"].unique().tolist()),
    }
    return tr, vl, te, denetim


# ════════════════════════════════════════════════════════════════════════════
# Egitim Dongusu
# ════════════════════════════════════════════════════════════════════════════

def guvenli_kaydet(nesne: dict, yol: Path):
    """Atomik kayit: once gecici dosyaya yaz, sonra yerine koy. Kayit sirasinda
    elektrik kesilse bile onceki checkpoint saglam kalir."""
    gecici = yol.with_name(yol.name + ".tmp")
    torch.save(nesne, gecici)
    os.replace(gecici, yol)


def rng_durumu() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def rng_yukle(d: dict):
    # Checkpoint map_location=cuda ile yuklenince RNG tensorleri de GPU'ya tasinir;
    # set_rng_state CPU ByteTensor bekler
    torch.set_rng_state(d["torch"].cpu())
    if d.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in d["cuda"]])
    np.random.set_state(d["numpy"])
    random.setstate(d["python"])


def egitim_dongusu(
    model: CLAPFinetune,
    kayip_fn: NTXentHardNeg,
    dl_tr: DataLoader,
    dl_vl: DataLoader,
    cfg: FinetuneConfig,
    ema: EMA,
    baslangic_epoch: int = 1,
    resume_metrik: float = -1.0,
    ckpt_adi: str = "clap_bbc_best.pt",
    durum: Optional[dict] = None,
    kesinti_epoch: Optional[int] = None,
) -> dict:
    """
    Tam egitim dongusu.
    durum: '_son.pt' checkpoint'inden gelen tam egitim durumu (kesintiden devam).
           Optimizer, LR takvimi, en iyi metrik, sabir sayaci, RNG ve epoch gecmisi
           geri yuklenir; boylece devam eden kosu kesintisiz kosuyla ayni yolu izler.
    kesinti_epoch: (yalniz test) bu epoch'un kaydindan sonra kesintiyi taklit et.
    Returns: egitim tarihi (epoch bazinda metrikler)
    """
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),   # CLAP makalesi onerisi
    )

    toplam_adim  = len(dl_tr) * cfg.epochs
    warmup_adim  = int(toplam_adim * cfg.warmup_orani)
    scheduler    = WarmupCosineScheduler(optimizer, warmup_adim, toplam_adim)
    # GradScaler yalnizca fp16'da gerekir (bf16/fp32 tasma riski tasimaz)
    scaler       = GradScaler(enabled=(DEVICE == "cuda" and cfg.hassasiyet == "fp16"))
    ac_ayar      = autocast_ayari(cfg.hassasiyet)

    en_iyi_metrik = resume_metrik  # cfg.secim_metrigi izlenir
    en_iyi_epoch  = baslangic_epoch - 1
    patience_sayac = 0
    tarih = []
    gecen_sn = 0.0

    if durum is not None and "optimizer" in durum:
        # Scheduler taban LR'yi yakaladiktan SONRA optimizer durumu yuklenir
        optimizer.load_state_dict(durum["optimizer"])
        scheduler.mevcut_adim = durum["scheduler_adim"]
        scaler.load_state_dict(durum["scaler"])
        en_iyi_metrik  = durum["en_iyi_metrik"]
        en_iyi_epoch   = durum["en_iyi_epoch"]
        patience_sayac = durum["patience_sayac"]
        tarih          = list(durum["tarih"])
        gecen_sn       = durum.get("gecen_sn", 0.0)
        rng_yukle(durum["rng"])
        log.info(f"Tam durum geri yuklendi: LR adimi {scheduler.mevcut_adim}, en iyi "
                 f"{cfg.secim_metrigi}={en_iyi_metrik:.4f} (epoch {en_iyi_epoch}), sabir {patience_sayac}")
        if patience_sayac >= cfg.patience:
            log.info("Erken durdurma kosulu zaten saglanmis; egitim atlaniyor.")
            baslangic_epoch = cfg.epochs + 1

    ckpt_yolu = CHECKPT_DIR / ckpt_adi
    CHECKPT_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Egitim basliyor: epoch {baslangic_epoch}/{cfg.epochs}, {toplam_adim} toplam adim")
    log.info(f"Warmup: {warmup_adim} adim, LR: {cfg.lr}")

    for epoch in range(baslangic_epoch, cfg.epochs + 1):
        model.train()
        epoch_kayip = 0.0
        epoch_nan, epoch_ornek = 0, 0
        t0 = time.time()

        for adim, batch in enumerate(dl_tr, 1):
            wav       = batch["wav"].to(DEVICE)
            promptlar = batch["prompt"]
            siniflar  = batch["sinif_idx"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(**ac_ayar):
                a_emb, t_emb = model(wav, promptlar)
                kayip = kayip_fn(a_emb, t_emb, siniflar)
            epoch_nan   += model.son_nan_sayisi
            epoch_ornek += wav.size(0)

            # NaN kayip: bu batch'i atla, optimizer adimi yapma
            if torch.isnan(kayip) or torch.isinf(kayip):
                log.warning(f"  [!] NaN/Inf kayip adim {adim}, batch atlaniyor")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(kayip).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.adim()
            ema.guncelle()

            epoch_kayip += kayip.item()

            if adim % cfg.log_adim == 0:
                lr_simdi = optimizer.param_groups[0]["lr"]
                log.info(
                    f"  Epoch {epoch}/{cfg.epochs}  "
                    f"Adim {adim}/{len(dl_tr)}  "
                    f"Kayip={kayip.item():.4f}  "
                    f"LR={lr_simdi:.2e}"
                )

        # ── Epoch sonu degerlendirme (EMA agirliklarla) ───────────────────
        ema.yukle()
        val_m = val_degerlendirme(model, dl_vl, kayip_fn, cfg.hassasiyet)
        ema.geri_al()

        sure = (time.time() - t0) / 60
        ort_kayip = epoch_kayip / len(dl_tr)
        nan_orani = epoch_nan / max(1, epoch_ornek)

        log.info(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={ort_kayip:.4f}  "
            f"val_loss={val_m['val_kayip']:.4f}  "
            f"A2T_R@1={val_m['A2T_R@1']:.4f}  "
            f"kNN={val_m['knn_acc']:.4f}  "
            f"ZS(en/tr)={val_m['zs_en']:.4f}/{val_m['zs_tr']:.4f}  "
            f"NaN(train)={nan_orani:.3f}  "
            f"({sure:.1f}dk)"
        )
        if nan_orani > 0:
            log.warning(f"  [!] Egitimde ses gommelerinin %{100*nan_orani:.1f}'i NaN "
                        f"(hassasiyet={cfg.hassasiyet}) — sonuc guvenilir degil")

        epoch_kayit = {
            "epoch": epoch,
            "train_loss": round(ort_kayip, 5),
            "nan_ses_orani_train": round(nan_orani, 4),
            **val_m,
            "sure_dk": round(sure, 1),
        }
        tarih.append(epoch_kayit)

        # En iyi model: cfg.secim_metrigi ile izle (v1: A2T_R@1)
        if val_m[cfg.secim_metrigi] > en_iyi_metrik:
            en_iyi_metrik  = val_m[cfg.secim_metrigi]
            en_iyi_epoch   = epoch
            patience_sayac = 0

            # EMA agirliklarla kaydet
            ema.yukle()
            guvenli_kaydet(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "ema_shadow":  ema.shadow,
                    "val_metrik":  val_m,
                    "cfg":         asdict(cfg),
                },
                ckpt_yolu,
            )
            ema.geri_al()
            log.info(f"  ✓ En iyi model kaydedildi: {cfg.secim_metrigi}={en_iyi_metrik:.4f}")
        else:
            patience_sayac += 1

        # Her epoch sonunda tam "son durum" checkpoint'i (kesintiden devam icin)
        gecen_sn += time.time() - t0
        son_ckpt_yolu = ckpt_yolu.parent / ckpt_yolu.name.replace(".pt", "_son.pt")
        guvenli_kaydet(
            {
                "epoch":          epoch,
                "model_state":    model.state_dict(),   # ham (EMA olmayan) agirliklar
                "ema_shadow":     ema.shadow,
                "val_metrik":     val_m,
                "cfg":            asdict(cfg),
                "optimizer":      optimizer.state_dict(),
                "scheduler_adim": scheduler.mevcut_adim,
                "scaler":         scaler.state_dict(),
                "en_iyi_metrik":  en_iyi_metrik,
                "en_iyi_epoch":   en_iyi_epoch,
                "patience_sayac": patience_sayac,
                "tarih":          tarih,
                "gecen_sn":       gecen_sn,
                "rng":            rng_durumu(),
            },
            son_ckpt_yolu,
        )
        log.info(f"  Son durum kaydedildi: epoch {epoch} -> {son_ckpt_yolu.name}")
        if kesinti_epoch is not None and epoch >= kesinti_epoch:
            log.warning(f"  [TEST] Kesinti taklidi: epoch {epoch} sonrasi cikiliyor (kod 3)")
            sys.exit(3)
        durdur = ROOT / "outputs" / "v2_loglar" / "kuyruk" / "DURDUR"
        if durdur.exists() and epoch < cfg.epochs and patience_sayac < cfg.patience:
            log.warning(f"  [DURDUR] Istek alindi: epoch {epoch} durumu kaydedildi, cikiliyor (kod 4)")
            sys.exit(4)
        if patience_sayac >= cfg.patience:
                log.info(
                    f"Erken durdurma: {patience_sayac} epoch iyilesme yok. "
                    f"En iyi: epoch {en_iyi_epoch}"
                )
                break

    return {
        "tarih":         tarih,
        "en_iyi_epoch":  en_iyi_epoch,
        "en_iyi_metrik": en_iyi_metrik,
        "ckpt":          str(ckpt_yolu),
        "gecen_sn":      gecen_sn,
    }


# ════════════════════════════════════════════════════════════════════════════
# Kuru Calistirma (Dry Run)
# ════════════════════════════════════════════════════════════════════════════

def kuru_calistirma(cfg: FinetuneConfig):
    """
    Egitim baslatmadan; veri, model ve batch akisini dogrula.
    Hafiza tahminleri de yapilir.
    """
    log.info("=" * 60)
    log.info("  KUR CALISTIRMA — egitim baslatilmiyor")
    log.info("=" * 60)

    # Veri
    log.info(f"\nVeri: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    log.info(f"  Toplam: {len(df)} satir, {df['sinif'].nunique()} sinif")

    df_tr, df_vl, df_te = veri_bol(df, cfg.val_orani, cfg.test_orani, cfg.seed)
    log.info(f"  Train={len(df_tr)}  Val={len(df_vl)}  Test={len(df_te)}")

    # Model
    log.info("\nModel yukleniyor...")
    model = CLAPFinetune(cfg).to(DEVICE)

    # Tek batch testi
    log.info("\nTek batch testi...")
    ds = BBCSesMetinVeriSeti(df_tr.head(cfg.batch_size * 2), ROOT,
                             tr_orani=cfg.tr_orani,
                             sadece_ingilizce=cfg.sadece_ingilizce)
    dl = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate_fn)
    batch = next(iter(dl))

    wav       = batch["wav"].to(DEVICE)
    promptlar = batch["prompt"]
    siniflar  = batch["sinif_idx"].to(DEVICE)

    log.info(f"  WAV shape: {wav.shape}, dtype: {wav.dtype}")
    log.info(f"  Promptlar[0]: '{promptlar[0]}'")

    kayip_fn = NTXentHardNeg(cfg.sicaklik, cfg.hard_neg_beta)

    # Kuru calistirmada fp32 kullan (fp16 overflow olasiligi)
    with autocast(enabled=False):
        a_emb, t_emb = model(wav, promptlar)
        kayip = kayip_fn(a_emb, t_emb, siniflar)

    log.info(f"  Audio emb: {a_emb.shape}")
    log.info(f"  Metin emb: {t_emb.shape}")
    log.info(f"  Kayip: {kayip.item():.4f}")

    # VRAM tahmini
    if DEVICE == "cuda":
        ayrilan = torch.cuda.memory_allocated() / 1e9
        ayrilmis = torch.cuda.memory_reserved() / 1e9
        log.info(f"\nVRAM: {ayrilan:.2f}GB ayrilan / {ayrilmis:.2f}GB rezerv")

    log.info("\nKuru calistirma basarili — egitim baslatmak icin --kuru_calistirma olmadan calistirin.")


# ════════════════════════════════════════════════════════════════════════════
# Ana
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLAP fine-tuning — BBC Ses Arsivi"
    )
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--batch_size",          type=int,   default=16)
    parser.add_argument("--lr",                  type=float, default=3e-5)
    parser.add_argument("--patience",            type=int,   default=7)
    parser.add_argument("--sicaklik",            type=float, default=0.07,
                        help="NT-Xent sicaklik parametresi tau")
    parser.add_argument("--hard_neg_beta",       type=float, default=0.3,
                        help="Hard negative yumusatma katsayisi (0=yok, 1=tam)")
    parser.add_argument("--n_frozen_audio",      type=int,   default=N_FROZEN_AUDIO,
                        help="Audio branch'te donuk blok sayisi")
    parser.add_argument("--n_frozen_text",       type=int,   default=N_FROZEN_TEXT,
                        help="Text branch'te donuk katman sayisi")
    parser.add_argument("--tr_orani",            type=float, default=0.5,
                        help="Batch'te Turkce prompt orani")
    parser.add_argument("--sadece_ingilizce",    action="store_true",
                        help="Ablasyon: sadece Ingilizce prompt kullan")
    parser.add_argument("--kuru_calistirma",     action="store_true",
                        help="Egitim baslatmadan dogrulama yap")
    parser.add_argument("--fp16",                action="store_true", default=True,
                        help="(v1 uyumlulugu; etkisiz — hassasiyet --hassasiyet ile secilir)")
    parser.add_argument("--surum",               choices=["v1", "v2"], default="v2",
                        help="v1: ilk gonderim davranisi (tum arsiv bolmesi, fp16, A2T_R@1 secimi, "
                             "results/v1, clap_bbc_*.pt). v2: yalniz siniflandirici-egitim bolmesi, "
                             "results/v2, clap_v2_*.pt")
    parser.add_argument("--hassasiyet",          choices=["fp32", "bf16", "fp16"], default=None,
                        help="Varsayilan: v1 -> fp16, v2 -> fp32")
    parser.add_argument("--secim_metrigi",       type=str, default=None,
                        help="En iyi model olcutu. Varsayilan: v1 -> A2T_R@1, v2 -> zs_ort")
    parser.add_argument("--duman",               action="store_true",
                        help="Duman testi: sinif basina birkac dosya, 1 epoch; ciktilar _duman alt klasorlerine")
    parser.add_argument("--duman_epoch",         type=int, default=1,
                        help="Duman testinde epoch sayisi (kesinti/devam testi icin 2)")
    parser.add_argument("--kesinti_epoch",       type=int, default=None,
                        help="(Yalniz test) bu epoch kaydedildikten sonra kesintiyi taklit et (cikis kodu 3)")
    parser.add_argument("--ema_decay",           type=float, default=0.999)
    parser.add_argument("--resume",              action="store_true",
                        help="Mevcut clap_bbc_best.pt checkpoint'ten devam et")
    parser.add_argument("--sentetik_veri",       type=str, default=None,
                        help="Sentetik WAV dizini (orn: data/synthetic_tur2). "
                             "Promptlar turkce_promptlar.json'dan alinir.")
    parser.add_argument("--dongu",               type=str, default=None,
                        help="Dongu etiketi — checkpoint adini degistirir "
                             "(orn: dongu2 -> clap_bbc_dongu2_best.pt)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    cfg = FinetuneConfig(
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        lr               = args.lr,
        patience         = args.patience,
        sicaklik         = args.sicaklik,
        hard_neg_beta    = args.hard_neg_beta,
        n_frozen_audio   = args.n_frozen_audio,
        n_frozen_text    = args.n_frozen_text,
        tr_orani         = args.tr_orani,
        sadece_ingilizce = args.sadece_ingilizce,
        fp16             = args.fp16,
        ema_decay        = args.ema_decay,
        surum            = args.surum,
        hassasiyet       = args.hassasiyet or ("fp16" if args.surum == "v1" else "fp32"),
        secim_metrigi    = args.secim_metrigi or ("A2T_R@1" if args.surum == "v1" else "zs_ort"),
        parcali_okuma    = (args.surum == "v2"),
    )
    if args.duman:
        cfg.epochs = args.duman_epoch
        cfg.log_adim = 5

    # ── Cikti yollari (surume gore; v1 dosyalarinin uzerine yazilmaz) ─────
    global RESULT_DIR, CHECKPT_DIR
    RESULT_DIR = ROOT / "results" / cfg.surum
    if args.duman:
        RESULT_DIR  = RESULT_DIR / "_duman"
        CHECKPT_DIR = CHECKPT_DIR / "_duman"

    log.info("=" * 60)
    log.info("  CLAP Fine-Tuning — BBC Ses Arsivi")
    log.info(f"  Cihaz   : {DEVICE}")
    log.info(f"  Epochs  : {cfg.epochs}  Batch: {cfg.batch_size}  LR: {cfg.lr}")
    log.info(f"  Tau     : {cfg.sicaklik}  Hard-neg-beta: {cfg.hard_neg_beta}")
    log.info(f"  Donuk   : audio={cfg.n_frozen_audio} blok, text={cfg.n_frozen_text} katman")
    log.info(f"  TR/EN   : {cfg.tr_orani:.0%} Turkce" +
             (" (ABLASYON: sadece EN)" if cfg.sadece_ingilizce else ""))
    log.info(f"  Surum   : {cfg.surum}  Hassasiyet: {cfg.hassasiyet}  "
             f"Secim: {cfg.secim_metrigi}  Parcali okuma: {cfg.parcali_okuma}"
             + ("  [DUMAN TESTI]" if args.duman else ""))
    log.info("=" * 60)

    if args.kuru_calistirma:
        kuru_calistirma(cfg)
        return

    # ── Checkpoint adini belirle ──────────────────────────────────────────
    onek = "clap_bbc" if cfg.surum == "v1" else "clap_v2"
    ckpt_en_iyi_adi = f"{onek}_best.pt"
    if args.dongu:
        ckpt_en_iyi_adi = f"{onek}_{args.dongu}_best.pt"
    log.info(f"  Checkpoint: {CHECKPT_DIR / ckpt_en_iyi_adi}")

    # ── Veri ─────────────────────────────────────────────────────────────
    log.info(f"\nVeri yukleniyor: {PARQUET}")
    df = pd.read_parquet(PARQUET)

    # Sentetik veri ekle (Dongu 2+)
    if args.sentetik_veri:
        import json as _json
        synt_dir = ROOT / args.sentetik_veri
        promptlar_yol = ROOT / "data" / "processed" / "turkce_promptlar.json"
        with open(promptlar_yol, encoding="utf-8") as f:
            promptlar_veri = _json.load(f)

        synt_satirlar = []
        for sinif_dir in synt_dir.iterdir():
            if not sinif_dir.is_dir():
                continue
            sinif = sinif_dir.name
            if sinif not in promptlar_veri:
                continue
            promptlar_en = promptlar_veri[sinif].get("ingilizce", [])
            promptlar_tr = promptlar_veri[sinif].get("turkce", promptlar_en)
            wavlar = [w for w in sinif_dir.glob("*.wav") if w.stat().st_size > 1000]
            for i, wav in enumerate(wavlar):
                synt_satirlar.append({
                    "dosya":     str(wav.relative_to(ROOT)),
                    "sinif":     sinif,
                    "sinif_idx": df[df["sinif"] == sinif]["sinif_idx"].iloc[0]
                                 if sinif in df["sinif"].values else 0,
                    "n_segs":    1,
                    "prompt_en": promptlar_en[i % len(promptlar_en)] if promptlar_en else "",
                    "prompt_tr": promptlar_tr[i % len(promptlar_tr)] if promptlar_tr else "",
                })
        if synt_satirlar:
            df_synt = pd.DataFrame(synt_satirlar)
            df = pd.concat([df, df_synt], ignore_index=True)
            log.info(f"  Sentetik {len(df_synt)} satir eklendi ({synt_dir.name})")
        else:
            log.warning(f"  Sentetik dizinde gecerli WAV bulunamadi: {synt_dir}")

    denetim = None
    if cfg.surum == "v2":
        df_tr, df_vl, df_te, denetim = veri_bol_v2(df, cfg.val_orani, cfg.test_orani, cfg.seed)
        log.info(f"  v2 bolme: CLAP yalnizca siniflandirici egitim bolumunu goruyor "
                 f"(ortusme kontrolu gecti): {denetim['clap_dosya']}")
    else:
        df_tr, df_vl, df_te = veri_bol(df, cfg.val_orani, cfg.test_orani, cfg.seed)
    if args.duman:
        kucuk = lambda d, n: (d.groupby("sinif_idx", group_keys=False)
                              .apply(lambda g: g.sample(min(len(g), n), random_state=cfg.seed))
                              .reset_index(drop=True))
        df_tr, df_vl, df_te = kucuk(df_tr, 4), kucuk(df_vl, 2), kucuk(df_te, 2)
    log.info(f"  Train={len(df_tr)}  Val={len(df_vl)}  Test={len(df_te)}")

    ds_tr = BBCSesMetinVeriSeti(
        df_tr, ROOT,
        tr_orani=cfg.tr_orani,
        sadece_ingilizce=cfg.sadece_ingilizce,
        augment=True,
        parcali_okuma=cfg.parcali_okuma,
    )
    ds_vl = BBCSesMetinVeriSeti(
        df_vl, ROOT,
        tr_orani=0.5,   # val'da her iki dili de degerlendir
        augment=False,
        parcali_okuma=cfg.parcali_okuma,
    )
    ds_te = BBCSesMetinVeriSeti(
        df_te, ROOT,
        tr_orani=0.5,
        augment=False,
        parcali_okuma=cfg.parcali_okuma,
    )

    sampler = sinif_dengeli_sampler(df_tr)
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    dl_vl = DataLoader(
        ds_vl, batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )
    dl_te = DataLoader(
        ds_te, batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────
    model    = CLAPFinetune(cfg).to(DEVICE)
    kayip_fn = NTXentHardNeg(cfg.sicaklik, cfg.hard_neg_beta)
    ema      = EMA(model, decay=cfg.ema_decay)
    # Hazir (ince-ayarsiz) agirliklarin kopyasi: sonda ayni test bolumunde referans
    # degerlendirme icin (8,5 GB'lik checkpoint'i ikinci kez yuklememek icin)
    hazir_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ── Resume ───────────────────────────────────────────────────────────
    baslangic_epoch = 1
    resume_metrik   = -1.0
    durum           = None
    # Once "son durum" checkpoint'ini dene, yoksa "en iyi"yi kullan
    ckpt_son_adi    = ckpt_en_iyi_adi.replace(".pt", "_son.pt")
    ckpt_yolu_son   = CHECKPT_DIR / ckpt_son_adi
    ckpt_yolu_resume = ckpt_yolu_son if ckpt_yolu_son.exists() else CHECKPT_DIR / ckpt_en_iyi_adi
    if args.resume and ckpt_yolu_resume.exists():
        ck = torch.load(ckpt_yolu_resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state"])
        ema.shadow = ck["ema_shadow"]
        tamamlanan_epoch = ck.get("epoch", 0)
        baslangic_epoch  = tamamlanan_epoch + 1
        resume_metrik    = ck.get("val_metrik", {}).get(cfg.secim_metrigi, -1.0)
        if "optimizer" in ck:
            durum = ck          # v2 tam durum: optimizer, LR takvimi, en iyi metrik, RNG, gecmis
        else:
            log.warning("Eski bicim checkpoint: yalniz agirliklar yuklendi (optimizer/LR sifirdan).")
        log.info(f"Resume: {ckpt_yolu_resume.name} yuklendi (epoch {tamamlanan_epoch}), "
                 f"epoch {baslangic_epoch}'den devam")
    elif args.resume:
        log.warning("--resume verildi ama checkpoint bulunamadi, sifirdan basliyor.")

    # ── Egitim ───────────────────────────────────────────────────────────
    sonuc = egitim_dongusu(model, kayip_fn, dl_tr, dl_vl, cfg, ema,
                           baslangic_epoch=baslangic_epoch,
                           resume_metrik=resume_metrik,
                           ckpt_adi=ckpt_en_iyi_adi,
                           durum=durum,
                           kesinti_epoch=args.kesinti_epoch)
    egitim_suresi = sonuc["gecen_sn"] / 3600   # kesintiler dahil toplam egitim suresi

    # ── Test Degerlendirmesi (EMA agirliklarla) ───────────────────────────
    log.info("\nTest degerlendirmesi (EMA agirliklar)...")
    ckpt = torch.load(sonuc["ckpt"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_m = val_degerlendirme(model, dl_te, kayip_fn, cfg.hassasiyet)
    log.info("Test metrikleri:")
    for k, v in test_m.items():
        log.info(f"  {k}: {v}")

    # Referans: ince-ayarsiz (hazir) CLAP'in ayni test bolumundeki degerleri
    log.info("\nReferans: hazir CLAP (ince-ayarsiz) ayni test bolumunde...")
    model.load_state_dict(hazir_state)
    hazir_m = val_degerlendirme(model, dl_te, kayip_fn, cfg.hassasiyet)
    for k, v in hazir_m.items():
        log.info(f"  {k}: {v}")

    # ── Kaydet ───────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tarih_str = datetime.now().strftime("%Y%m%d_%H%M")
    etiket = f"_{args.dongu}" if args.dongu else ""
    cikti = RESULT_DIR / f"clap_finetune{etiket}_{tarih_str}.json"
    kayit = {
        "tarih":          tarih_str,
        "cfg":            asdict(cfg),
        "dongu":          args.dongu,
        "sentetik_veri":  args.sentetik_veri,
        "checkpoint":     sonuc["ckpt"],
        "egitim_saat":    round(egitim_suresi, 2),
        "en_iyi_epoch":   sonuc["en_iyi_epoch"],
        "secim_metrigi":  cfg.secim_metrigi,
        "val_en_iyi":     sonuc["en_iyi_metrik"],
        "test":           test_m,
        "test_hazir_clap": hazir_m,
        "tarih_detay":    sonuc["tarih"],
    }
    if denetim is not None:
        with open(RESULT_DIR / f"clap_bolme{etiket}_{tarih_str}.json", "w", encoding="utf-8") as f:
            json.dump(denetim, f, ensure_ascii=False, indent=1)
    with open(cikti, "w", encoding="utf-8") as f:
        json.dump(kayit, f, ensure_ascii=False, indent=2)

    log.info(f"\nSonuclar kaydedildi: {cikti}")
    log.info(f"Toplam sure: {egitim_suresi:.2f} saat")
    log.info(f"En iyi checkpoint: {sonuc['ckpt']}")


if __name__ == "__main__":
    main()
