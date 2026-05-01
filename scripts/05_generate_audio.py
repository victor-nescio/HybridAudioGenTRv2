"""
AudioLDM2 ile sentetik ses üretimi — Tur 1 (pretrained, ince ayar yok).

Her sınıf için İngilizce promptlardan N_KLIP ses klip üretir.
Çıktı: data/synthetic/<sinif_adi>/<sinif>_<prompt_idx>_<uuid>.wav

Kullanım:
    python scripts/05_generate_audio.py
    python scripts/05_generate_audio.py --n_klip 10 --siniflar insaat_sesleri dini_mekan
    python scripts/05_generate_audio.py --siniflar all --n_klip 50
"""

import argparse
import json
import sys
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from audioldm2 import build_model, text_to_audio

PROMPTLAR_YOL = ROOT / "data" / "processed" / "turkce_promptlar.json"
SENTETIK_DIR  = ROOT / "data" / "synthetic"

MODEL_ADI      = "audioldm2-full"   # audioldm2 paketi içindeki model adı
SURE_SN        = 5.0                # 5 saniyelik klip (eğitim segmentleriyle eşleşir)
ORNEKLEME_HZ   = 16000
SURE_URET_SN   = 10.0   # audioldm2 sadece 10s destekliyor
NEGATIF_PROMPT = "low quality, noise, distortion, silence, music"


def model_yukle(model_adi: str = MODEL_ADI):
    print(f"  AudioLDM2 modeli yükleniyor ({model_adi})...")
    model = build_model(model_name=model_adi)
    print("  Model hazır.")
    return model


def ses_uret(model, prompt: str, n: int, n_adim: int = 200,
             guidance_scale: float = 3.5) -> np.ndarray:
    """
    Tek prompt için n ses klip üretir.
    Döner: (n, örnekler) float32 numpy array
    """
    audios = text_to_audio(
        latent_diffusion=model,
        text=prompt,
        transcription="",
        seed=42,
        duration=SURE_URET_SN,   # audioldm2 sadece 10s destekliyor
        batchsize=n,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=n,
        ddim_steps=n_adim,
    )
    return audios  # (n, örnekler)


def sinif_uret(model, sinif: str, promptlar: list[str], n_klip: int,
               n_adim: int = 200) -> int:
    cikis_dir = SENTETIK_DIR / sinif
    cikis_dir.mkdir(parents=True, exist_ok=True)

    # Sadece 0'dan buyuk dosyalari say (yarim kalmis dosyalari atla)
    mevcut = [p for p in cikis_dir.glob("*.wav") if p.stat().st_size > 1000]
    print(f"\n[{sinif}]  hedef={n_klip}  mevcut={len(mevcut)}")

    if len(mevcut) >= n_klip:
        print(f"  Zaten {len(mevcut)} klip var, atlanıyor.")
        return len(mevcut)

    eksik = n_klip - len(mevcut)
    n_prompt = len(promptlar)

    # Her prompta eşit klip dağıt
    klip_sayac = {i: 0 for i in range(n_prompt)}
    for k in range(eksik):
        klip_sayac[k % n_prompt] += 1

    toplam_uretilen = 0
    for p_idx, prompt in enumerate(promptlar):
        gerekli = klip_sayac[p_idx]
        if gerekli == 0:
            continue

        print(f"  [{p_idx+1}/{n_prompt}] '{prompt[:60]}' -> {gerekli} klip")

        for k in range(gerekli):
            # Her klip 1'er uretilir — RTX 3060 12GB icin batchsize>1 OOM veriyor
            torch.cuda.empty_cache()
            try:
                audios = ses_uret(model, prompt, 1, n_adim)
            except Exception as e:
                print(f"  HATA (klip {k+1}/{gerekli}): {e}")
                continue

            for audio in audios:
                # audioldm2 cikisi: (1, 1, ornekler) ya da (1, ornekler) -> (ornekler,)
                audio = np.squeeze(audio)
                audio = np.clip(audio, -1.0, 1.0)
                # 10s -> 5s: egitim segmentleriyle eslessin
                hedef_ornek = int(SURE_SN * ORNEKLEME_HZ)
                if audio.ndim == 1 and len(audio) > hedef_ornek:
                    audio = audio[:hedef_ornek]
                isim = f"{sinif}_{p_idx}_{uuid.uuid4().hex[:8]}.wav"
                sf.write(cikis_dir / isim, audio, samplerate=ORNEKLEME_HZ)
                toplam_uretilen += 1
                print(f"    [{toplam_uretilen}] kaydedildi: {isim}")

    print(f"  {toplam_uretilen} yeni klip uretildi -> {cikis_dir}")
    return len(mevcut) + toplam_uretilen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_klip",   type=int, default=50,
                        help="Her sinif icin uretilecek klip sayisi (varsayilan 50)")
    parser.add_argument("--siniflar", nargs="+", default=None,
                        help="Uretilecek siniflar ('all' ya da boslukla ayrilmis isimler). "
                             "Varsayilan: F1<0.5 olan zayif siniflar")
    parser.add_argument("--adimlar",  type=int, default=200,
                        help="Diffusion adim sayisi (varsayilan 200)")
    parser.add_argument("--model",    type=str, default=MODEL_ADI,
                        help=f"AudioLDM2 model adi (varsayilan: {MODEL_ADI})")
    args = parser.parse_args()

    with open(PROMPTLAR_YOL, encoding="utf-8") as f:
        promptlar_veri = json.load(f)

    # Varsayılan: F1 < 0.5 olan zayıf sınıflar (fusion sonuçlarına göre)
    ZAYIF_SINIFLAR = ["insaat_sesleri", "dini_mekan", "acil_servis", "diger_hayvanlar"]

    if args.siniflar is None:
        hedef_siniflar = ZAYIF_SINIFLAR
    elif args.siniflar == ["all"]:
        hedef_siniflar = list(promptlar_veri.keys())
    else:
        hedef_siniflar = args.siniflar

    gecersiz = [s for s in hedef_siniflar if s not in promptlar_veri]
    if gecersiz:
        print(f"HATA: Bilinmeyen siniflar: {gecersiz}")
        print(f"Gecerli siniflar: {list(promptlar_veri.keys())}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  AudioLDM2 Ses Uretimi - Tur 1 (Pretrained)")
    print(f"  Cihaz      : {device}")
    print(f"  Model      : {args.model}")
    print(f"  Siniflar   : {hedef_siniflar}")
    print(f"  Klip/sinif : {args.n_klip}")
    print(f"  Sure       : {SURE_SN}s  |  Adim: {args.adimlar}")
    print("=" * 60)

    model = model_yukle(args.model)

    ozet = {}
    for sinif in hedef_siniflar:
        promptlar_en = promptlar_veri[sinif]["ingilizce"]
        n = sinif_uret(model, sinif, promptlar_en, args.n_klip, args.adimlar)
        ozet[sinif] = n

    print("\n" + "=" * 60)
    print("  Ozet:")
    for sinif, n in ozet.items():
        print(f"    {sinif:<30} {n} klip")
    print("=" * 60)

    ozet_yol = ROOT / "results" / "sentetik_uretim_ozet.json"
    ozet_yol.parent.mkdir(exist_ok=True)
    with open(ozet_yol, "w", encoding="utf-8") as f:
        json.dump({"tur": 1, "model": args.model, "siniflar": ozet,
                   "n_klip_hedef": args.n_klip}, f, ensure_ascii=False, indent=2)
    print(f"\nOzet kaydedildi: {ozet_yol}")


if __name__ == "__main__":
    main()
