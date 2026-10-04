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
# NOT: Bu negatif prompt hicbir zaman text_to_audio'ya verilmedi (v1 ve v2); uretim standart
# siniflandiricisiz rehberlikle (bos kosulsuz girdi, guidance 3.5) yapilir. v1 makalesindeki
# Tablo II bu acidan hataliydi; sabit yalnizca kayit amaciyla tutuluyor.
NEGATIF_PROMPT = "low quality, noise, distortion, silence, music"   # KULLANILMIYOR


def model_yukle(model_adi: str = MODEL_ADI, ckpt_yolu: str = None):
    print(f"  AudioLDM2 modeli yükleniyor ({model_adi})...")

    # --- transformers surum uyumsuzlugu yamasi ---
    # Yeni transformers surumlerinde 'position_ids' artik kalici (persistent)
    # buffer degil; ancak checkpoint'lerde mevcut. strict=True yukleme
    # "Unexpected key(s): ...position_ids" hatasi veriyor. Asagidaki gecici
    # yama yalnizca model insasi sirasinda bu fazla anahtarlari eler.
    _orig_lsd = torch.nn.Module.load_state_dict

    def _yamali_lsd(self, state_dict, strict=True, **kw):
        sd = {k: v for k, v in state_dict.items()
              if not k.endswith("position_ids")}
        return _orig_lsd(self, sd, strict=False, **kw)

    torch.nn.Module.load_state_dict = _yamali_lsd
    try:
        if ckpt_yolu:
            print(f"  Checkpoint: {ckpt_yolu}")
            model = build_model(ckpt_path=ckpt_yolu, model_name=model_adi)
        else:
            model = build_model(model_name=model_adi)
    finally:
        torch.nn.Module.load_state_dict = _orig_lsd   # global yamayi geri al

    print("  Model hazır.")
    return model


def ses_uret(model, prompt: str, n: int, n_adim: int = 200,
             guidance_scale: float = 3.5, seed: int = 42) -> np.ndarray:
    """
    Tek prompt için n ses klip üretir.
    `seed` her klip için DEĞİŞTİRİLEREK çağrılmalıdır; aksi hâlde aynı
    prompt+seed kombinasyonu birebir aynı klibi üretir (bkz. çeşitlilik
    hatası). Döner: (n, örnekler) float32 numpy array.
    """
    audios = text_to_audio(
        latent_diffusion=model,
        text=prompt,
        transcription="",
        seed=seed,
        duration=SURE_URET_SN,   # audioldm2 sadece 10s destekliyor
        batchsize=n,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=n,
        ddim_steps=n_adim,
    )
    return audios  # (n, örnekler)


def sinif_uret(model, sinif: str, promptlar: list[str], n_klip: int,
               n_adim: int = 200, base_seed: int = 1000) -> int:
    """
    Sınıf için `n_klip` adet ÇEŞİTLİ sentetik klip üretir.

    Her klip global indeksine bağlı BENZERSİZ bir seed ile üretilir
    (seed = base_seed + global_idx). Bu sayede:
      - aynı prompttan üretilen klipler farklı seed -> farklı ses (çeşitlilik),
      - kesinti sonrası resume güvenli (mevcut sayısından devam eder; seed
        çakışması olmaz),
      - tekrar üretilebilir (aynı indeks -> aynı seed).
    Promptlar global indeks üzerinden döngüsel atanır (idx % n_prompt),
    böylece promptlara eşit dağılım korunur.
    """
    cikis_dir = SENTETIK_DIR / sinif
    cikis_dir.mkdir(parents=True, exist_ok=True)

    # Sadece 0'dan buyuk dosyalari say (yarim kalmis dosyalari atla)
    mevcut = [p for p in cikis_dir.glob("*.wav") if p.stat().st_size > 1000]
    n_mevcut = len(mevcut)
    n_prompt = len(promptlar)
    print(f"\n[{sinif}]  hedef={n_klip}  mevcut={n_mevcut}  prompt={n_prompt}")

    if n_mevcut >= n_klip:
        print(f"  Zaten {n_mevcut} klip var, atlanıyor.")
        return n_mevcut

    toplam_uretilen = 0
    durdur = ROOT / "outputs" / "v2_loglar" / "kuyruk" / "DURDUR"
    for idx in range(n_mevcut, n_klip):          # global klip indeksi (resume güvenli)
        if durdur.exists():
            print(f"  [DURDUR] Istek alindi: {sinif} icin {n_mevcut + toplam_uretilen} klip hazir, "
                  f"cikiliyor (kod 4). Ayni komut kaldigi yerden devam eder.")
            sys.exit(4)
        p_idx   = idx % n_prompt                 # promptları döngüsel ata
        prompt  = promptlar[p_idx]
        seed    = base_seed + idx                # BENZERSİZ seed (her klip farklı)

        torch.cuda.empty_cache()
        try:
            audios = ses_uret(model, prompt, 1, n_adim, seed=seed)
        except Exception as e:
            print(f"  HATA (klip idx={idx}, seed={seed}): {e}")
            continue

        for audio in audios:
            # audioldm2 cikisi: (1, 1, ornekler) ya da (1, ornekler) -> (ornekler,)
            audio = np.squeeze(audio)
            audio = np.clip(audio, -1.0, 1.0)
            # 10s -> 5s: egitim segmentleriyle eslessin
            hedef_ornek = int(SURE_SN * ORNEKLEME_HZ)
            if audio.ndim == 1 and len(audio) > hedef_ornek:
                audio = audio[:hedef_ornek]
            # Dosya adına seed gömülür -> izlenebilirlik
            isim = f"{sinif}_{p_idx}_s{seed}_{uuid.uuid4().hex[:8]}.wav"
            sf.write(cikis_dir / isim, audio, samplerate=ORNEKLEME_HZ)
            toplam_uretilen += 1
            print(f"    [{n_mevcut + toplam_uretilen}/{n_klip}] seed={seed} "
                  f"prompt={p_idx} -> {isim}")

    print(f"  {toplam_uretilen} yeni klip uretildi -> {cikis_dir}")
    return n_mevcut + toplam_uretilen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_klip",   type=int, default=50,
                        help="Her sinif icin uretilecek klip sayisi (varsayilan 50)")
    parser.add_argument("--siniflar", nargs="+", default=None,
                        help="Uretilecek siniflar ('all' ya da boslukla ayrilmis isimler). "
                             "Varsayilan: F1<0.5 olan zayif siniflar")
    parser.add_argument("--adimlar",  type=int, default=200,
                        help="Diffusion adim sayisi (varsayilan 200)")
    parser.add_argument("--model",      type=str, default=MODEL_ADI,
                        help=f"AudioLDM2 model adi (varsayilan: {MODEL_ADI})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ozel checkpoint yolu (fine-tuned CLAP icin)")
    parser.add_argument("--tur",        type=str, default="1",
                        help="Uretim turu etiketi (cikti klasorunu etkiler, varsayilan: 1)")
    parser.add_argument("--dil",        type=str, default="ingilizce",
                        choices=["ingilizce", "turkce"],
                        help="Prompt dili (varsayilan: ingilizce)")
    parser.add_argument("--cikti_dir",  type=str, default=None,
                        help="(v2) Cikti klasoru, orn. data/synthetic_v2/C2p. Verilirse --tur klasor "
                             "kuralini gecersiz kilar; ozet results/v2/uretim/<klasor_adi>.json'a yazilir")
    parser.add_argument("--base_seed",  type=int, default=1000,
                        help="Klip seed tabani; her klip base_seed+global_idx "
                             "ile uretilir (turlar arasi farkli taban onerilir)")
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
    # Tur 2 icin ayri cikti klasoru
    global SENTETIK_DIR
    if args.cikti_dir:
        SENTETIK_DIR = ROOT / args.cikti_dir
        SENTETIK_DIR.mkdir(parents=True, exist_ok=True)
    elif args.tur != "1":
        SENTETIK_DIR = ROOT / "data" / f"synthetic_tur{args.tur}"
        SENTETIK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  AudioLDM2 Ses Uretimi - Tur {args.tur}")
    print(f"  Cihaz      : {device}")
    print(f"  Model      : {args.model}")
    if args.checkpoint:
        print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Siniflar   : {hedef_siniflar}")
    print(f"  Klip/sinif : {args.n_klip}")
    print(f"  Dil        : {args.dil}")
    print(f"  Sure       : {SURE_SN}s  |  Adim: {args.adimlar}")
    print(f"  Cikti      : {SENTETIK_DIR}")
    print("=" * 60)

    model = model_yukle(args.model, ckpt_yolu=args.checkpoint)

    ozet = {}
    for sinif in hedef_siniflar:
        promptlar = promptlar_veri[sinif].get(args.dil, promptlar_veri[sinif]["ingilizce"])
        if not promptlar:
            print(f"  UYARI: {sinif} icin {args.dil} prompt yok, ingilizce kullaniliyor.")
            promptlar = promptlar_veri[sinif]["ingilizce"]
        n = sinif_uret(model, sinif, promptlar, args.n_klip, args.adimlar,
                       base_seed=args.base_seed)
        ozet[sinif] = n

    print("\n" + "=" * 60)
    print("  Ozet:")
    for sinif, n in ozet.items():
        print(f"    {sinif:<30} {n} klip")
    print("=" * 60)

    ozet_yol = (ROOT / "results" / "v2" / "uretim" / f"{Path(args.cikti_dir).name}.json" if args.cikti_dir
                else ROOT / "results" / "v1" / f"sentetik_uretim_ozet_tur{args.tur}.json")
    ozet_yol.parent.mkdir(parents=True, exist_ok=True)
    with open(ozet_yol, "w", encoding="utf-8") as f:
        json.dump({"tur": args.tur, "model": args.model, "checkpoint": args.checkpoint,
                   "cikti_dir": args.cikti_dir, "dil": args.dil, "adimlar": args.adimlar,
                   "base_seed": args.base_seed, "siniflar": ozet,
                   "n_klip_hedef": args.n_klip}, f, ensure_ascii=False, indent=2)
    print(f"\nOzet kaydedildi: {ozet_yol}")


if __name__ == "__main__":
    main()
