"""
Fine-tuned CLAP agirliklarini AudioLDM2 checkpoint'ine entegre et.

Akis:
  1. AudioLDM2 orijinal checkpoint'ini yukle
  2. clap_bbc_best.pt'den fine-tuned CLAP agirliklarini al
  3. AudioLDM2 state_dict'teki 'clap.model.*' keylerini guncelle
  4. Yeni checkpoint'i kaydet → audioldm2_clap_finetuned.pth

Kullanim:
    python scripts/09_integrate_clap.py
    python scripts/09_integrate_clap.py --kuru  # sadece istatistik, kaydetme
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLAP_CKPT   = ROOT / "checkpoints" / "clap_bbc_best.pt"
CIKTI_CKPT  = ROOT / "checkpoints" / "audioldm2_clap_finetuned.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kuru", action="store_true",
                        help="Kaydetmeden sadece key eslesmesini kontrol et")
    parser.add_argument("--clap_ckpt", type=str, default=None,
                        help="CLAP checkpoint yolu (varsayilan: clap_bbc_best.pt)")
    parser.add_argument("--cikti", type=str, default=None,
                        help="Cikti checkpoint adi (varsayilan: audioldm2_clap_finetuned.pth)")
    args = parser.parse_args()

    global CLAP_CKPT, CIKTI_CKPT
    if args.clap_ckpt:
        CLAP_CKPT = ROOT / "checkpoints" / args.clap_ckpt
    if args.cikti:
        CIKTI_CKPT = ROOT / "checkpoints" / args.cikti

    # ── AudioLDM2 orijinal checkpoint ────────────────────────────────────
    print("AudioLDM2 checkpoint indiriliyor / kontrol ediliyor...")
    from audioldm2 import download_checkpoint
    audioldm2_yol = download_checkpoint("audioldm2-full")
    print(f"  Kaynak: {audioldm2_yol}")

    tam_ckpt = torch.load(audioldm2_yol, map_location="cpu")
    sd = tam_ckpt["state_dict"]
    print(f"  Toplam key: {len(sd)}")

    clap_keyler = [k for k in sd if k.startswith("clap.model.")]
    print(f"  'clap.model.*' key sayisi: {len(clap_keyler)}")

    # ── Fine-tuned CLAP checkpoint ────────────────────────────────────────
    print(f"\nFine-tuned CLAP yukleniyor: {CLAP_CKPT}")
    clap_ckpt = torch.load(CLAP_CKPT, map_location="cpu")
    ft_epoch  = clap_ckpt.get("epoch", "?")
    ft_metrik = clap_ckpt.get("val_metrik", {})
    print(f"  Epoch: {ft_epoch}")
    print(f"  Val metrikleri: {ft_metrik}")

    # model_state: 'model.X' keyleri
    ft_sd = clap_ckpt["model_state"]
    print(f"  Fine-tuned key sayisi: {len(ft_sd)}")

    # ── Key esleme: 'model.X' → 'clap.model.X' ───────────────────────────
    # AudioLDM2'de 'clap.model.X', finetune modelinde 'model.X'
    guncellenen = 0
    eksik_ft    = []
    for k in clap_keyler:
        # 'clap.model.X' → 'model.X'
        ft_k = k[len("clap."):]   # 'clap.model.X' → 'model.X'
        if ft_k in ft_sd:
            if sd[k].shape == ft_sd[ft_k].shape:
                sd[k] = ft_sd[ft_k].clone()
                guncellenen += 1
            else:
                print(f"  [!] Shape uyumsuz: {k}  {sd[k].shape} vs {ft_sd[ft_k].shape}")
        else:
            eksik_ft.append(ft_k)

    print(f"\nGuncellenen key: {guncellenen} / {len(clap_keyler)}")
    if eksik_ft:
        print(f"Fine-tuned'da eksik {len(eksik_ft)} key (normal — logit_scale vb.):")
        for k in eksik_ft[:10]:
            print(f"  {k}")

    # logit_scale_a / logit_scale_t guncelle (clap. prefix'siz)
    for scale_k in ["clap.logit_scale_a", "clap.logit_scale_t"]:
        ft_scale_k = scale_k[len("clap."):]  # 'logit_scale_a'
        # EMA shadow'da bu keyler var
        ema = clap_ckpt.get("ema_shadow", {})
        ema_k = f"model.{ft_scale_k}"
        if ema_k in ema and scale_k in sd:
            sd[scale_k] = ema[ema_k].clone()
            print(f"  logit_scale guncellendi: {scale_k} = {ema[ema_k].item():.4f}")

    # position_ids buffer'lari kaldir — build_model strict yukleme yapiyor,
    # bu keyler modelde registered buffer olarak zaten olusturuluyor
    position_id_keyler = [k for k in sd if k.endswith("position_ids")]
    for k in position_id_keyler:
        del sd[k]
    if position_id_keyler:
        print(f"  position_ids keyleri kaldirildi: {position_id_keyler}")

    if args.kuru:
        print("\n--kuru modu: kaydetme yapilmadi.")
        return

    # ── Kaydet ───────────────────────────────────────────────────────────
    tam_ckpt["state_dict"] = sd
    CIKTI_CKPT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nYeni checkpoint kaydediliyor: {CIKTI_CKPT}")
    torch.save(tam_ckpt, CIKTI_CKPT)
    boyut_gb = CIKTI_CKPT.stat().st_size / 1e9
    print(f"Kaydedildi: {boyut_gb:.2f} GB")
    print("\nTamamlandi. Tur 2 ses uretimi icin:")
    print("  python scripts/05_generate_audio.py --checkpoint checkpoints/audioldm2_clap_finetuned.pth")


if __name__ == "__main__":
    main()
