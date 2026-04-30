"""
BBC dosya adlarini Turkce'ye cevir ve ses-metin eslestirme tablosu olustur.

1. data/raw/ altindaki tum dosya/klasor adlarini toplar
2. Temizler (hash kodlari, uzanti, alt cizgi duzenler)
3. Google Translate ile toplu cevirir (deep-translator)
4. data/processed/dosya_ceviri.json  -- EN->TR esleme
5. data/processed/ses_metin_eslestirme.parquet -- her segment icin prompt

Kullanim:
    python scripts/00c_translate_filenames.py
    python scripts/00c_translate_filenames.py --sadece_parquet  # ceviri zaten varsa
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from deep_translator import GoogleTranslator

RAW_DIR   = ROOT / "data" / "raw"
CEVIRI_JSON = ROOT / "data" / "processed" / "dosya_ceviri.json"
PARQUET_OUT = ROOT / "data" / "processed" / "ses_metin_eslestirme.parquet"

SINIF_SIRASI = [
    "acil_servis", "at_sesleri", "ayak_sesleri", "cocuk_bebek_sesleri",
    "diger_hayvanlar", "dini_mekan", "doga_atmosferi", "endustri_fabrika",
    "ev_ici_sesler", "gemi_tekne_sesleri", "hava_olaylari", "insaat_sesleri",
    "kalabalik_sesleri", "kapi_sesleri", "kara_tasitlari", "kopek_sesleri",
    "kus_sesleri", "ofis_iletisim", "saat_can_sesleri", "sehir_atmosferi",
    "silah_patlama", "spor_eglence", "su_sesleri", "tarim_ciftlik",
    "tren_sesleri", "ucak_sesleri",
]
SINIF_IDX = {s: i for i, s in enumerate(SINIF_SIRASI)}


# ── Dosya adi temizleme ───────────────────────────────────────────────────────
def temizle(ham: str) -> str:
    """Ham dosya/klasor adini okunabilir Ingilizce cumleye donusturur."""
    ad = ham
    # Hash kodlari: ..07037461.wav veya .07037444.wav
    ad = re.sub(r'\.{1,2}\d{7,9}\.wav$', '', ad, flags=re.IGNORECASE)
    ad = re.sub(r'\.wav$', '', ad, flags=re.IGNORECASE)
    # Bas/son nokta ve bosluk
    ad = ad.strip(' .')
    # BBC'nin "_" kullanimi: "British Rail_ Southern Region" -> "British Rail: Southern Region"
    ad = re.sub(r'_\s+', ': ', ad)
    ad = ad.replace('_S ', "'s ").replace('_s ', "'s ")
    # Birden fazla bosluk
    ad = re.sub(r'\s+', ' ', ad)
    return ad.strip()


# ── Toplu ceviri ──────────────────────────────────────────────────────────────
def toplu_cevir(isimler: list[str], toplu_boyut: int = 40,
                gecikme: float = 1.2) -> dict[str, str]:
    """
    Ingilizce isim listesini Turkce'ye cevir.
    Mevcut ceviriyi yuklerse atlar (kurtarma destekli).
    """
    # Mevcut kaydedilmis ceviriyi yukle
    if CEVIRI_JSON.exists():
        with open(CEVIRI_JSON, encoding="utf-8") as f:
            ceviri = json.load(f)
    else:
        ceviri = {}

    eksik = [n for n in isimler if n not in ceviri]
    if not eksik:
        print(f"  Tum {len(isimler)} ceviri zaten mevcut.")
        return ceviri

    print(f"  {len(ceviri)} mevcut, {len(eksik)} yeni cevirilecek "
          f"({len(eksik)//toplu_boyut + 1} toplu istek)")

    translator = GoogleTranslator(source="en", target="tr")

    for i in range(0, len(eksik), toplu_boyut):
        grup = eksik[i: i + toplu_boyut]
        try:
            # GoogleTranslator tek seferde liste alabiliyor
            sonuclar = translator.translate_batch(grup)
            for en, tr in zip(grup, sonuclar):
                ceviri[en] = tr if tr else en  # bos gelirse orijinali koru
            print(f"  [{i+len(grup)}/{len(eksik)}] cevirildi")
        except Exception as e:
            print(f"  HATA (grup {i}): {e} — tek tek denenecek")
            for en in grup:
                try:
                    tr = translator.translate(en)
                    ceviri[en] = tr if tr else en
                except Exception:
                    ceviri[en] = en  # hata olursa orijinal kalsin

        # Ara kaydet (cokme korumasi)
        with open(CEVIRI_JSON, "w", encoding="utf-8") as f:
            json.dump(ceviri, f, ensure_ascii=False, indent=2)

        time.sleep(gecikme)

    return ceviri


# ── Sinif promptlari yukle ────────────────────────────────────────────────────
def sinif_promptlari_yukle() -> dict:
    yol = ROOT / "data" / "processed" / "turkce_promptlar.json"
    with open(yol, encoding="utf-8") as f:
        return json.load(f)


# ── ses_metin_eslestirme.parquet olustur ──────────────────────────────────────
def eslestirme_olustur(ceviri: dict, sinif_promptlari: dict) -> pd.DataFrame:
    """
    veri_listesi.parquet'teki her satir icin prompt atar.

    Oncelik:
      1. Dosya adi temizlenip ceviride varsa: o Turkce aciklama
      2. Yoksa: sinif promptlarindan dongusel secim
    """
    veri = pd.read_parquet(ROOT / "data" / "processed" / "veri_listesi.parquet")

    prompt_tr_list = []
    prompt_en_list = []
    rng_sayac = {s: 0 for s in SINIF_SIRASI}

    for _, satir in veri.iterrows():
        sinif = satir["sinif"]
        dosya_adi = Path(satir["dosya"]).stem  # uzantisiz dosya adi
        temiz_ad  = temizle(dosya_adi)

        # 1. Dosya adi cevirisi varsa kullan
        if temiz_ad in ceviri:
            prompt_tr = ceviri[temiz_ad]
            prompt_en = temiz_ad
        else:
            # 2. Sinif duzeyinden dongusel sec
            prompts_tr = sinif_promptlari[sinif]["turkce"]
            prompts_en = sinif_promptlari[sinif]["ingilizce"]
            idx = rng_sayac[sinif] % len(prompts_tr)
            prompt_tr  = prompts_tr[idx]
            prompt_en  = prompts_en[idx]
            rng_sayac[sinif] += 1

        prompt_tr_list.append(prompt_tr)
        prompt_en_list.append(prompt_en)

    veri["prompt_tr"] = prompt_tr_list
    veri["prompt_en"] = prompt_en_list
    return veri[["dosya", "sinif", "sinif_idx", "n_segs", "prompt_tr", "prompt_en"]]


# ── Ana ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sadece_parquet", action="store_true",
                        help="Ceviri JSON hazirsa sadece parquet olustur")
    args = parser.parse_args()

    # 1. Tum dosya adlarini topla
    print("Dosya adlari toplanıyor...")
    ham_adlar = set()
    dosya_sinif: dict[str, str] = {}  # temiz_ad -> sinif

    for sinif_dir in sorted(RAW_DIR.iterdir()):
        if not sinif_dir.is_dir():
            continue
        sinif = sinif_dir.name
        if sinif not in SINIF_IDX:
            continue
        for dosya in sinif_dir.iterdir():
            temiz = temizle(dosya.stem if dosya.suffix else dosya.name)
            if temiz:
                ham_adlar.add(temiz)
                dosya_sinif[temiz] = sinif

    isimler = sorted(ham_adlar)
    print(f"  {len(isimler)} benzersiz dosya adi bulundu")

    # 2. Cevir
    if not args.sadece_parquet:
        print("\nCeviriler baslatiliyor (Google Translate)...")
        ceviri = toplu_cevir(isimler)
    else:
        if not CEVIRI_JSON.exists():
            print("HATA: dosya_ceviri.json bulunamadi, --sadece_parquet kullanilamaz.")
            sys.exit(1)
        with open(CEVIRI_JSON, encoding="utf-8") as f:
            ceviri = json.load(f)
        print(f"  Mevcut ceviri yuklendi: {len(ceviri)} girdi")

    print(f"\nCeviri tamamlandi: {len(ceviri)} girdi -> {CEVIRI_JSON}")

    # Ornek goster
    print("\nOrnek ceviriler:")
    for en, tr in list(ceviri.items())[:10]:
        print(f"  {en:<55} -> {tr}")

    # 3. ses_metin_eslestirme.parquet olustur
    print("\nses_metin_eslestirme.parquet olusturuluyor...")
    sinif_promptlari = sinif_promptlari_yukle()
    df = eslestirme_olustur(ceviri, sinif_promptlari)
    df.to_parquet(PARQUET_OUT, index=False)

    print(f"  {len(df):,} satir kaydedildi -> {PARQUET_OUT}")
    print(f"\nOrnek (ilk 5 satir):")
    print(df[["sinif", "prompt_tr", "prompt_en"]].head())

    # Kapsam istatistigi
    dosya_bazli = sum(1 for row in df.itertuples()
                      if Path(row.dosya).stem in ceviri or
                         temizle(Path(row.dosya).stem) in ceviri)
    print(f"\nDosya bazli prompt kapsami : {dosya_bazli:,} / {len(df):,} "
          f"({100*dosya_bazli/len(df):.1f}%)")
    print(f"Sinif bazli prompt (rotasyon): {len(df)-dosya_bazli:,} segment")


if __name__ == "__main__":
    main()
