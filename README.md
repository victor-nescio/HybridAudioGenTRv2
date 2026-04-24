# HybridAudioGenTR v2

Hibrit ses sınıflandırma ve üretim sistemi. BBC Sound Effects arşivi üzerinde
26 Türkçe sınıf için LGBM → CNN/ResNet18 → AST sınıflandırma ve AudioLDM üretim pipeline'ı.

## Klasör Yapısı

```
sounds/          Ham BBC ses arşivi (git'e dahil değil)
data/
  raw/           26 sınıfa göre düzenlenmiş kaynak klasörler
  processed/     Özellik çıkarımı sonrası .parquet / .npz
src/
  data/
    labels.py        26 sınıf tanımı + eşleşme kuralları
    preprocessing.py Ses ön işleme (DC, pre-emphasis, trim, normalize)
    features.py      MFCC/log-mel/spektral özellik çıkarımı
  models/        LGBM, CNN, AST model tanımları
  training/      Eğitim döngüleri
  utils/         Yardımcı fonksiyonlar
scripts/
  00_organize_data.py  BBC klasörlerini data/raw/ altında sınıflara taşır
notebooks/       Keşifsel analizler
checkpoints/     Model ağırlıkları (git'e dahil değil)
outputs/         Üretilen sesler, raporlar (git'e dahil değil)
```

## Sınıflar (26)

| Klasör | Türkçe Ad |
|---|---|
| ucak_sesleri | Uçak Sesleri |
| tren_sesleri | Tren Sesleri |
| kara_tasitlari | Kara Taşıtları |
| gemi_tekne_sesleri | Gemi ve Tekne Sesleri |
| at_sesleri | At Sesleri |
| kopek_sesleri | Köpek Sesleri |
| kus_sesleri | Kuş Sesleri |
| diger_hayvanlar | Diğer Hayvan Sesleri |
| hava_olaylari | Hava Olayları |
| su_sesleri | Su Sesleri |
| doga_atmosferi | Doğa Atmosferi |
| sehir_atmosferi | Şehir Atmosferi |
| kalabalik_sesleri | Kalabalık Sesleri |
| cocuk_bebek_sesleri | Çocuk ve Bebek Sesleri |
| silah_patlama | Silah ve Patlama |
| acil_servis | Acil Servis Sesleri |
| endustri_fabrika | Endüstri ve Fabrika |
| insaat_sesleri | İnşaat Sesleri |
| ev_ici_sesler | Ev İçi Sesler |
| kapi_sesleri | Kapı Sesleri |
| saat_can_sesleri | Saat ve Çan Sesleri |
| ayak_sesleri | Ayak Sesleri |
| ofis_iletisim | Ofis ve İletişim |
| tarim_ciftlik | Tarım ve Çiftlik |
| spor_eglence | Spor ve Eğlence |
| dini_mekan | Dini Mekan Sesleri |

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Veri Organizasyonu

```bash
# Kuru çalışma (sadece rapor):
python scripts/00_organize_data.py

# Gerçek taşıma:
python scripts/00_organize_data.py --execute
```

## Pipeline (Planlanan)

1. `00_organize_data.py` — BBC arşivini 26 sınıfa taşı
2. `01_extract_features.py` — MFCC + spektral özellik çıkar → data/processed/
3. `02_train_lgbm.py` — Baseline LGBM sınıflandırıcı
4. `03_train_cnn.py` — CNN/ResNet18 log-mel üzerinde
5. `04_finetune_ast.py` — Audio Spectrogram Transformer fine-tune
6. `05_finetune_audioldm.py` — AudioLDM üretim modeli
7. `06_evaluate.py` — Üretilen sesleri sınıflandırıcıyla değerlendir
