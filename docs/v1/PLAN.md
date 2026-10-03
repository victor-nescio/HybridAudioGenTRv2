# Doktora Tezi — Geliştirme Planı

**Başlık:** Ses İşlemede Üretken Yapay Zeka Modelleri: Hibrit Yaklaşımlar ve Uygulamalar  
**Veri:** BBC Sound Effects Archive — 26 sınıf, 15.356 WAV, 312.725 segment (5 sn)  
**Ortam:** `audiogen` conda env · Python 3.10 · PyTorch cu118 · RTX 3060 11 GB

---

## Tez Katkıları (Hedef)

1. BBC ses arşivinde çok-model hibrit sınıflandırma (LGBM + CNN + BiLSTM + AST)
2. Üretken veri artırma → az örnekli sınıflarda F1 iyileştirmesi (Seviye-2 iteratif döngü)
3. Türkçe prompt ile AudioLDM fine-tuning — literatürde ilk Türkçe ses üretim çalışması
4. FAD ve downstream F1 arasındaki ilişkinin deneysel analizi

---

## Tamamlananlar

### Altyapı
- [x] `src/data/preprocessing.py` — WAV yükleme, yeniden örnekleme, mono dönüşüm
- [x] `src/data/features.py` — 467-dim özellik vektörü (MFCC-320, Mel-80, Spektral-28, Tonal-36, Zamansal-3); STFT önbelleği ile hız optimizasyonu
- [x] `src/data/labels.py` — 26 sınıf Türkçe taksonomi + AudioLDM İngilizce prompt eşleşmeleri
- [x] `src/data/dataset.py` — BBCSegmentDataset (mod: cnn/lstm/ast), dataloader_olustur, veri_bol

### Özellik Çıkarım
- [x] `scripts/01_extract_features.py` — joblib paralel, BATCH_SIZE=500, sürdürülebilir (crash-safe)
- [x] `data/processed/lgbm_ozellikler.parquet` — 312.725 segment, 467-dim, 0 hata

### Sınıflandırma — Geleneksel ML
- [x] `src/models/lgbm.py` — VARSAYILAN_PARAMS, lgbm_olustur, optuna_hedef, shap_grup_onemi, cv_degerlendir
- [x] `scripts/02_model_selection.py` — 8 algoritma karşılaştırması (LR, kNN, SVM, RF, ET, GB, XGBoost, LightGBM)
- [x] `scripts/02_train_lgbm.py` — Optuna HPO, 5-fold CV, SHAP görselleştirme (kazanan algoritma için)
- [x] `checkpoints/lgbm_model.pkl` — Test Macro F1: **0.5857**, CV: 0.5529 ± 0.0155
- [x] `notebooks/lgbm_baseline.ipynb` — Tam pipeline, tüm çıktılar gömülü

### Sınıflandırma — Derin Öğrenme (Modeller Hazır, Eğitim Bekliyor)
- [x] `src/models/cnn.py` — SpectrogramCNN (ResNet18 transfer), SimpleCNN (ablasyon)
- [x] `src/models/lstm.py` — BiLSTMClassifier (temporal attention), SimpleLSTM (ablasyon)
- [x] `scripts/03_train_cnn_lstm.py` — AMP, label smoothing, CosineAnnealingLR, early stopping
- [x] `notebooks/cnn_lstm.ipynb` — Eğitim döngüsü + görselleştirme notebook

---

## Sıradaki Adımlar

### Aşama 1 — Derin Öğrenme Modellerini Tamamla

#### 1.1 CNN + BiLSTM Eğitimi
```
python scripts/03_train_cnn_lstm.py --model cnn   --epochs 30
python scripts/03_train_cnn_lstm.py --model lstm  --epochs 30
```
- Çıktı: `models/cnn_best.pt`, `models/lstm_best.pt`
- Hedef: CNN F1 > 0.70, BiLSTM F1 > 0.65
- Tahmini süre: CNN ~35 dk, BiLSTM ~25 dk (RTX 3060)

#### 1.2 AST Modeli
**Dosya:** `src/models/ast_model.py`

HuggingFace'ten `MIT/ast-finetuned-audioset-10-10-0.4593` yüklenir, son katman 26 sınıfa adapte edilir. Dataset pipeline'ında `mod='ast'` zaten hazır (AudioSet normalizasyonu uygulanıyor).

```python
from transformers import ASTForAudioClassification
# Son katman: Linear(768, 26)
# Giriş: (B, 1, 128, 313) — mevcut dataset çıktısı
```

```
python scripts/03_train_cnn_lstm.py --model ast --epochs 20 --lr 1e-4
```
- Hedef: AST F1 > 0.75 (literatür: BBC benzeri veride ~0.80)
- Gereklilik: `pip install transformers`

#### 1.3 Late Fusion
**Dosya:** `scripts/04_fusion.py`

Tüm modellerin test kümesi softmax çıktılarını birleştirir:

| Yöntem | Açıklama |
|--------|----------|
| Ağırlıklı ortalama | Val F1'e göre ağırlık |
| Öğrenilen fusion | LogisticRegression meta-learner |
| Oracle upper bound | Her örnek için en iyi modeli seç |

Hedef: Fusion F1 > 0.80

---

### Aşama 2 — Türkçe Etiketleme

**Amaç:** Mevcut BBC ses dosyalarına Türkçe metin açıklamaları ekleyerek AudioLDM'in Türkçe prompt anlayışını geliştirmek.

#### 2.1 Prompt Genişletme
**Dosya:** `data/processed/turkce_promptlar.json`

`src/data/labels.py`'deki her sınıf için mevcut tek İngilizce prompt → 5 Türkçe + 5 İngilizce varyasyon:

```json
{
  "kopek_sesleri": {
    "turkce": [
      "köpek havlaması",
      "büyük bir köpeğin öfkeli havlaması",
      "sokak köpeğinin uzaktan sesi",
      "köpek uluma ve havlama",
      "yavru köpek ciyaklaması"
    ],
    "ingilizce": [
      "dog barking loudly",
      "angry dog barking in yard",
      "distant dog bark at night",
      "dog howling and barking",
      "puppy whimpering and yelping"
    ]
  }
}
```

**Yapılacak:** 26 sınıfın her biri için bu listeyi elle yaz (yaklaşık 2-3 saat iş).

#### 2.2 Ses-Metin Eşleştirme Verisi Oluşturma
**Dosya:** `data/processed/ses_metin_eslestirme.parquet`

Her WAV segmentini otomatik olarak Türkçe prompt'lardan biriyle eşleştir:

```
dosya | sinif_idx | segment_no | prompt_tr | prompt_en
```

Bu veri hem AudioLDM fine-tuning hem CLAP fine-tuning için kullanılacak.

---

### Aşama 3 — Üretken Döngü (Seviye-2 İteratif)

#### 3.1 AudioLDM Kurulum ve İlk Üretim Deneyi
```
pip install audioldm2
```

**Dosya:** `scripts/05_generate_audio.py`

```
Girdi : Sınıf adı + Türkçe/İngilizce prompt listesi
Çıktı : data/synthetic/<sinif_adi>/<uuid>.wav
```

İlk deney: pretrained AudioLDM2 (fine-tuning yok), İngilizce promptlar, her sınıf için 50 klip.

#### 3.2 FAD Ölçümü
**Dosya:** `scripts/06_measure_fad.py`

```
pip install fadtk
```

Her sınıf için:
- Gerçek ses dağılımı (BBC test seti)
- Üretilen ses dağılımı (sentetik)
- FAD skoru hesapla (VGGish + CLAP encoder'larıyla karşılaştırmalı)

```
Çıktı: results/fad_scores.json
{
  "kopek_sesleri": {"fad_vggish": 12.3, "fad_clap": 8.7},
  "kalabalik_sesleri": {"fad_vggish": 18.1, "fad_clap": 14.2},
  ...
}
```

#### 3.3 İteratif Döngü — Seviye 2

```
TUR 1
├── CNN (mevcut) ile test setini tahmin et
├── Per-class F1 hesapla
├── F1 < 0.5 olan sınıfları tespit et  →  "zayıf sınıflar"
├── AudioLDM ile zayıf sınıflar için 100 klip/sınıf üret
├── FAD ile kaliteyi ölç
└── Gerçek + sentetik karışık dataset ile CNN'i yeniden eğit

TUR 2
├── Yeniden eğitilen CNN ile F1 ölç
├── Hâlâ zayıf olan sınıflar için fine-tuned AudioLDM kullan
│   (Türkçe promptlarla fine-tune edilmiş model)
└── Yeniden eğit → F1 değişimini raporla
```

**Dosya:** `scripts/07_iterative_loop.py`

Her turda kaydedilecekler:
- Hangi sınıflar zayıftı
- Kaç sentetik klip eklendi
- FAD skoru
- Yeni F1 (before/after tablosu)

#### 3.4 AudioLDM Türkçe Fine-tuning
**Dosya:** `scripts/08_finetune_audioldm.py`

`data/processed/ses_metin_eslestirme.parquet` → CLAP text encoder fine-tuning.

Adımlar:
1. CLAP text encoder'ı dondur, ses encoder'ı serbest bırak → 5 epoch
2. İkisini birden fine-tune et → 10 epoch
3. FAD karşılaştırması: pretrained vs fine-tuned

---

### Aşama 4 — Değerlendirme ve Raporlama

#### 4.1 Ana Karşılaştırma Tablosu
**Dosya:** `scripts/09_final_eval.py`

| Model | Test Macro F1 | Eğitim Verisi |
|-------|--------------|---------------|
| LGBM Baseline | 0.5857 | Gerçek |
| CNN (ResNet18) | TBD | Gerçek |
| BiLSTM + Attention | TBD | Gerçek |
| AST | TBD | Gerçek |
| Late Fusion | TBD | Gerçek |
| CNN + Sentetik (Tur 1) | TBD | Gerçek + Sentetik |
| CNN + Sentetik (Tur 2, TR fine-tune) | TBD | Gerçek + Sentetik (TR) |

#### 4.2 İstatistiksel Anlamlılık
Her model çifti için McNemar testi (segment düzeyinde).

#### 4.3 Görselleştirme
- Grad-CAM: CNN'in spektrogram üzerinde neye baktığı
- Attention ağırlıkları: BiLSTM'in hangi zaman dilimlerine yoğunlaştığı
- FAD vs F1 scatter plot: üretim kalitesi ile sınıflandırma iyileştirmesi ilişkisi

---

## Dosya Yapısı (Hedef)

```
Drthesiscode/
├── data/
│   ├── raw/                         BBC WAV dosyaları
│   ├── processed/
│   │   ├── lgbm_ozellikler.parquet  ✓ Hazır
│   │   ├── veri_listesi.parquet     ✓ Hazır
│   │   ├── turkce_promptlar.json    → Aşama 2
│   │   └── ses_metin_eslestirme.parquet → Aşama 2
│   └── synthetic/                   → Aşama 3
│       └── <sinif_adi>/
├── src/
│   ├── data/
│   │   ├── preprocessing.py         ✓
│   │   ├── features.py              ✓
│   │   ├── labels.py                ✓
│   │   └── dataset.py               ✓
│   └── models/
│       ├── lgbm.py                  ✓
│       ├── cnn.py                   ✓
│       ├── lstm.py                  ✓
│       └── ast_model.py             → Aşama 1.2
├── scripts/
│   ├── 01_extract_features.py       ✓
│   ├── 02_train_lgbm.py             ✓
│   ├── 03_train_cnn_lstm.py         ✓
│   ├── 04_fusion.py                 → Aşama 1.3
│   ├── 05_generate_audio.py         → Aşama 3.1
│   ├── 06_measure_fad.py            → Aşama 3.2
│   ├── 07_iterative_loop.py         → Aşama 3.3
│   ├── 08_finetune_audioldm.py      → Aşama 3.4
│   └── 09_final_eval.py             → Aşama 4
├── notebooks/
│   ├── lgbm_baseline.ipynb          ✓
│   ├── cnn_lstm.ipynb               ✓
│   ├── ast_fusion.ipynb             → Aşama 1.2-1.3
│   └── generative_loop.ipynb        → Aşama 3
├── models/                          Eğitilmiş ağırlıklar (.pt, .pkl)
├── checkpoints/                     ✓
├── results/                         JSON metrikler, görseller
└── PLAN.md                          Bu dosya
```

---

## Öncelik Sırası

```
[ ] 1. CNN + BiLSTM + AST eğitimi (gece)
[ ] 2. Klasik ML algoritma seçimi: python scripts/02_model_selection.py (gece)
[ ] 3. Kazanan algoritma → Optuna HPO: python scripts/02_train_lgbm.py
[ ] 4. Late fusion (scripts/04_fusion.py)
[ ] 4. Türkçe prompt listesi (elle yazım, data/processed/turkce_promptlar.json)
[ ] 5. AudioLDM kurulum + ilk üretim deneyi
[ ] 6. FAD ölçümü
[ ] 7. İteratif döngü (Tur 1)
[ ] 8. AudioLDM Türkçe fine-tuning
[ ] 9. İteratif döngü (Tur 2)
[ ] 10. Final değerlendirme + istatistiksel testler
```

---

## Notlar

- **Türkçe fine-tuning özgünlüğü:** CLAP ve AudioLDM tamamen İngilizce corpus ile eğitildi. Türkçe ses-metin fine-tuning literatürde neredeyse yok — bu tek başına bir konferans bildirisi.
- **FAD encoder seçimi:** VGGish (standart) + CLAP (modern) ile karşılaştırmalı ölç. Encoder seçiminin FAD skorunu nasıl etkilediği tezde tartışılacak.
- **Seviye-2 döngü sınırı:** Maksimum 3 tur. Her turda erken durdurma: F1 artışı < 0.01 ise dur.
- **Sentetik veri oranı:** Gerçek verinin max %20'si kadar sentetik ekle (aşırı artırma etkisini önlemek için).
