# Yeniden Üretim Pipeline'ı — seed çeşitlilik düzeltmesi sonrası

> **v1 (ilk gönderim) belgesi.** Bu dosya IEEE Access ilk gönderiminin (etiket `v1-submission`) yeniden üretim akışını anlatır; sonuçlar `results/v1/`, koşu scriptleri `scripts/v1/kosu/` altındadır.

**Neden:** `05_generate_audio.py` her klibi `seed=42` ile üretiyordu → prompt başına ~10 klip birebir aynı. Düzeltme: her klip `base_seed + global_idx` ile benzersiz seed alır (resume güvenli). Smoke-test doğrulandı: aynı prompt, farklı seed → farklı klip ✅.

**Eski (kopya) klipler yedeklendi:** `data/_dup_backup_20260531/` (4 tur × 4 sınıf). Orijinal dizinler temizlendi; Tur1'de smoke-test'ten 6 gerçek-çeşitli klip var (koşum bunları 50'ye tamamlar).

---

## ADIM 1 — ÜRETİM (gece, ~14 saat) ✅ kod hazır

Proje kökünde:
```powershell
powershell -ExecutionPolicy Bypass -File .\yeniden_uret_tum.ps1
```
4 turu sırayla üretir, her biri `outputs\uretim_turN.log`'a yazar, sonunda çeşitlilik kontrolü yapar. **Resume güvenli** — kesilirse aynı komutu tekrar çalıştır, kaldığı yerden devam eder.

| Tur | Model | Dil | base_seed | Çıktı |
|---|---|---|---|---|
| 1 | pretrained | EN | 1000 | data/synthetic |
| 2 | Döngü1 FT | EN | 2000 | data/synthetic_tur2 |
| 3 | Döngü1 FT | TR | 3000 | data/synthetic_tur3 |
| 4 | Döngü2 FT | EN | 4000 | data/synthetic_tur4 |

Beklenen sonuç: her sınıf 50 dosya / **50 benzersiz** içerik.

---

## ADIM 2 — FAD ÖLÇÜMÜ (~2 saat) ⚠️ önce fadtk kur

```powershell
& "C:\Users\User\anaconda3\envs\audiogen\python.exe" -m pip install fadtk
```
Sonra (VGGish ilk çalıştırmada iner):
```powershell
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
& $py scripts/06_measure_fad.py                                            # Tur1 -> fad_sonuclar.json
& $py scripts/06_measure_fad.py --sentetik_dir data/synthetic_tur2 --tur tur2
& $py scripts/06_measure_fad.py --sentetik_dir data/synthetic_tur3 --tur tur3
& $py scripts/06_measure_fad.py --sentetik_dir data/synthetic_tur4 --tur 4
```
> Not: `--tur 4` çıktıyı `fad_sonuclar_4.json`'a yazar (mevcut dosya adıyla uyumlu). Bu adım **Bulgu A'yı da çözer** (dini Tur4 FAD artık gerçek kliplerden hesaplanır).

---

## ADIM 3 — SINIFLANDIRMA (en uzun adım, ~16 saat) — `07_iterative_loop.py`

Her tur baseline + augmented CNN/LGBM eğitir (CNN ~3-4s/tur):
```powershell
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
& $py scripts/07_iterative_loop.py --tur_etiketi 1                                  # data/synthetic
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_tur2 --tur_etiketi tur2
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_tur3 --tur_etiketi tur3
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_tur4 --tur_etiketi tur4
```
Çıktı: `results/v1/iteratif_loop_*.json` (baseline/augmented makro-F1 + sınıf F1).

---

## ADIM 4 — McNEMAR (dakikalar)
```powershell
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
& $py scripts/12_mcnemar_test.py            # CNN bsl->aug (Tur1-3 + Tur4) -> ayri dosyalara kaydet
& $py scripts/v1/12c_mcnemar_lgbm_tum_turler.py
& $py scripts/v1/12d_mcnemar_cnn_vs_lgbm.py
```
> Bu **Bulgu #5'i de çözer** (CNN Tur1-3 McNemar yeniden üretilip arşivlenir). `12_mcnemar_test.py` çıktısını turlara göre ayrı dosyaya kaydedecek şekilde gözden geçirelim (üzerine yazmasın).

---

## ADIM 5 — FİGÜRLER + MAKALELER (benim işim, ~3-4 saat)

Yeni JSON'lar hazır olunca:
1. `12_makale_figurleri.py` içindeki **sabit-kodlu değerleri** yeni JSON'lardan güncelle (FAD, delta, paradoks, scatter).
2. Her iki makaledeki tüm FAD / augmentasyon / McNemar / "200 klip" sayılarını yenile.
3. Makale 2'nin paradoks yorumunu yeni (çeşitli) verilere göre gözden geçir.
4. Derle + doğrula.

---

## TOPLAM TAKVİM
- **Gece:** Adım 1 (üretim, ~14s)
- **Yarın:** Adım 2 (FAD, ~2s) + Adım 3 (sınıflandırma, ~16s — gece de sürebilir)
- **Sonra:** Adım 4 (McNemar, dk) + Adım 5 (figür/makale, benim ~4s)

İlk gece sadece **Adım 1**'i başlat yeter. Adım 3 (sınıflandırma) tek başına ~16s olduğundan onu da ikinci gece arka planda çalıştırabiliriz.
