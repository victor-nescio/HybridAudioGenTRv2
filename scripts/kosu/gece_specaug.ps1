# ============================================================
#  GECE SURUCUSU 3 — M2: SpecAugment temeli (hakem kontrolu)
#
#  Soru: tutarli siniflardaki kazanim generative'e mi ozgu,
#  yoksa ucuz SpecAugment de ayni kazanimi verir mi?
#
#  Tek kosu, --spec_augment ile (--sadece_aug YOK) iki CNN egitir:
#    baseline_specaug  = gercek veri + SpecAugment (sentetik YOK)
#                        -> "SpecAugment-only" temeli
#    augmented_specaug = gercek + sentetik(H1) + SpecAugment
#                        -> "SpecAugment + generative" (istifleme)
#  Kanonik cnn_baseline_iter_best.pt EZILMEZ (ayri tag).
#
#  Cikti: results/iteratif_loop_specaug_*.json
#    .cnn.baseline  = SpecAugment-only sinif F1
#    .cnn.augmented = stacked sinif F1
#  Karsilastirma: no-aug baseline (0.6716) vs SpecAugment-only vs
#                 generative-only (mevcut coh kosulari) vs stacked.
#
#  Kullanim: powershell -ExecutionPolicy Bypass -File .\gece_specaug.ps1
#  Tahmini: ~7 saat (2 CNN egitimi)
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"

"BASLADI $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "outputs\specaug_baslangic.txt" -Encoding utf8
Write-Host "=== M2 SpecAugment temeli BASLADI $(Get-Date -Format HH:mm:ss) ==="

& $py scripts/07_iterative_loop.py `
    --siniflar kopek_sesleri kus_sesleri hava_olaylari `
    --sentetik_dir data/synthetic_turcoh `
    --spec_augment --sadece_cnn `
    --tur_etiketi specaug 2>&1 | Tee-Object -FilePath "outputs\sinif_specaug.log"

Write-Host "=== M2 BITTI $(Get-Date -Format HH:mm:ss) ==="
"DONE specaug $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "outputs\specaug_done.txt" -Encoding utf8
