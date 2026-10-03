# ============================================================
#  Tum turlar (1-4) icin sinifландirma: baseline vs augmented
#  (CNN + LightGBM), YENI cesitli kliplerle.
#
#  #4 cozumu: TEK tutarli baseline. Tur1 tam modda baseline'i
#  egitir (cnn_baseline_iter_best.pt); Tur2-4 --sadece_aug ile
#  AYNI baseline'i yukler. Boylece 4 tur da ayni baseline'a karsi.
#
#  Kullanim:
#     powershell -ExecutionPolicy Bypass -File .\siniflandirma_tum.ps1
#
#  Tahmini sure: ~12-16 saat (5 CNN egitimi x ~2-3s + LGBM).
#  Her tur ayri log: outputs\sinif_turN.log
#  NOT: Tur mid-resume DEGIL; kesilirse o tur bastan. Biten
#  turlarin JSON'lari korunur (results/v1/iteratif_loop_*.json).
# ============================================================

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))  # repo koku
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$bs = 64   # batch (orijinal CNN ile uyumlu, hizli)
$baslangic = Get-Date

# Tur 1 — TAM mod: baseline (cnn_baseline_iter_best.pt) + augmented egitir
Write-Host "============================================================"
Write-Host "  TUR 1 (tam: baseline + augmented)  $(Get-Date -Format HH:mm:ss)"
Write-Host "============================================================"
& $py scripts/07_iterative_loop.py --tur_etiketi 1 --batch_size $bs 2>&1 | Tee-Object -FilePath "outputs\sinif_tur1.log"

# Tur 2-4 — --sadece_aug: AYNI baseline'i yukle, sadece augmented egit
foreach ($t in @("tur2","tur3","tur4")) {
    Write-Host "============================================================"
    Write-Host "  $t (--sadece_aug: paylasilan baseline + augmented)  $(Get-Date -Format HH:mm:ss)"
    Write-Host "============================================================"
    & $py scripts/07_iterative_loop.py --sentetik_dir "data/synthetic_$t" --tur_etiketi $t --sadece_aug --batch_size $bs 2>&1 | Tee-Object -FilePath "outputs\sinif_$t.log"
}

$sure = (Get-Date) - $baslangic
Write-Host ""
Write-Host "=== TUM SINIFLANDIRMA TAMAMLANDI - sure: $([int]$sure.TotalHours) saat $($sure.Minutes) dk ==="
Write-Host "Sonraki adim: McNemar testleri (12_mcnemar_test.py vb.)"
