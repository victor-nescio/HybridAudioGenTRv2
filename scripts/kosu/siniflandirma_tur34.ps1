# ============================================================
#  KALAN TURLAR: sadece Tur3 + Tur4 (gece yeniden baslatma).
#  Tur1/Tur2 ZATEN BITTI (JSON'lari results/ altinda).
#  Paylasilan baseline (cnn_baseline_iter_best.pt, Tur1'den)
#  --sadece_aug ile yuklenir; preload hizlandirmasi aktif.
#
#  Kullanim (proje kokunde ya da herhangi bir yerden):
#     powershell -ExecutionPolicy Bypass -File .\siniflandirma_tur34.ps1
#
#  Tahmini sure: ~5-6 saat (2 tur x [preload'lu CNN ~1-1.5s + LGBM ~1.75s]).
#  ISTENILEN AN DURDURULABILIR: durdurulursa o tur bastan baslar;
#  biten turlarin JSON'lari korunur.
# ============================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$bs = 64
$baslangic = Get-Date

# Guvenlik: paylasilan baseline var mi?
if (-not (Test-Path "models\cnn_baseline_iter_best.pt")) {
    Write-Host "HATA: models\cnn_baseline_iter_best.pt yok! Tur1 baseline'i gerekli."
    exit 1
}
Write-Host "Paylasilan baseline mevcut. Tur3+Tur4 baslatiliyor..."

foreach ($t in @("tur3","tur4")) {
    Write-Host "============================================================"
    Write-Host "  $t (--sadece_aug: paylasilan baseline + augmented)  $(Get-Date -Format HH:mm:ss)"
    Write-Host "============================================================"
    & $py scripts/07_iterative_loop.py --sentetik_dir "data/synthetic_$t" --tur_etiketi $t --sadece_aug --batch_size $bs 2>&1 | Tee-Object -FilePath "outputs\sinif_$t.log"
    Write-Host "  $t tamamlandi: $(Get-Date -Format HH:mm:ss)"
}

$sure = (Get-Date) - $baslangic
Write-Host ""
Write-Host "=== TUR3+TUR4 TAMAMLANDI - sure: $([int]$sure.TotalHours) saat $($sure.Minutes) dk ==="
Write-Host "Sonraki adim: McNemar testleri"
