# ============================================================
#  ZINCIRLEME SURUCU — CLAP tutarlilik bitince M2'yi baslatir.
#
#  1) results\v1\sinif_tutarlilik_clap.json olusana kadar bekle
#     (CLAP embedding kosusunun bitis sinyali) + GPU bosalsin.
#  2) M2 SpecAugment temelini calistir (07_iterative_loop
#     --spec_augment, ~7 saat, 2 CNN: SpecAugment-only + stacked).
#  Kanonik baseline EZILMEZ (ayri tag).
#
#  Kullanim: powershell -ExecutionPolicy Bypass -File .\gece_clap_sonra_m2.ps1
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))  # repo koku
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"

# --- 1) CLAP bitisini bekle ---
Write-Host "=== CLAP tutarlilik bekleniyor... $(Get-Date -Format HH:mm:ss) ==="
$bekle = 0
while (-not (Test-Path "results\v1\sinif_tutarlilik_clap.json")) {
    Start-Sleep -Seconds 60
    $bekle += 1
    if ($bekle -ge 120) { Write-Host "UYARI: 2 saat beklendi, yine de devam."; break }
}
Write-Host "=== CLAP bitti (ya da zaman asimi). GPU bosalmasi icin 30sn bekle. $(Get-Date -Format HH:mm:ss) ==="
Start-Sleep -Seconds 30

# --- 2) M2 SpecAugment temeli ---
"BASLADI $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "outputs\specaug_baslangic.txt" -Encoding utf8
Write-Host "=== M2 SpecAugment BASLADI $(Get-Date -Format HH:mm:ss) ==="
& $py scripts/07_iterative_loop.py `
    --siniflar kopek_sesleri kus_sesleri hava_olaylari `
    --sentetik_dir data/synthetic_turcoh `
    --spec_augment --sadece_cnn `
    --tur_etiketi specaug 2>&1 | Tee-Object -FilePath "outputs\sinif_specaug.log"
Write-Host "=== M2 BITTI $(Get-Date -Format HH:mm:ss) ==="
"DONE specaug $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "outputs\specaug_done.txt" -Encoding utf8
