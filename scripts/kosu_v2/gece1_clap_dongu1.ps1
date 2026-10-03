# ============================================================
#  GECE 1 (v2) - CLAP Dongu 1, sizintisiz bolme
#
#  CLAP yalnizca siniflandiricinin EGITIM bolumundeki kayitlarla
#  ince-ayarlanir (val/test ile ortusme kodda assert ile engellenir).
#  Ciktilar:
#    checkpoints/clap_v2_dongu1_best.pt (+ _son.pt, resume icin)
#    results/v2/clap_finetune_dongu1_<tarih>.json   (metrikler + hazir CLAP referansi)
#    results/v2/clap_bolme_dongu1_<tarih>.json      (bolme denetimi)
#    outputs/v2_loglar/clap_dongu1.log              (UTF-8 log)
#    outputs/v2_loglar/clap_dongu1_DONE.txt | _HATA.txt
#
#  Kullanim (herhangi bir klasorden):
#    powershell -ExecutionPolicy Bypass -File scripts\kosu_v2\gece1_clap_dongu1.ps1
#    ... -Devam      # kesintiden sonra son epoch'tan devam (--resume)
# ============================================================
param([switch]$Devam)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$EPOCHS     = 10
$HASSASIYET = "fp32"

$ek = ""
if ($Devam) { $ek = "--resume" }
"BASLADI $(Get-Date -Format 'yyyy-MM-dd HH:mm')  epochs=$EPOCHS hassasiyet=$HASSASIYET $ek" |
    Out-File "$logdir\clap_dongu1_baslangic.txt" -Encoding utf8
Write-Host "=== CLAP v2 Dongu 1 basliyor $(Get-Date -Format HH:mm) - log: $logdir\clap_dongu1.log ==="

# cmd /c: log UTF-8 kalir, stderr PowerShell hata kaydina sarilmaz
cmd /c "`"$py`" -u scripts\08_clap_finetune.py --surum v2 --dongu dongu1 --epochs $EPOCHS --patience 3 --hassasiyet $HASSASIYET $ek >> $logdir\clap_dongu1.log 2>&1"
$kod = $LASTEXITCODE

if ($kod -eq 0) {
    "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\clap_dongu1_DONE.txt" -Encoding utf8
    Write-Host "=== BITTI $(Get-Date -Format HH:mm) ==="
} else {
    "HATA kod=$kod $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\clap_dongu1_HATA.txt" -Encoding utf8
    Write-Host "=== HATA (kod=$kod) - log'a bakin; devam icin -Devam ile yeniden calistirin ==="
}
