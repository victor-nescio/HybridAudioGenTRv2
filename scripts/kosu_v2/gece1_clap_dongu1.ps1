# ============================================================
#  GECE 1 (v2) - CLAP Dongu 1, sizintisiz bolme
#
#  CLAP yalnizca siniflandiricinin EGITIM bolumundeki kayitlarla
#  ince-ayarlanir (val/test ile ortusme kodda assert ile engellenir).
#
#  Kesintiye dayanikli:
#   - Kosu boyunca Windows uyku moduna gecmez (kalici ayar degismez).
#   - Her epoch sonunda tam durum atomik olarak kaydedilir.
#   - PC kapanir/kesilirse AYNI KOMUTU yeniden calistirin: yarim kalan
#     kosu son tamamlanan epoch'tan otomatik devam eder (en fazla ~1 epoch kaybi).
#   - Is tamamlandiysa yeniden baslatmaz (-Sifirdan ile zorlanabilir).
#
#  Ciktilar:
#    checkpoints/clap_v2_dongu1_best.pt (+ _son.pt)
#    results/v2/clap_finetune_dongu1_<tarih>.json   (metrikler + hazir CLAP referansi)
#    results/v2/clap_bolme_dongu1_<tarih>.json      (bolme denetimi)
#    outputs/v2_loglar/clap_dongu1.log              (UTF-8 log, eklemeli)
#    outputs/v2_loglar/clap_dongu1_DONE.txt | _HATA.txt
#
#  Kullanim (herhangi bir klasorden):
#    powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\gece1_clap_dongu1.ps1
# ============================================================
param([switch]$Sifirdan)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$EPOCHS     = 10
$HASSASIYET = "fp32"
$DONE       = "$logdir\clap_dongu1_DONE.txt"
$SON_CKPT   = "checkpoints\clap_v2_dongu1_best_son.pt"

if ((Test-Path $DONE) -and -not $Sifirdan) {
    Write-Host "Bu is zaten tamamlanmis ($DONE). Yeniden calistirmak icin -Sifirdan kullanin."
    exit 0
}
Remove-Item "$logdir\clap_dongu1_HATA.txt" -ErrorAction SilentlyContinue

$ek = ""
if ((Test-Path $SON_CKPT) -and -not $Sifirdan) {
    $ek = "--resume"
    Write-Host "Yarim kalmis kosu bulundu: son tamamlanan epoch'tan devam ediliyor."
}

# Uyku modunu kosu suresince engelle (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
Add-Type -Namespace KosuGuc -Name Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
[void][KosuGuc.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor [uint32]1)

try {
    "BASLADI $(Get-Date -Format 'yyyy-MM-dd HH:mm')  epochs=$EPOCHS hassasiyet=$HASSASIYET $ek" |
        Out-File "$logdir\clap_dongu1_baslangic.txt" -Append -Encoding utf8
    Write-Host "=== CLAP v2 Dongu 1 basliyor $(Get-Date -Format HH:mm) - log: $logdir\clap_dongu1.log ==="

    # cmd /c: log UTF-8 kalir, stderr PowerShell hata kaydina sarilmaz
    cmd /c "`"$py`" -u scripts\08_clap_finetune.py --surum v2 --dongu dongu1 --epochs $EPOCHS --patience 3 --hassasiyet $HASSASIYET $ek >> $logdir\clap_dongu1.log 2>&1"
    $kod = $LASTEXITCODE

    if ($kod -eq 0) {
        "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File $DONE -Encoding utf8
        Write-Host "=== BITTI $(Get-Date -Format HH:mm) ==="
    } else {
        "HATA kod=$kod $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\clap_dongu1_HATA.txt" -Encoding utf8
        Write-Host "=== HATA (kod=$kod) - log'a bakin; ayni komutla yeniden calistirinca kaldigi yerden devam eder ==="
    }
}
finally {
    [void][KosuGuc.Win32]::SetThreadExecutionState($ES_CONTINUOUS)   # uyku engelini kaldir
}
