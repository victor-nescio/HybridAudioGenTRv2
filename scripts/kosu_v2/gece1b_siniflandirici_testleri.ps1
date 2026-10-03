# ============================================================
#  GECE 1b (v2) - siniflandirici dogrulama testleri
#
#  CLAP dongu 1 (gece1_clap_dongu1.ps1) bitene kadar bekler, sonra:
#   1) Devam testi (~1 sa): kesintisiz 2 epoch (_testA) ile
#      "1 epoch + kesinti + devam" (_testB) birebir ayni mi?
#   2) v1-v2 uzlastirma (~4 sa): yeni kod + seed 42 baseline,
#      v1 baseline'ini (test macro-F1 0.6716, epoch gecmisi) birebir uretmeli.
#      Bu kosu ayni zamanda v2'nin seed-42 baseline sonucudur.
#
#  Kesintiye dayanikli: ayni komut yeniden calistirilinca tamamlanan
#  adimlari atlar, yarim kalan siniflandirici kosusuna epoch'tan devam eder.
#  Kosu boyunca uyku engellenir.
#
#  Ciktilar: outputs/v2_loglar/gece1b.log, siniflandirici_devam_testi.txt,
#            uzlasma_testi.txt, gece1b_DONE.txt
#  Kullanim: powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\gece1b_siniflandirici_testleri.ps1
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar"
$log    = "$logdir\gece1b.log"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

if (Test-Path "$logdir\gece1b_DONE.txt") { Write-Host "gece1b zaten tamamlanmis."; exit 0 }

Add-Type -Namespace KosuGuc1b -Name Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
[void][KosuGuc1b.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor [uint32]1)

function Kos([string]$argumanlar) {
    "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  python $argumanlar" | Out-File $log -Append -Encoding utf8
    cmd /c "`"$py`" -u $argumanlar >> $log 2>&1"
    return $LASTEXITCODE
}

try {
    # --- 0) CLAP kosusunun bitmesini bekle (DONE veya HATA) ---
    while (-not ((Test-Path "$logdir\clap_dongu1_DONE.txt") -or (Test-Path "$logdir\clap_dongu1_HATA.txt"))) {
        Write-Host "CLAP dongu 1 bekleniyor... $(Get-Date -Format HH:mm)"
        Start-Sleep -Seconds 120
    }
    Write-Host "CLAP bitti. Siniflandirici testleri basliyor $(Get-Date -Format HH:mm)"

    # --- 1) Devam testi ---
    if (-not (Test-Path "$logdir\siniflandirici_devam_testi.txt")) {
        Remove-Item results\v2\siniflandirma\_test*.json, results\v2\tahminler\_test*.npz, models\v2\cnn__test* -ErrorAction SilentlyContinue
        $k = Kos "scripts\15_siniflandirma_v2.py --konfig _testA --seed 7 --epochs 2"
        $k = Kos "scripts\15_siniflandirma_v2.py --konfig _testB --seed 7 --epochs 2 --kesinti_epoch 1"
        $k = Kos "scripts\15_siniflandirma_v2.py --konfig _testB --seed 7 --epochs 2"
        cmd /c "`"$py`" scripts\denetim\siniflandirici_dogrulama.py devam > $logdir\siniflandirici_devam_testi.txt 2>&1"
        Get-Content "$logdir\siniflandirici_devam_testi.txt" | Select-Object -First 1 | Write-Host
        Remove-Item results\v2\siniflandirma\_test*.json, results\v2\tahminler\_test*.npz, models\v2\cnn__test* -ErrorAction SilentlyContinue
    }

    # --- 2) v1-v2 uzlastirma = v2 baseline seed 42 ---
    $k = Kos "scripts\15_siniflandirma_v2.py --konfig baseline --seed 42"
    if (Test-Path "results\v2\siniflandirma\baseline_s42.json") {
        cmd /c "`"$py`" scripts\denetim\siniflandirici_dogrulama.py uzlasma > $logdir\uzlasma_testi.txt 2>&1"
        Get-Content "$logdir\uzlasma_testi.txt" | Select-Object -First 1 | Write-Host
        "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\gece1b_DONE.txt" -Encoding utf8
    } else {
        "HATA: baseline_s42.json olusmadi (kod=$k) $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\gece1b_HATA.txt" -Encoding utf8
    }
}
finally {
    [void][KosuGuc1b.Win32]::SetThreadExecutionState($ES_CONTINUOUS)
}
