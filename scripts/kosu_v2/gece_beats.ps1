# ============================================================
#  v2 GECE KUYRUGU 2 (16 Ekim 2026): ikinci siniflandirici (BEATs) + ezber + TR-EN
#
#  Ana kuyrukla (kuyruk_v2.ps1) ayni mantik ve ayni log klasoru: tamamlanan adim
#  <ad>.DONE ile atlanir; kesintide ayni komut kaldigi yerden devam eder (BEATs gomme:
#  500 kayitlik blok). Duraklatma: scripts\kosu_v2\durdur.ps1 (ayni DURDUR dosyasi).
#
#  Adimlar:
#   B0      BEATs gommeleri (gercek val/test/train + sentetik)
#   B_*     BEATs + MLP kafasi: baseline / H1 / C4p x seed 42-44 (R3, R5.3; on kayitta yok)
#   E_*     Ezber analizi (VGGish en yakin komsu; R1.1, R5.1, K2)
#   T_*     Ince-ayarli CLAP ile TR-EN prompt benzerligi (R2.8)
#
#  Kullanim:
#    powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\gece_beats.ps1 [-Listele]
# ============================================================
param([switch]$Listele)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar\kuyruk"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$GA  = "kopek_sesleri,kus_sesleri,hava_olaylari"
$GB  = "insaat_sesleri,dini_mekan,tarim_ciftlik,acil_servis"
$GK  = "diger_hayvanlar"
$H1  = "--kaynak data/synthetic_turcoh:$GA,$GK --kaynak data/synthetic:insaat_sesleri,dini_mekan,acil_servis --kaynak data/synthetic_v2/H1ek:tarim_ciftlik"
$C4p = "--kaynak data/synthetic_v2/C4p:$GA,$GB,$GK"
$BP  = "scripts\19_beats_probe.py"
$EZ  = "scripts\17_ezber_analizi.py"
$TR  = "scripts\denetim\tr_en_benzerlik.py"

$adimlar = @()
$adimlar += ,@("B0_beats_gomme", "$BP gomme")
foreach ($tohum in 42, 43, 44) {
  $adimlar += ,@("B_baseline_s$tohum", "$BP kafa --konfig baseline --seed $tohum")
  $adimlar += ,@("B_H1_s$tohum",       "$BP kafa --konfig H1 --seed $tohum $H1 --n_klip 50")
  $adimlar += ,@("B_C4p_s$tohum",      "$BP kafa --konfig C4p --seed $tohum $C4p --n_klip 50")
}
$adimlar += ,@("E_H1",  "$EZ $H1 --etiket H1")
$adimlar += ,@("E_C4p", "$EZ $C4p --etiket C4p")
$adimlar += ,@("T_dongu1", "$TR --ckpt checkpoints/clap_v2_dongu1_best.pt")
$adimlar += ,@("T_dongu2", "$TR --ckpt checkpoints/clap_v2_dongu2_best.pt")
$adimlar += ,@("E_C2p", "$EZ --kaynak data/synthetic_v2/C2p:$GA,$GB,$GK --etiket C2p")
$adimlar += ,@("E_C3p", "$EZ --kaynak data/synthetic_v2/C3p:$GA,$GB,$GK --etiket C3p")

if ($Listele) {
  foreach ($adim in $adimlar) {
    $durum = if (Test-Path "$logdir\$($adim[0]).DONE") { "TAMAM" } elseif (Test-Path "$logdir\$($adim[0]).HATA") { "HATA" } else { "-" }
    "{0,-20} {1}" -f $adim[0], $durum
  }
  exit 0
}

if (Test-Path "$logdir\DURDUR") {
  Remove-Item "$logdir\DURDUR"
  Write-Host "Onceki duraklatma istegi kaldirildi; kuyruk devam ediyor."
}

Add-Type -Namespace KosuGucB -Name Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
[void][KosuGucB.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor [uint32]1)
try {
  foreach ($adim in $adimlar) {
    $ad, $komut = $adim[0], $adim[1]
    if (Test-Path "$logdir\$ad.DONE") { continue }
    if (Test-Path "$logdir\DURDUR") { Write-Host "=== Duraklatildi (adimlar arasi) ==="; break }
    Remove-Item "$logdir\$ad.HATA" -ErrorAction SilentlyContinue
    Write-Host "=== $ad basliyor $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
    "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $komut" | Out-File "$logdir\$ad.log" -Append -Encoding utf8
    cmd /c "`"$py`" -u $komut >> $logdir\$ad.log 2>&1"
    $kod = $LASTEXITCODE
    if ($kod -eq 0) {
      "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\$ad.DONE" -Encoding utf8
    } elseif ($kod -eq 4) {
      Write-Host "=== $ad DURAKLATILDI $(Get-Date -Format 'HH:mm') - devam icin bu scripti yeniden calistirin ==="
      break
    } else {
      # Bagimsiz adimlar: bir analiz hatasi digerlerini durdurmasin; yalniz gomme hatasi durdurur
      "HATA kod=$kod $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\$ad.HATA" -Encoding utf8
      Write-Host "=== $ad HATA (kod=$kod); log: $logdir\$ad.log ==="
      if ($ad -eq "B0_beats_gomme") { break }
    }
  }
}
finally {
  [void][KosuGucB.Win32]::SetThreadExecutionState($ES_CONTINUOUS)
}
