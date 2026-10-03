# ============================================================
#  v2 ANA DENEY KUYRUGU (revizyon, Access-2026-36903)
#
#  Adimlar sirayla calisir; her adim bitince isaretlenir
#  (outputs/v2_loglar/kuyruk/<adim>.DONE). Kesinti/kapanmada AYNI KOMUT
#  yeniden calistirilir: tamamlanan adimlar atlanir, yarim kalanlar
#  kaldigi yerden devam eder (uretim: klip bazinda; CLAP/siniflandirici:
#  epoch bazinda). Kosu boyunca uyku engellenir.
#
#  Kullanim:
#    powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\kuyruk_v2.ps1
#    ... -SadeceAdim 05_C3p_uretim     # tek adim
#    ... -Listele                      # adim durumlarini goster, calistirma
#
#  Tasarim (revizyon plani K1-K7):
#   - CLAP yalnizca siniflandirici-egitim bolumuyle (08 --surum v2).
#   - Uretim: Grup A (kopek, kus, hava) + Grup B (insaat, dini mekan, acil
#     servis) + kontrol (diger hayvanlar), sinif basina 50 klip; C4p'de
#     Grup A + kontrol icin doz serisi icin 200 klip.
#   - Siniflandirici: patience 30 (erken durdurma yok -> son epoch modeli
#     val'e dayanmaz; val+test birlikte degerlendirilebilir).
# ============================================================
param([string]$SadeceAdim = "", [switch]$Listele)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar\kuyruk"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$A   = "kopek_sesleri,kus_sesleri,hava_olaylari"
$B   = "insaat_sesleri,dini_mekan,acil_servis"
$K   = "diger_hayvanlar"
$TUM = "kopek_sesleri kus_sesleri hava_olaylari insaat_sesleri dini_mekan acil_servis diger_hayvanlar"
$H1  = "--kaynak data/synthetic_turcoh:$A,$K --kaynak data/synthetic:$B"
$PAT = "--patience 30"

# (ad, komut) - komutlar repo kokunden calisir
$adimlar = @(
  @("01_CLAP_dongu1",   "scripts\08_clap_finetune.py --surum v2 --dongu dongu1 --epochs 10 --patience 3 --hassasiyet fp32 --resume"),
  @("02_entegre_d1",    "scripts\09_integrate_clap.py --clap_ckpt clap_v2_dongu1_best.pt --cikti audioldm2_clap_v2_dongu1.pth"),
  @("03_C2p_uretim",    "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C2p --checkpoint checkpoints/audioldm2_clap_v2_dongu1.pth --dil ingilizce --base_seed 9000 --n_klip 50 --siniflar $TUM"),
  @("04_C3p_uretim",    "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C3p --checkpoint checkpoints/audioldm2_clap_v2_dongu1.pth --dil turkce --base_seed 10000 --n_klip 50 --siniflar $TUM"),
  @("05_CLAP_dongu2",   "scripts\08_clap_finetune.py --surum v2 --dongu dongu2 --sentetik_veri data/synthetic_v2/C2p --epochs 10 --patience 3 --hassasiyet fp32 --resume"),
  @("06_entegre_d2",    "scripts\09_integrate_clap.py --clap_ckpt clap_v2_dongu2_best.pt --cikti audioldm2_clap_v2_dongu2.pth"),
  @("07_C4p_uretim",    "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C4p --checkpoint checkpoints/audioldm2_clap_v2_dongu2.pth --dil ingilizce --base_seed 11000 --n_klip 50 --siniflar insaat_sesleri dini_mekan acil_servis"),
  @("08_C4p_uretim_doz","scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C4p --checkpoint checkpoints/audioldm2_clap_v2_dongu2.pth --dil ingilizce --base_seed 11000 --n_klip 200 --siniflar kopek_sesleri kus_sesleri hava_olaylari diger_hayvanlar")
)
# Siniflandirici kosulari (her biri ~3,5-4 sa); oncelik sirasina gore
$C2p = "--kaynak data/synthetic_v2/C2p:$A,$B,$K --n_klip 50"
$C3p = "--kaynak data/synthetic_v2/C3p:$A,$B,$K --n_klip 50"
$C4p = "--kaynak data/synthetic_v2/C4p:$A,$B,$K --n_klip 50"
foreach ($s in 42, 43, 44, 45, 46) {
  if ($s -ne 42) { $adimlar += ,@("S_baseline_s$s", "scripts\15_siniflandirma_v2.py --konfig baseline --seed $s $PAT") }
  $adimlar += ,@("S_H1_s$s",  "scripts\15_siniflandirma_v2.py --konfig H1 --seed $s $H1 --n_klip 50 $PAT")
  $adimlar += ,@("S_C4p_s$s", "scripts\15_siniflandirma_v2.py --konfig C4p --seed $s $C4p $PAT")
}
foreach ($s in 42, 43, 44) {
  $adimlar += ,@("S_C2p_s$s",         "scripts\15_siniflandirma_v2.py --konfig C2p --seed $s $C2p $PAT")
  $adimlar += ,@("S_C3p_s$s",         "scripts\15_siniflandirma_v2.py --konfig C3p --seed $s $C3p $PAT")
  $adimlar += ,@("S_specaug_s$s",     "scripts\15_siniflandirma_v2.py --konfig specaug --seed $s --spec_augment $PAT")
  $adimlar += ,@("S_specaugC4p_s$s",  "scripts\15_siniflandirma_v2.py --konfig specaugC4p --seed $s $C4p --spec_augment $PAT")
  foreach ($n in 25, 50, 100, 200) {
    $adimlar += ,@("S_doz$n`_s$s", "scripts\15_siniflandirma_v2.py --konfig doz$n --seed $s --kaynak data/synthetic_v2/C4p:$A,$K --n_klip $n $PAT")
  }
}

# Bu gecenin ayri scriptleri (gece1 / gece1b) tamamladiysa ilgili adimlar tamam sayilir
if ((Test-Path "outputs\v2_loglar\clap_dongu1_DONE.txt") -and -not (Test-Path "$logdir\01_CLAP_dongu1.DONE")) {
  "DONE (gece1_clap_dongu1.ps1 ile)" | Out-File "$logdir\01_CLAP_dongu1.DONE" -Encoding utf8
}

if ($Listele) {
  foreach ($a in $adimlar) {
    $d = if (Test-Path "$logdir\$($a[0]).DONE") { "TAMAM" } else { "-" }
    "{0,-22} {1}" -f $a[0], $d
  }
  exit 0
}

Add-Type -Namespace KosuGucQ -Name Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
[void][KosuGucQ.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor [uint32]1)
try {
  foreach ($a in $adimlar) {
    $ad, $komut = $a[0], $a[1]
    if ($SadeceAdim -and $ad -ne $SadeceAdim) { continue }
    if (Test-Path "$logdir\$ad.DONE") { continue }
    Write-Host "=== $ad basliyor $(Get-Date -Format 'yyyy-MM-dd HH:mm') ==="
    "`n### $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $komut" | Out-File "$logdir\$ad.log" -Append -Encoding utf8
    cmd /c "`"$py`" -u $komut >> $logdir\$ad.log 2>&1"
    if ($LASTEXITCODE -eq 0) {
      "DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\$ad.DONE" -Encoding utf8
    } else {
      "HATA kod=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\$ad.HATA" -Encoding utf8
      Write-Host "=== $ad HATA (kod=$LASTEXITCODE) - kuyruk durdu; log: $logdir\$ad.log ==="
      break
    }
  }
}
finally {
  [void][KosuGucQ.Win32]::SetThreadExecutionState($ES_CONTINUOUS)
}
