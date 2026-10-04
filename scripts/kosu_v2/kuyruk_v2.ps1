# ============================================================
#  v2 ANA DENEY KUYRUGU (revizyon, Access-2026-36903)
#
#  Adimlar oncelik sirasiyla calisir; her adim bitince isaretlenir
#  (outputs/v2_loglar/kuyruk/<adim>.DONE). Kesinti/kapanmada AYNI KOMUT
#  yeniden calistirilir: tamamlanan adimlar atlanir, yarim kalanlar kaldigi
#  yerden devam eder (uretim: klip; CLAP/siniflandirici: epoch).
#  Kosu boyunca uyku engellenir.
#
#  DURAKLATMA: scripts\kosu_v2\durdur.ps1 calistirilinca mevcut epoch/klip
#  bitince durum kaydedilir ve kuyruk durur (is kaybi yok). Devam icin bu
#  scripti yeniden calistirmak yeterli.
#
#  Kullanim:
#    powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\kuyruk_v2.ps1
#    ... -Listele                      # adim durumlari
#    ... -SadeceAdim <ad>              # tek adim
#
#  Tasarim (revizyon plani; kararlar 4 Ekim 2026):
#   K3  Grup A: kopek, kus, hava. Grup B: en dusuk 3 baseline F1 (insaat, dini mekan,
#       tarim/ciftlik) + acil servis (1. faz surekliligi). Kontrol: diger hayvanlar.
#   K5  Siniflandirici: 30 epoch, erken durdurma YOK (patience 30); birincil analiz son
#       epoch modeli, val+test birlikte (docs/v2_onkayit.md).
#   K14 Icerik kontrolu: C4p klipleri, etiketler siniflar arasinda dongusel kaydirilmis.
#   Doz: 25/100/200 (Grup A + kontrol; Grup B 50'de sabit); doz 50 = ana C4p kosulari.
# ============================================================
param([string]$SadeceAdim = "", [switch]$Listele)

$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))   # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$logdir = "outputs\v2_loglar\kuyruk"
New-Item -ItemType Directory -Force $logdir | Out-Null
$env:PYTHONIOENCODING = "utf-8"

$A   = "kopek_sesleri,kus_sesleri,hava_olaylari"
$B   = "insaat_sesleri,dini_mekan,tarim_ciftlik,acil_servis"
$K   = "diger_hayvanlar"
$TUM = "kopek_sesleri kus_sesleri hava_olaylari insaat_sesleri dini_mekan tarim_ciftlik acil_servis diger_hayvanlar"
$AK  = "kopek_sesleri kus_sesleri hava_olaylari diger_hayvanlar"
$H1  = "--kaynak data/synthetic_turcoh:$A,$K --kaynak data/synthetic:insaat_sesleri,dini_mekan,acil_servis --kaynak data/synthetic_v2/H1ek:tarim_ciftlik"
$C2p = "--kaynak data/synthetic_v2/C2p:$A,$B,$K"
$C3p = "--kaynak data/synthetic_v2/C3p:$A,$B,$K"
$C4p = "--kaynak data/synthetic_v2/C4p:$A,$B,$K"
$PAT = "--patience 30 --n_klip 50"
$S   = "scripts\15_siniflandirma_v2.py"

$adimlar = @(
  # --- Uretim zinciri (birincil hipotez icin gerekenler once) ---
  @("00_H1_tarim",       "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/H1ek --dil ingilizce --base_seed 12000 --n_klip 50 --siniflar tarim_ciftlik"),
  @("01_CLAP_dongu1",    "scripts\08_clap_finetune.py --surum v2 --dongu dongu1 --epochs 10 --patience 3 --hassasiyet fp32 --resume"),
  @("02_entegre_d1",     "scripts\09_integrate_clap.py --clap_ckpt clap_v2_dongu1_best.pt --cikti audioldm2_clap_v2_dongu1.pth"),
  @("03_C2p_uretim",     "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C2p --checkpoint checkpoints/audioldm2_clap_v2_dongu1.pth --dil ingilizce --base_seed 9000 --n_klip 50 --siniflar $TUM"),
  @("04_CLAP_dongu2",    "scripts\08_clap_finetune.py --surum v2 --dongu dongu2 --sentetik_veri data/synthetic_v2/C2p --epochs 10 --patience 3 --hassasiyet fp32 --resume"),
  @("05_entegre_d2",     "scripts\09_integrate_clap.py --clap_ckpt clap_v2_dongu2_best.pt --cikti audioldm2_clap_v2_dongu2.pth"),
  @("06_C4p_uretim",     "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C4p --checkpoint checkpoints/audioldm2_clap_v2_dongu2.pth --dil ingilizce --base_seed 11000 --n_klip 50 --siniflar $TUM")
)
# --- Birincil hipotez: C4p vs baseline (5 seed); baseline s42 gece 1'de bitti ---
foreach ($s in 42, 43, 44, 45, 46) {
  $adimlar += ,@("S_C4p_s$s", "$S --konfig C4p --seed $s $C4p $PAT")
  if ($s -ne 42) { $adimlar += ,@("S_baseline_s$s", "$S --konfig baseline --seed $s $PAT") }
}
# --- Icerik kontrolu (K14) ---
foreach ($s in 42, 43, 44) { $adimlar += ,@("S_C4pkaydir_s$s", "$S --konfig C4p_kaydir --seed $s $C4p --etiket_kaydir $PAT") }
# --- Hazir uretici (H1), 5 seed ---
foreach ($s in 42, 43, 44, 45, 46) { $adimlar += ,@("S_H1_s$s", "$S --konfig H1 --seed $s $H1 $PAT") }
# --- C2p, C3p (3 seed); C3p uretimi burada ---
$adimlar += ,@("07_C3p_uretim", "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C3p --checkpoint checkpoints/audioldm2_clap_v2_dongu1.pth --dil turkce --base_seed 10000 --n_klip 50 --siniflar $TUM")
foreach ($s in 42, 43, 44) {
  $adimlar += ,@("S_C2p_s$s", "$S --konfig C2p --seed $s $C2p $PAT")
  $adimlar += ,@("S_C3p_s$s", "$S --konfig C3p --seed $s $C3p $PAT")
}
# --- SpecAugment kiyaslari (3 seed) ---
foreach ($s in 42, 43, 44) {
  $adimlar += ,@("S_specaug_s$s",    "$S --konfig specaug --seed $s --spec_augment $PAT")
  $adimlar += ,@("S_specaugC4p_s$s", "$S --konfig specaugC4p --seed $s $C4p --spec_augment $PAT")
}
# --- Doz serisi (Grup A + kontrol 25/100/200; Grup B 50'de sabit; doz 50 = C4p) ---
$adimlar += ,@("08_C4p_uretim_doz", "scripts\05_generate_audio.py --cikti_dir data/synthetic_v2/C4p --checkpoint checkpoints/audioldm2_clap_v2_dongu2.pth --dil ingilizce --base_seed 11000 --n_klip 200 --siniflar $AK")
foreach ($s in 42, 43, 44) {
  foreach ($n in 25, 100, 200) {
    $adimlar += ,@("S_doz$n`_s$s", "$S --konfig doz$n --seed $s --kaynak data/synthetic_v2/C4p:$A,$K#$n --kaynak data/synthetic_v2/C4p:$B#50 --patience 30")
  }
}

# Gece 1 scriptleri tamamladiysa ilgili adimlar tamam sayilir
if ((Test-Path "outputs\v2_loglar\clap_dongu1_DONE.txt") -and -not (Test-Path "$logdir\01_CLAP_dongu1.DONE")) {
  "DONE (gece1_clap_dongu1.ps1 ile)" | Out-File "$logdir\01_CLAP_dongu1.DONE" -Encoding utf8
}

if ($Listele) {
  foreach ($a in $adimlar) {
    $d = if (Test-Path "$logdir\$($a[0]).DONE") { "TAMAM" } elseif (Test-Path "$logdir\$($a[0]).HATA") { "HATA" } else { "-" }
    "{0,-20} {1}" -f $a[0], $d
  }
  exit 0
}

if (Test-Path "$logdir\DURDUR") {
  Remove-Item "$logdir\DURDUR"
  Write-Host "Onceki duraklatma istegi kaldirildi; kuyruk devam ediyor."
}

Add-Type -Namespace KosuGucQ -Name Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648
[void][KosuGucQ.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor [uint32]1)
try {
  foreach ($a in $adimlar) {
    $ad, $komut = $a[0], $a[1]
    if ($SadeceAdim -and $ad -ne $SadeceAdim) { continue }
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
      "HATA kod=$kod $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File "$logdir\$ad.HATA" -Encoding utf8
      Write-Host "=== $ad HATA (kod=$kod) - kuyruk durdu; log: $logdir\$ad.log ==="
      break
    }
  }
}
finally {
  [void][KosuGucQ.Win32]::SetThreadExecutionState($ES_CONTINUOUS)
}
