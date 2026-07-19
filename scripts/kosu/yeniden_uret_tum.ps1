# ============================================================
#  Tum turlari (1-4) DUZELTILMIS seed mantigiyla yeniden uretir.
#  Her klip benzersiz seed (base_seed + global_idx) ile uretilir;
#  resume guvenli (kesilirse mevcut sayisindan devam eder).
#
#  Kullanim (proje kokunde ya da herhangi bir yerden):
#     powershell -ExecutionPolicy Bypass -File .\yeniden_uret_tum.ps1
#
#  Tahmini sure: ~14 saat (4 tur x 4 sinif x 50 klip x ~1 dk).
#  Her tur ayri log dosyasina yazar: outputs\uretim_turN.log
#  ISTENILEN AN DURDURULABILIR: Ctrl+C ya da process kill;
#  ayni komutu tekrar calistir, kaldigi yerden devam eder.
# ============================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot   # her zaman proje kokunden calis
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$ckptD1 = "checkpoints/audioldm2_clap_finetuned.pth"        # Dongu 1 (Tur 2-3)
$ckptD2 = "checkpoints/audioldm2_clap_dongu2_finetuned.pth" # Dongu 2 (Tur 4)
$baslangic = Get-Date

function Uret($tur, $dil, $ckpt, $seed, $etiket) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  TUR $tur  ($etiket)   baslangic: $(Get-Date -Format HH:mm:ss)"
    Write-Host "============================================================"
    $log = "outputs\uretim_tur$tur.log"
    if ($ckpt) {
        & $py scripts/05_generate_audio.py --n_klip 50 --tur $tur --dil $dil --checkpoint $ckpt --base_seed $seed 2>&1 | Tee-Object -FilePath $log
    } else {
        & $py scripts/05_generate_audio.py --n_klip 50 --tur $tur --dil $dil --base_seed $seed 2>&1 | Tee-Object -FilePath $log
    }
    Write-Host "  TUR $tur tamamlandi: $(Get-Date -Format HH:mm:ss)"
}

# Tur 1: pretrained AudioLDM2 (checkpoint yok), Ingilizce prompt
Uret "1" "ingilizce" $null   1000 "pretrained + EN"
# Tur 2: Dongu 1 ince ayarli, Ingilizce prompt
Uret "2" "ingilizce" $ckptD1 2000 "Dongu1 FT + EN"
# Tur 3: Dongu 1 ince ayarli, Turkce prompt
Uret "3" "turkce"    $ckptD1 3000 "Dongu1 FT + TR"
# Tur 4: Dongu 2 ince ayarli, Ingilizce prompt
Uret "4" "ingilizce" $ckptD2 4000 "Dongu2 FT + EN"

# Cesitlilik dogrulamasi ayri scriptte:  python scripts/cesitlilik_kontrol.py
& $py scripts/cesitlilik_kontrol.py

$sure = (Get-Date) - $baslangic
Write-Host ""
Write-Host "=== TUM URETIM TAMAMLANDI - toplam sure: $([int]$sure.TotalHours) saat $($sure.Minutes) dk ==="
Write-Host "Sonraki adim: FAD olcumu, siniflandirma, McNemar (bkz YENIDEN_URETIM_PIPELINE.md)"
