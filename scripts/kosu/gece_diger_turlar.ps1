# ============================================================
#  GECE SURUCUSU — diger turlar icin tutarli-sinif uretimleri
#  (kopek/kus/hava) + siniflandirma.
#
#  KENDI-KENDINI TETIKLER: once mevcut H1 siniflandirmasinin
#  bitmesini bekler (outputs\sinif_coherent_done.txt), GPU
#  cakismasini onler; sonra 3 turu sirayla calistirir:
#    coh2 = Dongu1 (CLAP FT) + EN
#    coh3 = Dongu1 (CLAP FT) + TR
#    coh4 = Dongu2 (CLAP FT) + EN
#
#  Her tur: kopek/kus/hava x 50 uretim + augmented CNN siniflandirma.
#  diger_hayvanlar kontrolu ana turlarin (Tur2/3/4) mevcut
#  sonuclarindan alinir (yeniden uretilmez).
#
#  Kullanim: powershell -ExecutionPolicy Bypass -File .\gece_diger_turlar.ps1
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$ckptD1 = "checkpoints/audioldm2_clap_finetuned.pth"
$ckptD2 = "checkpoints/audioldm2_clap_dongu2_finetuned.pth"
$siniflar = @("kopek_sesleri","kus_sesleri","hava_olaylari")

# --- 1) Mevcut H1 siniflandirmasinin bitmesini bekle ---
Write-Host "=== Mevcut H1 siniflandirmasi bekleniyor... $(Get-Date -Format HH:mm) ==="
$bekleme = 0
while (-not (Test-Path "outputs\sinif_coherent_done.txt")) {
    Start-Sleep -Seconds 120
    $bekleme += 2
    if ($bekleme -ge 240) { Write-Host "UYARI: 4 saat beklendi, yine de devam ediliyor."; break }
}
Write-Host "=== H1 bitti (ya da zaman asimi). Diger turlar basliyor. $(Get-Date -Format HH:mm) ==="

function TurKosu($tur, $dil, $ckpt, $seed, $etiket) {
    Write-Host ""
    Write-Host "############################################################"
    Write-Host "#  $etiket  (tur=$tur, dil=$dil)  baslangic: $(Get-Date -Format HH:mm:ss)"
    Write-Host "############################################################"
    # Uretim
    Write-Host "--- Uretim ($etiket) ---"
    & $py scripts/05_generate_audio.py --n_klip 50 --tur $tur --dil $dil --checkpoint $ckpt --base_seed $seed --siniflar kopek_sesleri kus_sesleri hava_olaylari 2>&1 | Tee-Object -FilePath "outputs\uretim_$tur.log"
    # Siniflandirma
    Write-Host "--- Siniflandirma ($etiket) ---"
    & $py scripts/07_iterative_loop.py --sentetik_dir "data/synthetic_tur$tur" --siniflar kopek_sesleri kus_sesleri hava_olaylari --sadece_aug --sadece_cnn --batch_size 64 --tur_etiketi $etiket 2>&1 | Tee-Object -FilePath "outputs\sinif_$tur.log"
    Write-Host "#  $etiket tamamlandi: $(Get-Date -Format HH:mm:ss)"
    "DONE $etiket $(Get-Date -Format HH:mm)" | Out-File -FilePath "outputs\$tur`_done.txt" -Encoding utf8
}

# --- 2) Uc tur (oncelik sirasi: Dongu1+EN en bilgilendirici) ---
TurKosu "coh2" "ingilizce" $ckptD1 6000 "coherent2_Dongu1_EN"
TurKosu "coh3" "turkce"    $ckptD1 7000 "coherent3_Dongu1_TR"
TurKosu "coh4" "ingilizce" $ckptD2 8000 "coherent4_Dongu2_EN"

Write-Host ""
Write-Host "=== TUM DIGER TURLAR TAMAMLANDI $(Get-Date -Format HH:mm) ==="
"TUM_DIGER_TURLAR_DONE $(Get-Date)" | Out-File -FilePath "outputs\gece_diger_turlar_done.txt" -Encoding utf8
