# ============================================================
#  TUTARLI-HEDEF DENEYI (H1): augmentasyon akustik olarak tutarli
#  siniflarda ise yariyor mu? Catch-all "diger hayvanlar" ile karsilastir.
#
#  Hedefler: kopek (tutarli), kus (tutarli), hava (tutarli),
#            diger_hayvanlar (catch-all kontrol)
#  Uretim: pretrained AudioLDM2 + EN prompt, seed-varied (base 5000)
#  Siniflandirma: paylasilan baseline + augmented (--sadece_aug, --sadece_cnn)
#
#  Kullanim: powershell -ExecutionPolicy Bypass -File .\coherent_deney.ps1
#  Tahmini: ~3.3s uretim + ~3s siniflandirma
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))  # repo koku
$py = "C:\Users\User\anaconda3\envs\audiogen\python.exe"

Write-Host "=== ADIM 1: URETIM (tutarli siniflar + catch-all kontrol) $(Get-Date -Format HH:mm) ==="
& $py scripts/05_generate_audio.py --n_klip 50 --tur coh --dil ingilizce --base_seed 5000 --siniflar kopek_sesleri kus_sesleri hava_olaylari diger_hayvanlar 2>&1 | Tee-Object -FilePath "outputs\uretim_coherent.log"

Write-Host ""
Write-Host "=== ADIM 2: SINIFLANDIRMA (paylasilan baseline + augmented) $(Get-Date -Format HH:mm) ==="
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_turcoh --siniflar kopek_sesleri kus_sesleri hava_olaylari diger_hayvanlar --sadece_aug --sadece_cnn --batch_size 64 --tur_etiketi coherent 2>&1 | Tee-Object -FilePath "outputs\sinif_coherent.log"

Write-Host ""
Write-Host "=== TUTARLI-HEDEF DENEYI TAMAMLANDI $(Get-Date -Format HH:mm) ==="
