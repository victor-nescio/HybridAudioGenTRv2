# ============================================================
#  GECE SURUCUSU 2 — coh4 (Dongu2+EN) uretim+siniflandirma +
#  TUM koşularin (coh2/coh3/coh4) McNemar istatistiksel testleri.
#
#  H1 McNemar zaten yapildi (results/v1/mcnemar_coherent.json).
#  Bu surucu:
#   1) coh3 McNemar  (ckpt cnn_aug_coherent_coh3.pt hazir)
#   2) coh4: uret (Dongu2+EN seed8000) -> siniflandir -> ckpt yedekle -> McNemar
#   3) coh2: yeniden siniflandir (klipler diskte) -> ckpt yedekle -> McNemar
#   4) toplu ozet
#
#  Egitim seed'li (deterministik), coh2 re-run ayni sonucu uretir.
#  Kullanim: powershell -ExecutionPolicy Bypass -File .\gece_coh4_mcnemar.ps1
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..\.."))  # repo koku
$py     = "C:\Users\User\anaconda3\envs\audiogen\python.exe"
$ckptD2 = "checkpoints/audioldm2_clap_dongu2_finetuned.pth"
$log    = "outputs\gece_coh4.log"
function Say($m) { "$((Get-Date).ToString('HH:mm:ss'))  $m" | Tee-Object -FilePath $log -Append }

Say "=== GECE SURUCUSU 2 BASLADI ==="

# --- 1) coh3 McNemar (checkpoint hazir) ---
Say "[1/?] coh3 McNemar..."
& $py scripts/v1/mcnemar_coherent.py --aug_ckpt cnn_aug_coherent_coh3.pt --etiket coh3_Dongu1_TR --out results/v1/mcnemar_coh3.json 2>&1 | Tee-Object -FilePath $log -Append
Say "coh3 McNemar bitti."

# --- 2) coh4 uretim (Dongu2 + EN, seed 8000) ---
Say "[2a] coh4 URETIM (Dongu2+EN seed8000)..."
if (Test-Path data\synthetic_turcoh4) { Remove-Item data\synthetic_turcoh4 -Recurse -Force }
& $py scripts/05_generate_audio.py --n_klip 50 --tur coh4 --dil ingilizce --checkpoint $ckptD2 --base_seed 8000 --siniflar kopek_sesleri kus_sesleri hava_olaylari 2>&1 | Tee-Object -FilePath "outputs\uretim_coh4.log"
Say "coh4 uretim bitti ($((Get-ChildItem data\synthetic_turcoh4 -Recurse -Filter *.wav -ErrorAction SilentlyContinue).Count) klip)."

# --- coh4 siniflandirma ---
Say "[2b] coh4 SINIFLANDIRMA..."
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_turcoh4 --siniflar kopek_sesleri kus_sesleri hava_olaylari --sadece_aug --sadece_cnn --batch_size 64 --tur_etiketi coherent4_Dongu2_EN 2>&1 | Tee-Object -FilePath "outputs\sinif_coh4.log"
Copy-Item models\cnn_augmented_iter_best.pt models\cnn_aug_coherent_coh4.pt -Force
Say "coh4 siniflandirma bitti, ckpt yedeklendi (cnn_aug_coherent_coh4.pt)."

# --- coh4 McNemar ---
Say "[2c] coh4 McNemar..."
& $py scripts/v1/mcnemar_coherent.py --aug_ckpt cnn_aug_coherent_coh4.pt --etiket coh4_Dongu2_EN --out results/v1/mcnemar_coh4.json 2>&1 | Tee-Object -FilePath $log -Append
Say "coh4 McNemar bitti."
"DONE coh4 $(Get-Date -Format HH:mm)" | Out-File -FilePath "outputs\coh4_done.txt" -Encoding utf8

# --- 3) coh2 yeniden siniflandirma (checkpoint kurtarma) ---
Say "[3a] coh2 YENIDEN SINIFLANDIRMA (deterministik, ckpt kurtarma)..."
& $py scripts/07_iterative_loop.py --sentetik_dir data/synthetic_turcoh2 --siniflar kopek_sesleri kus_sesleri hava_olaylari --sadece_aug --sadece_cnn --batch_size 64 --tur_etiketi coherent2_Dongu1_EN_rerun 2>&1 | Tee-Object -FilePath "outputs\sinif_coh2_rerun.log"
Copy-Item models\cnn_augmented_iter_best.pt models\cnn_aug_coherent_coh2.pt -Force
Say "coh2 yeniden siniflandirma bitti, ckpt yedeklendi (cnn_aug_coherent_coh2.pt)."

# --- coh2 McNemar ---
Say "[3b] coh2 McNemar..."
& $py scripts/v1/mcnemar_coherent.py --aug_ckpt cnn_aug_coherent_coh2.pt --etiket coh2_Dongu1_EN --out results/v1/mcnemar_coh2.json 2>&1 | Tee-Object -FilePath $log -Append
Say "coh2 McNemar bitti."

Say "=== TUM ISLER TAMAMLANDI ==="
"TUM_COH4_MCNEMAR_DONE $(Get-Date)" | Out-File -FilePath "outputs\gece_coh4_mcnemar_done.txt" -Encoding utf8
