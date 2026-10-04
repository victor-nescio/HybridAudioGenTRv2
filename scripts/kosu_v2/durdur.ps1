# Kuyrugu temiz duraklatir: mevcut epoch (siniflandirici ~10 dk, CLAP ~40 dk) ya da
# mevcut klip (~1 dk) bitince durum kaydedilir ve kuyruk durur. Is kaybi yok.
# Devam: powershell -ExecutionPolicy Bypass -File F:\Drthesiscode\scripts\kosu_v2\kuyruk_v2.ps1
$f = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")) "outputs\v2_loglar\kuyruk\DURDUR"
New-Item -ItemType Directory -Force (Split-Path $f) | Out-Null
"DURDUR $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File $f -Encoding utf8
Write-Host "Duraklatma istegi birakildi. Kuyruk mevcut epoch/klip bitince duracak."
Write-Host "Devam etmek icin kuyruk_v2.ps1'i yeniden calistirin."
