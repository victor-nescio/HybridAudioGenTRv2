Write-Host "FAD Tur2 bekleniyor..."
while (Get-Process | Where-Object { $_.Name -like "*python*" -and $_.StartTime -gt (Get-Date).AddHours(-2) }) {
    Start-Sleep -Seconds 30
}
Write-Host "Tur2 bitti, Tur3 baslatiliyor..."
Start-Process -FilePath "C:\Users\User\anaconda3\envs\audiogen\python.exe" -ArgumentList "scripts/06_measure_fad.py --sentetik_dir data/synthetic_tur3 --tur tur3" -WorkingDirectory "f:\Drthesiscode" -RedirectStandardOutput "f:\Drthesiscode\fad_tur3.log" -RedirectStandardError "f:\Drthesiscode\fad_tur3_err.log" -NoNewWindow -Wait
Write-Host "Tur3 de bitti."
