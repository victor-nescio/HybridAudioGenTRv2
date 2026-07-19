@echo off
SET PYTHON=C:\Users\User\anaconda3\envs\audiogen\python.exe
SET ROOT=f:\Drthesiscode
SET LOG=%ROOT%\gece_log.txt

echo ============================================================ > %LOG%
echo  Gece Egitim Basladi: %date% %time% >> %LOG%
echo ============================================================ >> %LOG%

cd /d %ROOT%

:: ── 1. CNN Augmented (30 epoch, adil karsilastirma) ─────────────────────────
echo. >> %LOG%
echo [1/4] CNN Augmented egitim basliyor (30 epoch)... %time% >> %LOG%
echo [1/4] CNN Augmented egitim basliyor (30 epoch)...
%PYTHON% scripts\07_iterative_loop.py --sadece_aug --sadece_cnn --epochs 30 --patience 7 >> %LOG% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo HATA: CNN augmented basarisiz. Devam ediliyor... >> %LOG%
    echo HATA: CNN augmented basarisiz.
) ELSE (
    echo CNN augmented tamamlandi: %time% >> %LOG%
    echo CNN augmented tamamlandi.
)

:: ── 2. LightGBM Augmented Loop ─────────────────────────────────────────────
echo. >> %LOG%
echo [2/4] LightGBM Augmented Loop basliyor... %time% >> %LOG%
echo [2/4] LightGBM Augmented Loop basliyor...
%PYTHON% scripts\07_iterative_loop.py --sadece_aug --sadece_lgbm >> %LOG% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo HATA: LightGBM loop basarisiz. Devam ediliyor... >> %LOG%
    echo HATA: LightGBM loop basarisiz.
) ELSE (
    echo LightGBM tamamlandi: %time% >> %LOG%
    echo LightGBM tamamlandi.
)

:: ── 2. fadtk Kur ────────────────────────────────────────────────────────────
echo. >> %LOG%
echo [2/3] fadtk kuruluyor... %time% >> %LOG%
echo [3/4] fadtk kuruluyor...
%PYTHON% -m pip install fadtk >> %LOG% 2>&1

:: ── 3. FAD Olcumu ───────────────────────────────────────────────────────────
echo [3/4] FAD olcumu basliyor... %time% >> %LOG%
echo [3/4] FAD olcumu basliyor...
%PYTHON% scripts\06_measure_fad.py >> %LOG% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo UYARI: FAD olcumu basarisiz (ihtiyari adim). >> %LOG%
    echo UYARI: FAD olcumu basarisiz.
) ELSE (
    echo FAD tamamlandi: %time% >> %LOG%
)

:: ── 4. PhD Raporu ────────────────────────────────────────────────────────────
echo. >> %LOG%
echo [4/4] PhD raporu guncelleniyor... %time% >> %LOG%
echo [4/4] PhD raporu guncelleniyor...
%PYTHON% scripts\11_phd_report.py >> %LOG% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo HATA: Rapor basarisiz. >> %LOG%
    echo HATA: Rapor basarisiz.
) ELSE (
    echo Rapor tamamlandi: %time% >> %LOG%
)

echo. >> %LOG%
echo ============================================================ >> %LOG%
echo  Tum islemler tamamlandi: %date% %time% >> %LOG%
echo ============================================================ >> %LOG%
echo.
echo Tum islemler tamamlandi! Log: %LOG%
pause
