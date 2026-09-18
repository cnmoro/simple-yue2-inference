@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set YUE2_DEVICE=cuda
set YUE2_BACKEND=torch-eager
set YUE2_MEMORY_GIB=16
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente .venv nao encontrado nesta pasta.
    echo        Rode a instalacao antes de continuar.
    pause
    exit /b 1
)

rem Locate ffmpeg: PATH first, then the winget install location.
if not defined YUE2_FFMPEG for /f "delims=" %%i in ('where ffmpeg 2^>nul') do if not defined YUE2_FFMPEG set "YUE2_FFMPEG=%%i"
if not defined YUE2_FFMPEG for /f "delims=" %%i in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\ffmpeg.exe" 2^>nul') do if not defined YUE2_FFMPEG set "YUE2_FFMPEG=%%i"

netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":7860 " >nul
if %errorlevel%==0 (
    echo Servidor YuE2 ja esta rodando.
    start "" "http://127.0.0.1:7860"
    exit /b 0
)

call ".venv\Scripts\activate.bat"

echo Abrindo http://127.0.0.1:7860 no navegador em alguns segundos...
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep 8; Start-Process 'http://127.0.0.1:7860'"

echo.
echo Iniciando YuE2 Studio. Feche esta janela ou pressione CTRL+C para encerrar.
if defined YUE2_FFMPEG echo MP3 disponivel via: %YUE2_FFMPEG%
echo.
python app.py

echo.
echo [servidor encerrado] Se houve erro acima, copie a mensagem e me envie.
pause
