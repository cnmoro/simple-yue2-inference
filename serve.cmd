@echo off
rem Run the YuE2 webUI with no console window. Logs go to logs\server.log.
rem Launch it by double-clicking run_hidden.vbs, and stop it with stop.cmd.
setlocal
cd /d "%~dp0"

if not exist "logs" mkdir "logs"

netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":7860 " >nul
if %errorlevel%==0 (
    echo [%date% %time%] already listening on 7860, nothing to do>> "logs\server.log"
    exit /b 0
)

if not exist ".venv\Scripts\pythonw.exe" (
    echo [%date% %time%] .venv\Scripts\pythonw.exe not found, install first>> "logs\server.log"
    exit /b 1
)

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set YUE2_DEVICE=cuda
set YUE2_BACKEND=torch-eager
set YUE2_MEMORY_GIB=16
set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

if not defined YUE2_FFMPEG for /f "delims=" %%i in ('where ffmpeg 2^>nul') do if not defined YUE2_FFMPEG set "YUE2_FFMPEG=%%i"
if not defined YUE2_FFMPEG for /f "delims=" %%i in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\ffmpeg.exe" 2^>nul') do if not defined YUE2_FFMPEG set "YUE2_FFMPEG=%%i"

echo [%date% %time%] starting>> "logs\server.log"
".venv\Scripts\pythonw.exe" app.py >> "logs\server.log" 2>&1
echo [%date% %time%] stopped, exit code %errorlevel%>> "logs\server.log"
