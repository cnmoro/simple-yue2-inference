@echo off
rem Stop the YuE2 webUI that run_hidden.vbs started.
setlocal
cd /d "%~dp0"
set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":7860 "') do (
    set "FOUND=1"
    echo stopping pid %%p
    taskkill /f /pid %%p >nul 2>&1
)
if not defined FOUND echo nothing listening on 7860
