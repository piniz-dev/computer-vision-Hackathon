@echo off
cd /d "%~dp0"
echo ========================================================
echo  Avvio Demo Live: Visione A (Occhi) + Visione B (Grok)
echo ========================================================
echo.
rem Usa il Python del .venv (dove setup.bat installa i pacchetti), non quello di sistema
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if "%PY%"=="python" echo [Attenzione] .venv non trovato: esegui prima setup.bat
"%PY%" demo_live.py %*
if errorlevel 1 (
    echo.
    echo [Info] Se non hai una webcam collegata, provo con la modalita replay:
    "%PY%" demo_live.py --replay %*
)
pause
