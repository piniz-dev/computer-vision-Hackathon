@echo off
cd /d "%~dp0"
echo ========================================================
echo  Avvio Demo Live: Visione A (Occhi) + Visione B (Grok)
echo ========================================================
echo.
python demo_live.py
if errorlevel 1 (
    echo.
    echo [Info] Se non hai una webcam collegata, provo con la modalita replay:
    python demo_live.py --replay
)
pause
