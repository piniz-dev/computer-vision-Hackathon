@echo off
echo ========================================================
echo  Installazione Ambiente Grok Hackathon (Visione A + B)
echo ========================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato! Assicurati di aver installato Python 3.11+ e aggiunto al PATH.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [*] Creazione virtual environment .venv...
    python -m venv .venv
)

echo [*] Attivazione .venv...
call .venv\Scripts\activate

echo [*] Aggiornamento pip...
python -m pip install --upgrade pip

echo [*] Installazione dipendenze Visione B (Grok)...
pip install -r vision\grok\requirements.txt

echo [*] Installazione dipendenze Visione A (Occhi)...
pip install -r vision\eyes\requirements.txt

if not exist ".env" (
    echo [*] Creazione .env da .env.example...
    copy .env.example .env
)

echo.
echo ========================================================
echo  Installazione completata con successo!
echo  Per attivare l'ambiente esegui: .venv\Scripts\activate
echo  Per eseguire i test:            python -m pytest -v
echo ========================================================
pause
