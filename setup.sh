#!/bin/bash
set -e

echo "========================================================"
echo " Installazione Ambiente Grok Hackathon (Visione A + B)"
echo "========================================================"

if ! command -v python3 &> /dev/null; then
    echo "[ERRORE] python3 non trovato! Installa Python 3.11+."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "[*] Creazione virtual environment .venv..."
    python3 -m venv .venv
fi

echo "[*] Attivazione .venv..."
source .venv/bin/activate

echo "[*] Aggiornamento pip..."
pip install --upgrade pip

echo "[*] Installazione dipendenze Visione B (Grok)..."
pip install -r vision/grok/requirements.txt

echo "[*] Installazione dipendenze Visione A (Occhi)..."
pip install -r vision/eyes/requirements.txt

if [ ! -f ".env" ]; then
    echo "[*] Creazione .env da .env.example..."
    cp .env.example .env
fi

echo "========================================================"
echo " Installazione completata!"
echo " Per attivare l'ambiente esegui: source .venv/bin/activate"
echo " Per eseguire i test:            python -m pytest -v"
echo "========================================================"
