"""Tutti i test girano in modalità finta, con il log in una cartella temporanea."""
import numpy as np
import pytest

from vision.grok import client, voice_tools


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_MOCK", "1")
    monkeypatch.setattr(client, "LOG_PATH", tmp_path / "grok_calls.jsonl")
    monkeypatch.setattr(client, "_schema_supported", True)
    voice_tools._context.clear()


@pytest.fixture
def img():
    return np.full((240, 320, 3), 200, np.uint8)


@pytest.fixture
def objs():
    return [
        {"id": 3, "label": "cup", "name": "tazza rossa con manico", "conf": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2],
         "color": "rosso"},
        {"id": 5, "label": "marker", "name": "pennarello rosso", "conf": 0.8, "bbox": [0.5, 0.1, 0.1, 0.3],
         "color": "rosso"},
        {"id": 7, "label": "bottle", "name": None, "conf": 0.8, "bbox": [0.7, 0.5, 0.1, 0.3], "color": "blue"},
        {"id": 8, "label": "cup", "name": None, "conf": 0.8, "bbox": [0.2, 0.6, 0.1, 0.2], "color": "verde"},
    ]
