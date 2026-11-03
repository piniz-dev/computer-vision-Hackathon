"""Client: codifica immagini, nuovo tentativo, timeout, JSON non valido, ripiego su json_object, log."""
import base64
import json
import time

import cv2
import numpy as np
import openai
from pydantic import BaseModel

from vision.grok import client
from vision.grok.client import ask_json, encode_image

try:  # le versioni recenti dell'SDK OpenAI usano httpx2, le precedenti httpx
    import httpx2 as httpx
except ImportError:
    import httpx

REQ = httpx.Request("POST", "https://api.x.ai/v1/chat/completions")


class S(BaseModel):
    text: str


def test_encode_image_resizes_keeping_aspect():
    url = encode_image(np.zeros((1080, 1920, 3), np.uint8), max_side=1024)
    assert url.startswith("data:image/jpeg;base64,")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]), np.uint8), 1)
    assert img.shape == (576, 1024, 3)


def test_small_image_not_upscaled():
    url = encode_image(np.zeros((100, 50, 3), np.uint8))
    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]), np.uint8), 1)
    assert img.shape == (100, 50, 3)


def test_valid_first_try():
    r, meta = ask_json([], "p", S, mock=lambda: {"text": "ciao"})
    assert r.text == "ciao" and meta["attempts"] == 1 and meta["error"] is None


def test_invalid_json_then_valid():
    answers = iter(["non json", '{"text": "ok"}'])
    r, meta = ask_json([], "p", S, mock=lambda: next(answers))
    assert r.text == "ok" and meta["attempts"] == 2


def test_invalid_json_twice_gives_none():
    r, meta = ask_json([], "p", S, mock=lambda: '{"altro": 1}')
    assert r is None and meta["attempts"] == 2 and "JSON non valido" in meta["error"]


def test_timeout_twice_gives_none():
    def slow():
        raise TimeoutError
    r, meta = ask_json([], "p", S, mock=slow)
    assert r is None and meta["attempts"] == 2 and meta["error"] == "timeout"


def test_real_path_timeout_then_ok(monkeypatch):
    monkeypatch.setenv("GROK_MOCK", "0")
    calls = []

    def fake_api(content, schema, use_schema, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise openai.APITimeoutError(request=REQ)
        return '{"text": "ok"}', {"prompt": 10, "completion": 2}

    monkeypatch.setattr(client, "_call_api", fake_api)
    r, meta = ask_json([np.zeros((10, 10, 3), np.uint8)], "p", S, timeout=1.5)
    assert r.text == "ok" and meta["attempts"] == 2 and meta["tokens"]["completion"] == 2
    assert 1.4 < calls[0] <= 1.5 and 1.0 < calls[1] <= calls[0]  # il secondo tentativo ha il tempo rimasto


def test_real_path_deadline_is_total(monkeypatch):
    # Il timeout dell'SDK non scatta se il server tiene viva la connessione: lo imponiamo noi.
    monkeypatch.setenv("GROK_MOCK", "0")
    monkeypatch.setattr(client, "_call_api", lambda *a: time.sleep(3))
    r, meta = ask_json([], "p", S, timeout=0.3)
    assert r is None and meta["error"] == "timeout" and meta["latency_ms"] < 1000


def test_reasoning_effort_sent_to_openrouter(monkeypatch):
    sent = {}

    class Fake:
        def create(self, **kw):
            sent.update(kw)
            msg = type("M", (), {"content": '{"text": "ok"}'})
            return type("R", (), {"usage": None, "choices": [type("C", (), {"message": msg})]})

    monkeypatch.setattr(client, "_get_client", lambda: type("Cl", (), {"chat": type("Ch", (), {"completions": Fake()})}))
    monkeypatch.setattr(client, "BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("GROK_REASONING", raising=False)
    client._call_api([], S, True, 5)
    assert sent["extra_body"] == {"reasoning": {"effort": "minimal"}}
    monkeypatch.setattr(client, "BASE_URL", "https://api.x.ai/v1")
    client._call_api([], S, True, 5)
    assert sent["extra_body"] is None


def test_real_path_falls_back_to_json_object(monkeypatch):
    monkeypatch.setenv("GROK_MOCK", "0")
    modes = []

    def fake_api(content, schema, use_schema, timeout):
        modes.append(use_schema)
        if use_schema:
            raise openai.BadRequestError("json_schema non supportato", response=httpx.Response(400, request=REQ),
                                         body=None)
        assert "schema" in content[-1]["text"]  # senza output strutturato lo schema va nel prompt
        return '{"text": "ok"}', None

    monkeypatch.setattr(client, "_call_api", fake_api)
    r, meta = ask_json([], "p", S)
    assert r.text == "ok" and meta["mode"] == "json_object" and meta["attempts"] == 1
    assert modes == [True, False]
    ask_json([], "p", S)
    assert modes == [True, False, False]  # il ripiego viene ricordato


def test_missing_key_does_not_raise(monkeypatch):
    monkeypatch.setenv("GROK_MOCK", "0")
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setattr(client, "_client", None)
    r, meta = ask_json([], "p", S)
    assert r is None and "XAI_API_KEY" in meta["error"]


def test_log_written_without_images():
    ask_json([np.zeros((50, 50, 3), np.uint8)], "p", S, task="prova", mock=lambda: {"text": "ok"})
    line = json.loads(client.LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])
    assert line["task"] == "prova" and line["ok"] and line["response"] == {"text": "ok"}
    assert "base64" not in client.LOG_PATH.read_text(encoding="utf-8")
