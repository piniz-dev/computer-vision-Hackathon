"""Client Grok (xAI) tramite SDK OpenAI.

Usa l'endpoint Chat Completions di xAI (compatibile OpenAI). Cosa è stato
verificato sulla documentazione è scritto nel README.

Configurazione da `.env`:
  XAI_API_KEY   chiave (obbligatoria se non in modalità finta)
  XAI_BASE_URL  endpoint compatibile OpenAI (default xAI; per OpenRouter https://openrouter.ai/api/v1)
  XAI_MODEL     modello con input immagini (default grok-4.7; su OpenRouter x-ai/grok-4.7)
  GROK_MOCK=1   modalità finta: nessuna chiamata di rete
  GROK_TIMEOUT  secondi massimi per chiamata (default 8)
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import openai
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError

# Carica .env dalla root del progetto (o directory superiori)
_env_file = find_dotenv(usecwd=True)
if not _env_file:
    _candidate = Path(__file__).resolve().parents[2] / ".env"
    _env_file = str(_candidate) if _candidate.exists() else None
load_dotenv(_env_file)

BASE_URL = os.getenv("XAI_BASE_URL") or "https://api.x.ai/v1"
DEFAULT_MODEL = os.getenv("XAI_MODEL") or "grok-4.7"
DEFAULT_TIMEOUT = float(os.getenv("GROK_TIMEOUT", "8.0"))
LOG_PATH = Path("logs/grok_calls.jsonl")

# Se l'API rifiuta json_schema passiamo a json_object e lo ricordiamo
# per le chiamate successive.
_schema_supported = True
_client: openai.OpenAI | None = None


def is_mock() -> bool:
    return os.getenv("GROK_MOCK") == "1"


def encode_image(img_bgr: np.ndarray, max_side: int = 1024, quality: int = 85) -> str:
    """Ridimensiona (lato lungo <= max_side, proporzioni mantenute), JPEG, base64 -> data URL."""
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("codifica JPEG fallita")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY mancante (vedi .env.example)")
        # max_retries=0: i tentativi li gestiamo noi, così il tempo massimo resta sotto controllo.
        _client = openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=0)
    return _client


def _call_api(content: list, schema: type[BaseModel], use_schema: bool, timeout: float) -> tuple[str, dict | None]:
    """Una chiamata reale. Restituisce (testo della risposta, token)."""
    if use_schema:
        fmt = {"type": "json_schema",
               "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}
    else:
        fmt = {"type": "json_object"}
    resp = _get_client().chat.completions.create(
        model=os.getenv("XAI_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": content}],
        response_format=fmt,
        timeout=timeout,
    )
    usage = resp.usage
    tokens = {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens} if usage else None
    return resp.choices[0].message.content or "", tokens


def _log(entry: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # il log non deve mai bloccare il robot


def ask_json(
    images: list,
    prompt: str,
    schema: type[BaseModel],
    timeout: float = DEFAULT_TIMEOUT,
    task: str = "generic",
    mock: Callable[[], str | dict] | None = None,
) -> tuple[BaseModel | None, dict]:
    """Chiede a Grok una risposta JSON conforme a `schema`. Non solleva mai eccezioni.

    images: immagini BGR (numpy) oppure data URL già pronti.
    task:   nome del compito, solo per il log.
    mock:   in modalità finta (GROK_MOCK=1) viene chiamata al posto dell'API;
            restituisce il testo JSON (o un dict) che "risponderebbe" Grok.
            Se solleva TimeoutError simula un timeout.

    Restituisce (modello validato | None, meta) con meta =
    {latency_ms, tokens, attempts, error, mode, mock}.
    """
    global _schema_supported
    t0 = time.perf_counter()
    meta: dict[str, Any] = {"latency_ms": 0, "tokens": None, "attempts": 0, "error": None,
                            "mode": None, "mock": is_mock()}
    result: BaseModel | None = None
    raw: str | None = None

    try:
        content = [{"type": "image_url", "image_url": {"url": im if isinstance(im, str) else encode_image(im)}}
                   for im in images]
    except Exception as e:  # immagine non valida
        content = None
        meta["error"] = f"immagine non valida: {e}"

    while content is not None and meta["attempts"] < 2:  # 1 tentativo + 1 nuovo tentativo
        meta["attempts"] += 1
        use_schema = _schema_supported
        meta["mode"] = "json_schema" if use_schema else "json_object"
        text = prompt
        if not use_schema:  # senza output strutturato lo schema va nel prompt
            text += "\n\nRispondi SOLO con un oggetto JSON conforme a questo schema:\n" + json.dumps(
                schema.model_json_schema(), ensure_ascii=False)
        try:
            if meta["mock"]:
                if mock is None:
                    meta["error"] = "nessuna risposta finta definita per questo compito"
                    break
                out = mock()
                raw = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
            else:
                raw, meta["tokens"] = _call_api(content + [{"type": "text", "text": text}], schema, use_schema, timeout)
            result = schema.model_validate_json(raw)
            meta["error"] = None
            break
        except (openai.APITimeoutError, TimeoutError):
            meta["error"] = "timeout"
        except ValidationError as e:
            meta["error"] = f"JSON non valido: {e.errors()[0]['msg']}"
        except openai.BadRequestError as e:
            if use_schema:  # probabilmente json_schema non supportato: ripiego senza contare il tentativo
                _schema_supported = False
                meta["attempts"] -= 1
                meta["error"] = f"json_schema rifiutato, passo a json_object: {e}"
                continue
            meta["error"] = f"richiesta rifiutata: {e}"
            break
        except Exception as e:  # rete, chiave, ecc.: inutile riprovare
            meta["error"] = f"{type(e).__name__}: {e}"
            break

    meta["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _log({"ts": time.time(), "task": task, "latency_ms": meta["latency_ms"], "ok": result is not None,
          "attempts": meta["attempts"], "mode": meta["mode"], "mock": meta["mock"], "error": meta["error"],
          "tokens": meta["tokens"], "response": result.model_dump() if result else raw})
    return result, meta
