"""Strumenti (function tools) per la sessione realtime di Grok Voice.

Per il gruppo app:
  1. manda TOOLS in `session.update` -> session.tools
  2. a ogni evento `response.function_call_arguments.done` chiama
       result = dispatch(event["name"], event["arguments"], state)
     e rispondi con `conversation.item.create`:
       {"type": "function_call_output", "call_id": event["call_id"], "output": json.dumps(result)}
     poi `response.create` (dopo aver risposto a TUTTE le chiamate in parallelo).
  Ogni risultato ha "ok" e "say": la frase breve che la voce deve dire.

`state` è un VisionState: lo implementa Visione A. FakeVisionState legge da cartella.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from .local_resolver import display
from .marks import draw_marks
from .schemas import Detections, parse_objects
from .tasks import describe_scene, select, verify_action

_SOURCE = {"type": "string", "enum": ["arm", "dog"],
           "description": "Telecamera da usare: arm = braccio robotico, dog = cane robot."}

# Formato Grok Voice: tool "piatti" (type/name/description/parameters).
TOOLS = [
    {"type": "function", "name": "list_objects",
     "description": "Elenca gli oggetti visti dalla telecamera, ognuno con il suo numero.",
     "parameters": {"type": "object", "properties": {"source": _SOURCE}, "required": ["source"]}},
    {"type": "function", "name": "select_object",
     "description": "Seleziona l'oggetto descritto dall'utente (nome, colore, posizione o numero). "
                    "Se è ambiguo restituisce una domanda da fare all'utente.",
     "parameters": {"type": "object", "properties": {
         "source": _SOURCE,
         "description": {"type": "string", "description": "La richiesta dell'utente, con le sue parole."}},
         "required": ["source", "description"]}},
    {"type": "function", "name": "describe_scene",
     "description": "Descrive brevemente cosa vede la telecamera.",
     "parameters": {"type": "object", "properties": {"source": _SOURCE}, "required": ["source"]}},
    {"type": "function", "name": "verify_last_action",
     "description": "Controlla se l'ultima azione del robot è riuscita.",
     "parameters": {"type": "object", "properties": {"source": _SOURCE}, "required": ["source"]}},
]


class VisionState(ABC):
    """Stato della visione, implementato da Visione A."""

    @abstractmethod
    def get_frame(self, source: str) -> np.ndarray:
        """Ultimo frame originale (BGR)."""

    @abstractmethod
    def get_marked_frame(self, source: str) -> np.ndarray:
        """Ultimo frame con riquadri e numeri (Set-of-Mark)."""

    @abstractmethod
    def get_objects(self, source: str) -> list:
        """Oggetti dell'ultimo frame, nel formato del gruppo (lista di dict)."""

    @abstractmethod
    def get_last_action(self, source: str) -> dict | None:
        """Ultima azione del robot o None:
        {"action": "pick", "target_id": 3, "destination_id": None,
         "frame_before": <BGR>, "objects_before": [...]}.
        Lo stato "dopo" è quello attuale (get_frame / get_objects)."""


# Contesto della conversazione per sorgente: ultimo oggetto selezionato e ultima domanda.
# ponytail: un solo dialogo per processo; se servono più utenti va spostato nello state.
_context: dict[str, dict] = {}


def dispatch(tool_name: str, args: dict | str, state: VisionState) -> dict:
    """Esegue lo strumento richiesto. Non solleva mai eccezioni."""
    try:
        if isinstance(args, str):  # Grok Voice manda gli argomenti come stringa JSON
            args = json.loads(args or "{}")
        source = args.get("source", "arm")
        if tool_name == "list_objects":
            return _list_objects(source, state)
        if tool_name == "select_object":
            return _select_object(source, args.get("description", ""), state)
        if tool_name == "describe_scene":
            return _describe_scene(source, state)
        if tool_name == "verify_last_action":
            return _verify_last_action(source, state)
        return {"ok": False, "error": f"strumento sconosciuto: {tool_name}", "say": "Non so fare questa operazione."}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "say": "Ho avuto un problema, riprova tra poco."}


def _list_objects(source: str, state: VisionState) -> dict:
    objs = parse_objects(state.get_objects(source))
    if not objs:
        return {"ok": True, "objects": [], "say": "Non vedo nessun oggetto."}
    items = [{"id": o.id, "name": display(o)} for o in objs]
    listed = ", ".join(f"{i['name']} numero {i['id']}" for i in items)
    return {"ok": True, "objects": items, "say": f"Vedo {len(items)} oggetti: {listed}."}


def _select_object(source: str, description: str, state: VisionState) -> dict:
    objects = state.get_objects(source)
    ctx = _context.setdefault(source, {"last_selected_id": None, "last_question": None})
    res = select(state.get_marked_frame(source), objects, description, ctx)
    out = {"source": res.source, "latency_ms": res.meta.get("latency_ms")}
    if not res.ok:
        return out | {"ok": False, "id": None, "error": res.error, "say": "Non ho capito bene, puoi ripetere?"}
    if res.ambiguous:
        ctx["last_question"] = res.question
        return out | {"ok": False, "id": None, "ambiguous": True, "say": res.question}
    if res.target_id is None:
        return out | {"ok": False, "id": None, "say": f"Non lo trovo. {res.reason}".strip()}
    ctx.update(last_selected_id=res.target_id, last_question=None)
    obj = next(o for o in parse_objects(objects) if o.id == res.target_id)
    return out | {"ok": True, "id": res.target_id, "destination_id": res.destination_id, "action": res.action,
                  "say": f"Ho selezionato {display(obj)}."}


def _describe_scene(source: str, state: VisionState) -> dict:
    res = describe_scene(state.get_frame(source), state.get_objects(source))
    if not res.ok:
        return {"ok": False, "error": res.error, "say": "Non riesco a descrivere la scena adesso."}
    return {"ok": True, "say": res.text}


def _verify_last_action(source: str, state: VisionState) -> dict:
    last = state.get_last_action(source)
    if not last:
        return {"ok": False, "say": "Non ho ancora fatto nessuna azione."}
    res = verify_action(last["frame_before"], state.get_frame(source), last["objects_before"],
                        state.get_objects(source), last["action"], last["target_id"], last.get("destination_id"))
    if not res.ok:
        return {"ok": False, "error": res.error, "say": "Non riesco a controllare adesso."}
    say = ("Fatto, è riuscito." if res.success else f"Non è riuscito: {res.reason}") + (
        f" {res.suggestion}" if res.suggestion and not res.success else "")
    return {"ok": True, "success": res.success, "say": say}


class FakeVisionState(VisionState):
    """Legge da una cartella, per lavorare senza robot:
      <source>.jpg + <source>.json          frame e oggetti attuali (formato del gruppo)
      <source>_marked.jpg                   facoltativo, altrimenti disegnato con draw_marks
      <source>_before.jpg + _before.json    stato prima dell'ultima azione   } facoltativi,
      <source>_action.json                  {"action", "target_id", "destination_id"} } per la verifica
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def _img(self, name: str) -> np.ndarray:
        img = cv2.imread(str(self.folder / name))
        if img is None:
            raise FileNotFoundError(self.folder / name)
        return img

    def _objects(self, name: str) -> list:
        msg = Detections.model_validate_json((self.folder / name).read_text(encoding="utf-8"))
        return [o.model_dump() for o in msg.objects]

    def get_frame(self, source):
        return self._img(f"{source}.jpg")

    def get_marked_frame(self, source):
        if (self.folder / f"{source}_marked.jpg").exists():
            return self._img(f"{source}_marked.jpg")
        return draw_marks(self.get_frame(source), self.get_objects(source))

    def get_objects(self, source):
        return self._objects(f"{source}.json")

    def get_last_action(self, source):
        path = self.folder / f"{source}_action.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")) | {
            "frame_before": self._img(f"{source}_before.jpg"), "objects_before": self._objects(f"{source}_before.json")}
