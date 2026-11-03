"""Schemi pydantic: oggetti in ingresso (formato deciso dal gruppo) e risposte dei compiti."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class TrackedObject(BaseModel):
    """Un oggetto riconosciuto da Visione A."""
    id: int
    label: str                      # classe YOLO, es. "cup"
    name: str | None = None         # nome descrittivo, es. "tazza rossa con manico"
    conf: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]  # x, y, w, h normalizzati 0-1 (x, y = angolo in alto a sinistra)
    color: str | None = None


class Detections(BaseModel):
    """Messaggio completo di un frame: {"source", "frame_id", "ts", "objects"}."""
    source: str                     # "arm" | "dog" | ...
    frame_id: int
    ts: float                       # unix time in secondi
    objects: list[TrackedObject]


def parse_objects(objects: list) -> list[TrackedObject]:
    """Accetta dict o TrackedObject e restituisce sempre TrackedObject validati."""
    return [TrackedObject.model_validate(o) for o in objects]


# ---------------------------------------------------------------------------
# Risposte dei compiti
#
# Ogni risultato è UN solo modello: i campi "di Grok" finiscono nello schema
# JSON mandato all'API; ok/error/meta li aggiungiamo noi e Grok non li vede
# (SkipJsonSchema). Così il chiamante legge sempre res.ok, res.target_id, ...
# ---------------------------------------------------------------------------

def _strict_schema(schema: dict) -> None:
    # Output strutturato: tutti i campi obbligatori (i null sono espliciti) e niente "default".
    schema["required"] = list(schema.get("properties", {}))
    for prop in schema.get("properties", {}).values():
        prop.pop("default", None)


class GrokModel(BaseModel):
    model_config = ConfigDict(json_schema_extra=_strict_schema)


class Result(GrokModel):
    ok: SkipJsonSchema[bool] = True
    error: SkipJsonSchema[str | None] = None   # motivo quando ok=False
    meta: SkipJsonSchema[dict] = Field(default_factory=dict)  # latenza, tentativi, ... (vedi client.ask_json)


class ObjectName(GrokModel):
    id: int
    name: str              # 2-5 parole, con colore e attributi distintivi
    color: str | None


class NamesResult(Result):
    objects: list[ObjectName] = []


Action = Literal["pick", "place", "point", "go_to", "follow", "look", "none"]


class CommandResult(Result):
    target_id: int | None = None
    destination_id: int | None = None
    action: Action = "none"
    ambiguous: bool = False
    question: str | None = None    # domanda breve da far pronunciare alla voce se ambiguous
    confidence: float = Field(0.0, ge=0, le=1)
    reason: str = ""
    source: SkipJsonSchema[Literal["local", "grok"]] = "grok"  # da quale strada è arrivata la risposta


class SceneResult(Result):
    text: str = ""                 # massimo 2 frasi, da leggere a voce


class DetectedObject(GrokModel):
    """Oggetto trovato da Grok. Stessi campi di TrackedObject: l'id lo assegniamo noi."""
    id: SkipJsonSchema[int] = 0
    label: str                     # classe breve in inglese, minuscola (stile COCO)
    name: str                      # nome italiano di 2-5 parole
    color: str | None
    conf: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)  # x, y, w, h normalizzati 0-1


class DetectResult(Result):
    objects: list[DetectedObject] = []


class VerifyResult(Result):
    success: bool = False
    reason: str = ""
    suggestion: str | None = None
    confidence: float = Field(0.0, ge=0, le=1)
