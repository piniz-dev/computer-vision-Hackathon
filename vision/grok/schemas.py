"""Schemi pydantic: formato degli oggetti in ingresso (deciso dal gruppo, non cambiarlo)."""
from __future__ import annotations

from pydantic import BaseModel, Field


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
