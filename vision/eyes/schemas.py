"""Schemi Pydantic per Visione A ("Occhi").

Compatibili al 100% con il contratto di gruppo (`vision/grok/schemas.py`):
- `bbox` è sempre normalizzato `[x, y, w, h]` nell'intervallo `[0.0, 1.0]`
  dove `(x, y)` è l'angolo in alto a sinistra.
- I dizionari esportati da `EyeObject.to_group_dict()` o `model_dump()` possono
  essere passati direttamente a `vision.grok.schemas.parse_objects()`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class CoordsMM(BaseModel):
    """Coordinate cartesiane sul piano di lavoro del braccio (in millimetri) e angolo di presa."""
    x: float = Field(description="Coordinata X in millimetri nel riferimento del braccio")
    y: float = Field(description="Coordinata Y in millimetri nel riferimento del braccio")
    angle_deg: float = Field(default=0.0, ge=-90.0, le=90.0, description="Angolo di presa in gradi [-90, +90]")


class EyeObject(BaseModel):
    """Oggetto rilevato e tracciato da Visione A."""
    id: int
    label: str
    name: str | None = None
    conf: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]  # (x, y, w, h) normalizzati 0-1
    color: str | None = None
    coords_mm: CoordsMM | None = None

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x, y, w, h = v
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = max(0.0, min(1.0 - x, float(w)))
        h = max(0.0, min(1.0 - y, float(h)))
        return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))

    @property
    def center_norm(self) -> tuple[float, float]:
        """Centro normalizzato (cx, cy) in [0, 1]."""
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)


class DogTelemetry(BaseModel):
    """Valori per l'inseguimento visivo del cane robot."""
    target_id: int | None = None
    target_visible: bool = False
    error_x: float = Field(default=0.0, ge=-1.0, le=1.0, description="Errore orizzontale [-1=sinistra, 0=centro, +1=destra]")
    error_y: float = Field(default=0.0, ge=-1.0, le=1.0, description="Errore verticale [-1=alto, 0=centro, +1=basso]")
    bbox_height: float = Field(default=0.0, ge=0.0, le=1.0, description="Altezza normalizzata del riquadro (stima vicinanza)")


class FramePayload(BaseModel):
    """Messaggio JSON completo emesso a ogni frame per App, Robotica e Visione B."""
    source: str
    frame_id: int
    ts: float
    objects: list[EyeObject] = Field(default_factory=list)
    dog_telemetry: DogTelemetry | None = None
    safety_stop: bool = False
    safety_reason: str | None = None


class ZoneVerificationResult(BaseModel):
    """Risultato della misura oggettiva locale di verifica (sparito dalla zona / arrivato a destinazione)."""
    success: bool
    disappeared_from_source: bool
    arrived_at_destination: bool | None = None
    source_iou: float = 0.0
    destination_iou: float = 0.0
    reason: str
