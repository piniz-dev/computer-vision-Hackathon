"""Telemetria per l'inseguimento visivo del cane robot e Safety Stop per persone vicine."""
from __future__ import annotations

from typing import Sequence

from ..schemas import DogTelemetry, EyeObject

PERSON_LABELS = {"person", "persona", "pedestrian", "human"}


def _coerce_objects(objects: Sequence[EyeObject | dict]) -> list[EyeObject]:
    out: list[EyeObject] = []
    for o in objects:
        if isinstance(o, EyeObject):
            out.append(o)
        elif isinstance(o, dict):
            out.append(EyeObject.model_validate(o))
        else:
            out.append(EyeObject.model_validate(o.model_dump()))
    return out


def check_person_safety_stop(
    objects: Sequence[EyeObject | dict],
    conf_threshold: float = 0.50,
    height_threshold: float = 0.45,
    area_threshold: float = 0.20,
) -> tuple[bool, str | None]:
    """Verifica se una persona è troppo vicina alla telecamera del cane.

    Restituisce `(safety_stop, reason)`.
    """
    for obj in _coerce_objects(objects):
        if obj.label.lower() in PERSON_LABELS and obj.conf >= conf_threshold:
            _, _, bw, bh = obj.bbox
            area = bw * bh
            if bh >= height_threshold or area >= area_threshold:
                return (
                    True,
                    f"Persona vicina rilevata (id={obj.id}, h={bh:.2f}, area={area:.2f})",
                )
    return False, None


def compute_dog_telemetry(
    objects: Sequence[EyeObject | dict],
    target_id: int | None = None,
    target_label: str | None = None,
) -> DogTelemetry:
    """Calcola i parametri di visual servoing per il cane robot:
    - `target_visible`: True se l'oggetto target è nel frame
    - `error_x`: offset orizzontale normalizzato in `[-1.0, +1.0]` (`0.0` = centrato)
    - `error_y`: offset verticale normalizzato in `[-1.0, +1.0]`
    - `bbox_height`: altezza normalizzata `[0.0, 1.0]` come stima di vicinanza
    """
    parsed = _coerce_objects(objects)
    target: EyeObject | None = None

    if target_id is not None:
        target = next((o for o in parsed if o.id == target_id), None)
    elif target_label is not None:
        lbl = target_label.lower()
        candidates = [o for o in parsed if o.label.lower() == lbl or (o.name and lbl in o.name.lower())]
        if candidates:
            target = max(candidates, key=lambda o: o.conf)

    if target is None:
        return DogTelemetry(
            target_id=target_id,
            target_visible=False,
            error_x=0.0,
            error_y=0.0,
            bbox_height=0.0,
        )

    cx, cy = target.center_norm
    _, _, _, bh = target.bbox
    err_x = max(-1.0, min(1.0, (cx - 0.5) * 2.0))
    err_y = max(-1.0, min(1.0, (cy - 0.5) * 2.0))

    return DogTelemetry(
        target_id=target.id,
        target_visible=True,
        error_x=round(err_x, 4),
        error_y=round(err_y, 4),
        bbox_height=round(bh, 4),
    )


class DogNavigator:
    """Mantiene il target corrente del cane e calcola telemetria + safety stop a ogni frame."""

    def __init__(
        self,
        person_conf_threshold: float = 0.50,
        person_height_threshold: float = 0.45,
        person_area_threshold: float = 0.20,
    ):
        self.person_conf_threshold = person_conf_threshold
        self.person_height_threshold = person_height_threshold
        self.person_area_threshold = person_area_threshold
        self.target_id: int | None = None
        self.target_label: str | None = None

    def set_target(self, target_id: int | None = None, target_label: str | None = None) -> None:
        self.target_id = target_id
        self.target_label = target_label

    def evaluate(
        self,
        objects: Sequence[EyeObject | dict],
    ) -> tuple[DogTelemetry, bool, str | None]:
        """Restituisce `(telemetry, safety_stop, safety_reason)`."""
        safety_stop, reason = check_person_safety_stop(
            objects,
            conf_threshold=self.person_conf_threshold,
            height_threshold=self.person_height_threshold,
            area_threshold=self.person_area_threshold,
        )
        telemetry = compute_dog_telemetry(
            objects,
            target_id=self.target_id,
            target_label=self.target_label,
        )
        return telemetry, safety_stop, reason
