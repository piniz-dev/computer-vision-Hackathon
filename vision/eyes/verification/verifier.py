"""Misura oggettiva locale della verifica per Visione A ("Occhi").

Risponde in modo deterministico e istantaneo alle due domande chiave:
1. L'oggetto è sparito dalla zona di partenza (pick)?
2. L'oggetto è arrivato nella zona di destinazione (place)?
"""
from __future__ import annotations

from typing import Sequence

from ..schemas import EyeObject, FramePayload, ZoneVerificationResult
from ..tracking.tracker import compute_iou


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


def point_in_bbox(
    pt: tuple[float, float],
    zone_bbox: tuple[float, float, float, float],
    margin: float = 0.02,
) -> bool:
    """Restituisce True se il punto `(cx, cy)` cade dentro `zone_bbox = (x, y, w, h)` (+/- `margin`)."""
    px, py = pt
    zx, zy, zw, zh = zone_bbox
    return (zx - margin) <= px <= (zx + zw + margin) and (zy - margin) <= py <= (zy + zh + margin)


def check_disappearance(
    zone_bbox: tuple[float, float, float, float],
    objects_current: Sequence[EyeObject | dict],
    target_id: int | None = None,
    expected_label: str | None = None,
    iou_threshold: float = 0.25,
) -> tuple[bool, float]:
    """Verifica se l'oggetto è sparito dalla `zone_bbox`.

    Restituisce `(disappeared: bool, max_iou_in_zone: float)`.
    """
    objs = _coerce_objects(objects_current)
    max_iou = 0.0
    still_present = False

    for o in objs:
        id_match = target_id is not None and o.id == target_id
        label_match = expected_label is not None and o.label.lower() == expected_label.lower()
        if target_id is not None or expected_label is not None:
            if not (id_match or label_match):
                continue

        iou = compute_iou(o.bbox, zone_bbox)
        max_iou = max(max_iou, iou)
        if iou >= iou_threshold or point_in_bbox(o.center_norm, zone_bbox, margin=0.0):
            still_present = True

    return (not still_present), round(max_iou, 4)


def check_appearance(
    zone_bbox: tuple[float, float, float, float],
    objects_current: Sequence[EyeObject | dict],
    target_id: int | None = None,
    expected_label: str | None = None,
    exclude_id: int | None = None,
    iou_threshold: float = 0.15,
    margin: float = 0.04,
) -> tuple[bool, float]:
    """Verifica se l'oggetto è arrivato nella `zone_bbox` di destinazione.

    Restituisce `(arrived: bool, max_iou_in_zone: float)`.
    """
    objs = _coerce_objects(objects_current)
    max_iou = 0.0
    arrived = False

    for o in objs:
        if exclude_id is not None and o.id == exclude_id:
            continue
        id_match = target_id is not None and o.id == target_id
        label_match = expected_label is not None and o.label.lower() == expected_label.lower()
        if target_id is not None or expected_label is not None:
            if not (id_match or label_match):
                continue

        iou = compute_iou(o.bbox, zone_bbox)
        max_iou = max(max_iou, iou)
        if iou >= iou_threshold or point_in_bbox(o.center_norm, zone_bbox, margin=margin):
            arrived = True

    return arrived, round(max_iou, 4)


def verify_action_local(
    objects_before: Sequence[EyeObject | dict],
    objects_after: Sequence[EyeObject | dict],
    action: str,
    target_id: int,
    destination_id: int | None = None,
    source_zone_bbox: tuple[float, float, float, float] | None = None,
    dest_zone_bbox: tuple[float, float, float, float] | None = None,
) -> ZoneVerificationResult:
    """Esegue la verifica oggettiva confrontando lo stato prima e dopo l'azione."""
    before = _coerce_objects(objects_before)
    after = _coerce_objects(objects_after)

    target_before = next((o for o in before if o.id == target_id), None)
    if target_before is None and source_zone_bbox is None:
        return ZoneVerificationResult(
            success=False,
            disappeared_from_source=False,
            arrived_at_destination=None,
            reason=f"Oggetto target id={target_id} non presente nello stato iniziale.",
        )

    src_zone = source_zone_bbox or target_before.bbox  # type: ignore[union-attr]
    expected_label = target_before.label if target_before else None

    disappeared, src_iou = check_disappearance(
        src_zone,
        after,
        target_id=target_id,
        expected_label=expected_label,
    )

    # Trova eventuale zona di destinazione
    if dest_zone_bbox is None and destination_id is not None:
        dest_before = next((o for o in before if o.id == destination_id), None)
        if dest_before is not None:
            dest_zone_bbox = dest_before.bbox

    arrived: bool | None = None
    dst_iou = 0.0
    if dest_zone_bbox is not None:
        arrived, dst_iou = check_appearance(
            dest_zone_bbox,
            after,
            target_id=target_id,
            expected_label=expected_label,
            exclude_id=destination_id,
        )

    act = action.lower()
    if act == "pick":
        success = disappeared
        reason = (
            f"L'oggetto #{target_id} è stato rimosso dalla zona di partenza."
            if success
            else f"L'oggetto #{target_id} si trova ancora nella zona di partenza (IoU={src_iou:.2f})."
        )
    elif act == "place" or dest_zone_bbox is not None:
        success = bool(disappeared and arrived)
        if success:
            reason = f"L'oggetto #{target_id} ha lasciato la partenza ed è arrivato nella destinazione."
        elif not disappeared:
            reason = f"L'oggetto #{target_id} non si è mosso dalla zona di partenza."
        else:
            reason = f"L'oggetto #{target_id} è sparito dalla partenza ma non è nella destinazione."
    else:
        success = disappeared
        reason = "Verifica completata." if success else "Nessun cambiamento rilevato nella zona."

    return ZoneVerificationResult(
        success=success,
        disappeared_from_source=disappeared,
        arrived_at_destination=arrived,
        source_iou=src_iou,
        destination_iou=dst_iou,
        reason=reason,
    )


def verify_over_history(
    history: Sequence[FramePayload],
    zone_bbox: tuple[float, float, float, float],
    target_id: int | None = None,
    expected_label: str | None = None,
    min_frames: int = 3,
    mode: str = "disappear",
) -> bool:
    """Verifica stabile su più frame consecutivi (anti-sfarfallio)."""
    if not history:
        return False
    recent = list(history)[-min_frames:]
    matches = 0
    for payload in recent:
        if mode == "disappear":
            ok, _ = check_disappearance(zone_bbox, payload.objects, target_id=target_id, expected_label=expected_label)
        else:
            ok, _ = check_appearance(zone_bbox, payload.objects, target_id=target_id, expected_label=expected_label)
        if ok:
            matches += 1
    return matches >= min(len(recent), min_frames)
