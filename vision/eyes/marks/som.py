"""Disegno dei numeri sugli oggetti (Set-of-Mark) per Visione A ("Occhi").

Usato sia per il flusso video MJPEG dell'app sia per le chiamate multimodali a Grok (Visione B).
Mantiene la stessa tavolozza cromatica di `vision.grok.marks` così ogni ID ha sempre
lo stesso colore in tutto il sistema.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..schemas import EyeObject

# Tavolozza BGR coerente con vision.grok.marks
PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 255),
    (0, 200, 0),
    (255, 0, 0),
    (0, 200, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (0, 255, 128),
    (255, 128, 0),
]


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


def draw_marks(
    frame: np.ndarray,
    objects: Sequence[EyeObject | dict],
    show_grasp: bool = False,
    safety_stop: bool = False,
) -> np.ndarray:
    """Restituisce una copia di `frame` con riquadri e badge numerici (Set-of-Mark).

    - `frame`: immagine BGR originale (non viene modificata in-place).
    - `objects`: lista di `EyeObject` o `dict` nel formato del gruppo.
    - `show_grasp`: se True e `coords_mm` è presente, disegna anche il centro e l'asse di presa.
    - `safety_stop`: se True, disegna un bordo rosso di allerta sicurezza.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.6, min(h, w) / 600)

    for o in _coerce_objects(objects):
        x, y, bw, bh = o.bbox
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        color = PALETTE[o.id % len(PALETTE)]

        # Riquadro dell'oggetto
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # Opzionale: indicatore centro e angolo di presa per il braccio
        if show_grasp and o.coords_mm is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            rad = np.deg2rad(o.coords_mm.angle_deg)
            arm_len = max(12, min(x2 - x1, y2 - y1) // 3)
            dx = int(arm_len * np.cos(rad))
            dy = int(arm_len * np.sin(rad))
            cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), color, max(1, thick - 1), cv2.LINE_AA)
            cv2.circle(out, (cx, cy), max(3, thick + 1), color, -1)

        # Badge Set-of-Mark con solo il numero ID ad alto contrasto
        text = str(o.id)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 if y1 - th - base - 4 < 0 else y1 - th - base - 4
        bx2 = min(w - 1, x1 + tw + 6)
        by2 = min(h - 1, ty + th + base + 4)
        cv2.rectangle(out, (x1, ty), (bx2, by2), color, -1)

        b, g, r = color
        ink = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)
        cv2.putText(
            out,
            text,
            (x1 + 3, ty + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            ink,
            thick,
            cv2.LINE_AA,
        )

    if safety_stop:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), max(4, thick * 2))

    return out
