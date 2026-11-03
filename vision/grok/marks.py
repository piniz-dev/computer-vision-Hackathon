"""Disegno Set-of-Mark semplice: riquadro + id su ogni oggetto.

Visione A ha la sua versione per lo stream; questa serve per test e dataset.
"""
from __future__ import annotations

import cv2
import numpy as np

from .schemas import parse_objects

# Colori BGR ben distinguibili, scelti in base all'id (stesso id -> stesso colore).
_PALETTE = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255),
            (255, 255, 0), (0, 128, 255), (128, 0, 255), (0, 255, 128), (255, 128, 0)]


def draw_marks(frame: np.ndarray, objects: list) -> np.ndarray:
    """Restituisce una copia di `frame` con riquadro e numero (id) per ogni oggetto."""
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.6, min(h, w) / 600)
    for o in parse_objects(objects):
        x, y, bw, bh = o.bbox
        x1, y1 = int(x * w), int(y * h)
        x2, y2 = int((x + bw) * w), int((y + bh) * h)
        color = _PALETTE[o.id % len(_PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # Etichetta con l'id su sfondo pieno, nell'angolo in alto a sinistra del riquadro
        # (dentro l'immagine anche se l'oggetto tocca il bordo).
        text = str(o.id)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 if y1 - th - base - 4 < 0 else y1 - th - base - 4
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + base + 4), color, -1)
        b, g, r = color
        ink = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)  # testo leggibile
        cv2.putText(out, text, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thick, cv2.LINE_AA)
    return out
