"""Disegno dei numeri sugli oggetti (Set-of-Mark) per Visione A ("Occhi").

Usato sia per il flusso video MJPEG dell'app sia per le chiamate multimodali a Grok (Visione B).
Mantiene la stessa tavolozza cromatica di `vision.grok.marks` così ogni ID ha sempre
lo stesso colore in tutto il sistema.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..depth.estimator import estimate_distance_m, format_distance
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

LABEL_IT: dict[str, str] = {
    "person": "persona",
    "dog": "cane",
    "cat": "gatto",
    "cup": "tazza",
    "bottle": "bottiglia",
    "wine glass": "bicchiere",
    "fork": "forchetta",
    "knife": "coltello",
    "spoon": "cucchiaio",
    "bowl": "ciotola",
    "chair": "sedia",
    "couch": "divano",
    "potted plant": "pianta",
    "bed": "letto",
    "dining table": "tavolo",
    "tv": "televisore",
    "laptop": "computer",
    "mouse": "mouse",
    "remote": "telecomando",
    "keyboard": "tastiera",
    "cell phone": "telefono",
    "book": "libro",
    "clock": "orologio",
    "scissors": "forbici",
    "pen": "penna",
    "pencil": "matita",
    "backpack": "zaino",
    "umbrella": "ombrello",
    "handbag": "borsa",
    "suitcase": "valigia",
    "sports ball": "palla",
    "box": "scatola",
}


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
    show_labels: bool = True,
    min_conf: float = 0.0,
    show_conf: bool = False,
    show_dist: bool = True,
) -> np.ndarray:
    """Restituisce una copia di `frame` con riquadri e badge numerici/etichette (Set-of-Mark).

    - `frame`: immagine BGR originale (non viene modificata in-place).
    - `objects`: lista di `EyeObject` o `dict` nel formato del gruppo.
    - `show_grasp`: se True e `coords_mm` è presente, disegna anche il centro e l'asse di presa.
    - `safety_stop`: se True, disegna un bordo rosso di allerta sicurezza.
    - `show_labels`: se True, mostra sia l'ID sia il nome/classe dell'oggetto (es. [1] persona).
    - `min_conf`: soglia minima di accuratezza/confidenza [0.0 - 1.0] per visualizzare il riquadro.
    - `show_conf`: se True, mostra la percentuale di confidenza (es. [1] persona 87%).
    - `show_dist`: se True, mostra la distanza approssimativa accanto al label (es. ~45cm o ~1.8m).
    """
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    scale = max(0.55, min(h, w) / 700)

    for o in _coerce_objects(objects):
        if o.conf < min_conf:
            continue

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

        # Testo del badge: ID + etichetta riconoscibile + eventuale confidenza + distanza
        conf_str = f" {int(round(o.conf * 100))}%" if show_conf else ""
        dist_val = o.distance_m if o.distance_m is not None else estimate_distance_m(o.label, o.bbox)
        dist_str = f" {format_distance(dist_val)}" if (show_dist and dist_val) else ""
        if show_labels:
            if o.name:
                lbl = o.name
            else:
                base_lbl = LABEL_IT.get(o.label.lower(), o.label.replace("_", " "))  # "tappo_bianco" -> "tappo bianco"
                if o.color and o.color not in base_lbl:
                    lbl = f"{base_lbl} {o.color}"
                else:
                    lbl = base_lbl
            text = f"[{o.id}] {lbl}{conf_str}{dist_str}"
        else:
            text = f"{o.id}{conf_str}{dist_str}"

        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        ty = y1 if y1 - th - base - 6 < 0 else y1 - th - base - 6
        bx2 = min(w - 1, x1 + tw + 8)
        by2 = min(h - 1, ty + th + base + 6)
        cv2.rectangle(out, (x1, ty), (bx2, by2), color, -1)

        b, g, r = color
        ink = (0, 0, 0) if 0.114 * b + 0.587 * g + 0.299 * r > 140 else (255, 255, 255)
        cv2.putText(
            out,
            text,
            (x1 + 4, ty + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            ink,
            max(1, thick - 1),
            cv2.LINE_AA,
        )

    if safety_stop:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), max(4, thick * 2))

    return out
