"""Stima monoculare della distanza (Monocular Depth / Pinhole Size-Prior Estimator).

Calcola la distanza approssimativa in metri tra la fotocamera e ciascun oggetto
rilevato combinando:
1. Modello Pinhole (dimensione apparente normalizzata vs dimensione reale tipica della classe).
2. Euristica busto/spalle vs corpo intero per le persone davanti alla webcam.
3. Filtro EMA (Exponential Moving Average) per ciascun ID tracciato così da
   evitare oscillazioni numeriche tra un fotogramma e l'altro.
"""
from __future__ import annotations

from typing import Sequence

# Dimensioni reali medie (altezza_cm, larghezza_cm) per classi COCO e comuni
CLASS_PRIORS_CM: dict[str, tuple[float, float]] = {
    "person": (170.0, 48.0),
    "persona": (170.0, 48.0),
    "bottle": (24.0, 7.0),
    "bottiglia": (24.0, 7.0),
    "cup": (10.5, 8.0),
    "tazza": (10.5, 8.0),
    "wine glass": (18.0, 7.5),
    "bicchiere": (18.0, 7.5),
    "cell phone": (15.0, 7.5),
    "telefono": (15.0, 7.5),
    "laptop": (24.0, 34.0),
    "computer": (24.0, 34.0),
    "keyboard": (14.0, 38.0),
    "tastiera": (14.0, 38.0),
    "mouse": (11.0, 6.5),
    "remote": (16.0, 4.5),
    "telecomando": (16.0, 4.5),
    "book": (23.0, 16.0),
    "libro": (23.0, 16.0),
    "chair": (85.0, 50.0),
    "sedia": (85.0, 50.0),
    "couch": (85.0, 180.0),
    "divano": (85.0, 180.0),
    "potted plant": (35.0, 25.0),
    "pianta": (35.0, 25.0),
    "tv": (55.0, 95.0),
    "televisore": (55.0, 95.0),
    "backpack": (45.0, 32.0),
    "zaino": (45.0, 32.0),
    "handbag": (30.0, 35.0),
    "borsa": (30.0, 35.0),
    "suitcase": (60.0, 40.0),
    "valigia": (60.0, 40.0),
    "dog": (50.0, 35.0),
    "cane": (50.0, 35.0),
    "cat": (28.0, 20.0),
    "gatto": (28.0, 20.0),
    "bowl": (8.0, 16.0),
    "ciotola": (8.0, 16.0),
    "scissors": (17.0, 7.0),
    "forbici": (17.0, 7.0),
    "clock": (25.0, 25.0),
    "orologio": (25.0, 25.0),
    "vase": (25.0, 14.0),
    "vaso": (25.0, 14.0),
    "apple": (8.0, 8.0),
    "banana": (18.0, 4.5),
    "orange": (8.0, 8.0),
    "sports ball": (22.0, 22.0),
    "palla": (22.0, 22.0),
    "pen": (14.0, 1.5),
    "penna": (14.0, 1.5),
    "pencil": (15.0, 1.0),
    "matita": (15.0, 1.0),
    "fork": (19.0, 2.5),
    "knife": (21.0, 2.5),
    "spoon": (18.0, 3.5),
    "box": (16.0, 16.0),
    "scatola": (16.0, 16.0),
    "square": (14.0, 14.0),
    "rectangle": (18.0, 12.0),
}

DEFAULT_PRIOR_CM: tuple[float, float] = (18.0, 15.0)

# Focali normalizzate tipiche per una webcam 16:9 / 4:3 (FOV ~60-65 gradi)
FOCAL_NORM_Y: float = 1.15
FOCAL_NORM_X: float = 0.86


def estimate_distance_m(
    label: str,
    bbox: Sequence[float],
    focal_norm_y: float = FOCAL_NORM_Y,
    focal_norm_x: float = FOCAL_NORM_X,
) -> float:
    """Stima la distanza in metri dalla camera dato il `label` e la `bbox` `[x, y, w, h]` normalizzata."""
    if len(bbox) != 4:
        return 1.0

    x, y, bw, bh = [float(v) for v in bbox]
    bw = max(0.01, min(1.0, bw))
    bh = max(0.01, min(1.0, bh))

    clean_label = (label or "object").strip().lower()
    real_h_cm, real_w_cm = CLASS_PRIORS_CM.get(clean_label, DEFAULT_PRIOR_CM)

    # Caso speciale: persona inquadrata a mezzo busto / primo piano da webcam da scrivania
    if clean_label in ("person", "persona"):
        touches_bottom = (y + bh) >= 0.88
        aspect = bh / bw
        if touches_bottom and aspect < 1.85:
            # Inquadratura mezzo busto / spalle: usiamo la larghezza spalle (~46 cm) e altezza busto (~60 cm)
            dist_from_w = (focal_norm_x * 0.46) / bw
            dist_from_h = (focal_norm_y * 0.60) / bh
            dist_m = 0.6 * dist_from_w + 0.4 * dist_from_h
            return round(max(0.20, min(12.0, dist_m)), 2)

    # Per oggetti più larghi che alti (es. laptop, tastiera, divano) pesiamo maggiormente la larghezza
    if real_w_cm > real_h_cm * 1.25:
        dist_m = (focal_norm_x * (real_w_cm / 100.0)) / bw
    else:
        dist_h = (focal_norm_y * (real_h_cm / 100.0)) / bh
        dist_w = (focal_norm_x * (real_w_cm / 100.0)) / bw
        dist_m = 0.75 * dist_h + 0.25 * dist_w

    return round(max(0.15, min(15.0, dist_m)), 2)


def format_distance(dist_m: float | None) -> str:
    """Formatta la distanza in modo compatto per il badge (es. `~45cm` oppure `~1.8m`)."""
    if dist_m is None or dist_m <= 0:
        return ""
    if dist_m < 1.0:
        cm = int(round(dist_m * 100 / 5.0) * 5)  # arrotonda a step di 5 cm per stabilità visiva
        cm = max(10, cm)
        return f"~{cm}cm"
    return f"~{dist_m:.1f}m"


class DepthSmoother:
    """Mantiene una media mobile esponenziale (EMA) delle distanze per ciascun ID oggetto."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = max(0.05, min(1.0, alpha))
        self._history: dict[int, float] = {}

    def update(self, obj_id: int, label: str, bbox: Sequence[float]) -> float:
        raw_d = estimate_distance_m(label, bbox)
        if obj_id not in self._history:
            self._history[obj_id] = raw_d
            return raw_d
        prev = self._history[obj_id]
        smoothed = round(self.alpha * raw_d + (1.0 - self.alpha) * prev, 2)
        self._history[obj_id] = smoothed
        return smoothed

    def reset(self) -> None:
        self._history.clear()
