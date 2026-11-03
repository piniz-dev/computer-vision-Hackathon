"""Modulo per la gestione dello zoom digitale fluido (Digital Zoom / ROI Crop).

Permette di:
1. Zoomare su un punto specifico (cx, cy) normalizzato [0, 1] o su un oggetto (bbox).
2. Effettuare transizioni fluide tra il crop attuale e il crop target tramite interpolazione (Lerp).
3. Mappare coordinate tra frame originale (panoramico) e frame zoomato.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence
import numpy as np
import cv2


@dataclass
class CropWindow:
    """Rappresenta una finestra di crop con coordinate normalizzate [0.0, 1.0]."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0

    def width(self) -> float:
        return max(1e-6, self.x2 - self.x1)

    def height(self) -> float:
        return max(1e-6, self.y2 - self.y1)

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clamp(self) -> "CropWindow":
        """Assicura che la finestra rimanga nei limiti [0, 1]."""
        w = min(1.0, max(0.05, self.width()))
        h = min(1.0, max(0.05, self.height()))

        x1 = max(0.0, min(1.0 - w, self.x1))
        y1 = max(0.0, min(1.0 - h, self.y1))
        return CropWindow(x1, y1, x1 + w, y1 + h)


class DigitalZoomManager:
    """Gestisce il ritaglio digitale e l'interpolazione fluida dello zoom."""

    def __init__(
        self,
        lerp_alpha: float = 0.25,
        min_factor: float = 1.0,
        max_factor: float = 6.0,
    ) -> None:
        """
        Args:
            lerp_alpha: Fattore di smoothing per la transizione fluida (0.01 lento, 1.0 istantaneo).
            min_factor: Livello minimo di zoom (1.0 = panoramica completa).
            max_factor: Livello massimo di ingrandimento consentito.
        """
        self.lerp_alpha = lerp_alpha
        self.min_factor = min_factor
        self.max_factor = max_factor

        # Finestre di crop: current (quella renderizzata) e target (quella richiesta)
        self.current_window = CropWindow(0.0, 0.0, 1.0, 1.0)
        self.target_window = CropWindow(0.0, 0.0, 1.0, 1.0)

        self.current_factor: float = 1.0
        self.target_factor: float = 1.0
        self.target_id: Optional[int] = None

    def reset(self) -> None:
        """Riporta lo zoom alla vista panoramica completa (1.0x)."""
        self.target_factor = 1.0
        self.target_window = CropWindow(0.0, 0.0, 1.0, 1.0)
        self.target_id = None

    def zoom_to_point(self, cx: float, cy: float, factor: float) -> None:
        """
        Imposta il target di zoom centrato su (cx, cy) normalizzato [0, 1].
        """
        factor = float(np.clip(factor, self.min_factor, self.max_factor))
        self.target_factor = factor
        self.target_id = None

        if factor <= 1.001:
            self.reset()
            return

        w = 1.0 / factor
        h = 1.0 / factor

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        self.target_window = CropWindow(x1, y1, x2, y2).clamp()

    def zoom_to_bbox(self, bbox: Sequence[float], padding: float = 0.40) -> None:
        """
        Imposta il target di zoom per centrare e inquadrare una bounding box [x, y, w, h] normalizzata.
        """
        if len(bbox) != 4:
            return

        bx, by, bw, bh = [float(v) for v in bbox]
        bw = max(0.02, bw)
        bh = max(0.02, bh)
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        # Aggiungi margine attorno all'oggetto (almeno 1.5x, massimo max_factor)
        target_span = max(bw, bh) * (1.0 + padding)
        factor = float(np.clip(1.0 / max(0.08, target_span), max(1.5, self.min_factor), self.max_factor))

        saved_id = self.target_id
        self.zoom_to_point(cx, cy, factor)
        self.target_id = saved_id

    def set_target_object(self, obj_id: int) -> None:
        """Registra un ID oggetto su cui mantenere il tracking dello zoom."""
        self.target_id = obj_id

    def update_with_tracked_objects(self, objects: Sequence[dict | object]) -> None:
        """
        Se c'è un target_id agganciato, aggiorna la posizione del crop sulla sua bbox corrente.
        """
        if self.target_id is None:
            return

        for obj in objects:
            oid = obj.get("id") if isinstance(obj, dict) else getattr(obj, "id", None)
            if oid == self.target_id:
                bbox = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
                if bbox and len(bbox) == 4:
                    self.zoom_to_bbox(bbox)
                return

    def step(self) -> None:
        """Avanza di un frame nell'interpolazione fluida verso il target."""
        a = self.lerp_alpha

        # Interpolazione lineare su (x1, y1, x2, y2)
        cur = self.current_window
        tgt = self.target_window

        nx1 = cur.x1 + a * (tgt.x1 - cur.x1)
        ny1 = cur.y1 + a * (tgt.y1 - cur.y1)
        nx2 = cur.x2 + a * (tgt.x2 - cur.x2)
        ny2 = cur.y2 + a * (tgt.y2 - cur.y2)

        self.current_window = CropWindow(nx1, ny1, nx2, ny2).clamp()
        self.current_factor = 1.0 / max(1e-4, self.current_window.width())
        if self.target_factor <= 1.001 and self.current_factor <= 1.03:
            self.current_window = CropWindow(0.0, 0.0, 1.0, 1.0)
            self.current_factor = 1.0

    def apply(self, frame: np.ndarray, output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Applica il crop digitale corrente al frame e ridimensiona alle dimensioni desiderate
        (default: stesse dimensioni del frame in ingresso).
        """
        self.step()

        h, w = frame.shape[:2]
        cw = self.current_window

        # Se siamo praticamente a 1.0x, restituisci il frame originale
        if cw.width() >= 0.999 and cw.height() >= 0.999:
            if output_size is not None and (w, h) != output_size:
                return cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)
            return frame

        px1 = int(round(cw.x1 * w))
        py1 = int(round(cw.y1 * h))
        px2 = int(round(cw.x2 * w))
        py2 = int(round(cw.y2 * h))

        # Clamp su indici pixel
        px1 = max(0, min(w - 1, px1))
        py1 = max(0, min(h - 1, py1))
        px2 = max(px1 + 1, min(w, px2))
        py2 = max(py1 + 1, min(h, py2))

        crop = frame[py1:py2, px1:px2]

        out_w, out_h = output_size if output_size is not None else (w, h)
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    def full_to_zoomed_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Converte coordinate normalizzate [0, 1] dal frame completo al frame zoomato."""
        cw = self.current_window
        zx = (x - cw.x1) / cw.width()
        zy = (y - cw.y1) / cw.height()
        return float(zx), float(zy)

    def zoomed_to_full_coords(self, zx: float, zy: float) -> Tuple[float, float]:
        """Converte coordinate normalizzate [0, 1] dal frame zoomato al frame completo."""
        cw = self.current_window
        fx = cw.x1 + zx * cw.width()
        fy = cw.y1 + zy * cw.height()
        return float(fx), float(fy)

    def is_active(self) -> bool:
        """Ritorna True se lo zoom è significativamente attivo (> 1.05x)."""
        return self.current_factor > 1.05 or self.target_factor > 1.05
