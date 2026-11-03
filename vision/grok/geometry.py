"""Sagoma dell'oggetto dentro il suo riquadro (GrabCut di OpenCV, nessun modello)."""
from __future__ import annotations

import cv2
import numpy as np


def refine_box(frame: np.ndarray, bbox, max_side: int = 160):
    """(bbox stretto, contorno in pixel del frame) oppure None se nel riquadro non c'è un oggetto
    distinguibile dallo sfondo. bbox = [x, y, w, h] normalizzati 0-1."""
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    # ritaglio con un po' di margine attorno: GrabCut impara lo sfondo da lì
    mx, my = w * 0.15, h * 0.15
    x1, y1 = max(0, int((x - mx) * W)), max(0, int((y - my) * H))
    x2, y2 = min(W, int((x + w + mx) * W)), min(H, int((y + h + my) * H))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    s = min(1.0, max_side / max(crop.shape[:2]))  # piccolo = veloce, basta per un contorno
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
    sh, sw = small.shape[:2]
    rect = (max(1, int((x * W - x1) * s)), max(1, int((y * H - y1) * s)),
            max(2, min(sw - 2, int(w * W * s))), max(2, min(sh - 2, int(h * H * s))))
    mask = np.zeros((sh, sw), np.uint8)
    try:
        cv2.grabCut(small, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:  # riquadro tutto uniforme: GrabCut non ha niente da separare
        return None
    fg = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    frac = fg.sum() / max(1, rect[2] * rect[3])
    if not 0.05 < frac < 0.97:  # vuoto o tutto pieno: nessun oggetto separabile
        return None
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    c = (c.astype(np.float32) / s + (x1, y1)).astype(np.int32)  # di nuovo in pixel del frame
    bx, by, bw, bh = cv2.boundingRect(c)
    return [bx / W, by / H, bw / W, bh / H], c
