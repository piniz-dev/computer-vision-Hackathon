"""Rilevamento e tracciamento oggetti (YOLO + ByteTrack) per Visione A ("Occhi").

Caratteristiche:
- Integrazione nativa con `ultralytics.YOLO` + `bytetrack.yaml` per ID stabili.
- Tracker IoU/ByteTrack leggero (`SimpleByteTracker`) integrato sia come fallback
  quando YOLO non assegna ancora un ID al primo frame, sia in modalità `mock=True`
  (segmentazione OpenCV + tracking) per test rapidi senza GPU o pesi scaricati.
- Stima locale del colore dominante in italiano (`rosso`, `blu`, `verde`, `giallo`, ...)
  per abilitare subito la scorciatoia locale di Visione B (`local_resolver.py`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..schemas import CoordsMM, EyeObject


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Calcola l'Intersection over Union (IoU) tra due bbox `[x, y, w, h]`."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw * ah)
    area_b = max(0.0, bw * bh)
    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return inter_area / union


def estimate_dominant_color_it(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> str | None:
    """Stima il colore dominante dell'oggetto in italiano usando lo spazio HSV."""
    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = bbox
    # Ritaglia la regione centrale (50%) dell'oggetto per evitare lo sfondo ai bordi
    cx1 = max(0, int((x + 0.25 * w) * w_img))
    cy1 = max(0, int((y + 0.25 * h) * h_img))
    cx2 = min(w_img, int((x + 0.75 * w) * w_img))
    cy2 = min(h_img, int((y + 0.75 * h) * h_img))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    roi = frame_bgr[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_h = float(np.median(hsv[:, :, 0]))
    mean_s = float(np.median(hsv[:, :, 1]))
    mean_v = float(np.median(hsv[:, :, 2]))

    if mean_v < 45:
        return "nero"
    if mean_s < 35:
        return "bianco" if mean_v > 195 else "grigio"

    # Hue in OpenCV è [0, 179]
    if mean_h < 10 or mean_h >= 165:
        return "rosso"
    if 10 <= mean_h < 22:
        return "arancione"
    if 22 <= mean_h < 36:
        return "giallo"
    if 36 <= mean_h < 85:
        return "verde"
    if 85 <= mean_h < 135:
        return "blu"
    if 135 <= mean_h < 165:
        return "viola"
    return None


@dataclass
class _TrackSlot:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    conf: float
    missed: int = 0


class SimpleByteTracker:
    """Tracker stile ByteTrack a due stadi (alta/bassa confidenza + IoU) per ID stabili."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id: int = 1
        self._tracks: dict[int, _TrackSlot] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def update(
        self,
        detections: list[tuple[str, float, tuple[float, float, float, float]]],
    ) -> list[tuple[int, str, float, tuple[float, float, float, float]]]:
        """Associa le detection `(label, conf, bbox)` agli ID attivi e restituisce `(id, label, conf, bbox)`."""
        for trk in self._tracks.values():
            trk.missed += 1

        assigned_det: set[int] = set()
        assigned_trk: set[int] = set()
        results: list[tuple[int, str, float, tuple[float, float, float, float]]] = []

        # Costruisci coppie (iou, trk_id, det_idx)
        candidates: list[tuple[float, int, int]] = []
        for trk_id, trk in self._tracks.items():
            for d_idx, (label, _conf, bbox) in enumerate(detections):
                iou = compute_iou(trk.bbox, bbox)
                # Bonus se la classe coincide
                score = iou + (0.1 if trk.label == label and iou > 0.05 else 0.0)
                if iou >= self.iou_threshold:
                    candidates.append((score, trk_id, d_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _score, trk_id, d_idx in candidates:
            if trk_id in assigned_trk or d_idx in assigned_det:
                continue
            label, conf, bbox = detections[d_idx]
            trk = self._tracks[trk_id]
            trk.bbox = bbox
            trk.conf = conf
            trk.label = label
            trk.missed = 0
            assigned_trk.add(trk_id)
            assigned_det.add(d_idx)
            results.append((trk_id, label, conf, bbox))

        # Nuove tracce per le detection non associate
        for d_idx, (label, conf, bbox) in enumerate(detections):
            if d_idx in assigned_det:
                continue
            new_id = self._next_id
            self._next_id += 1
            self._tracks[new_id] = _TrackSlot(track_id=new_id, label=label, bbox=bbox, conf=conf, missed=0)
            results.append((new_id, label, conf, bbox))

        # Rimuovi tracce perse da troppi frame
        stale = [tid for tid, trk in self._tracks.items() if trk.missed > self.max_missed]
        for tid in stale:
            del self._tracks[tid]

        results.sort(key=lambda r: r[0])
        return results


class ObjectTracker:
    """Rilevatore e tracker unificato (YOLO + ByteTrack con modalità mock/fallback)."""

    def __init__(
        self,
        model_name: str | None = None,
        conf_threshold: float = 0.40,
        use_mock: bool | None = None,
        calibrator_fn: Callable[[np.ndarray, tuple[float, float, float, float]], CoordsMM | None] | None = None,
    ):
        self.model_name = model_name or os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.conf_threshold = conf_threshold
        self.calibrator_fn = calibrator_fn
        self._fallback_tracker = SimpleByteTracker()
        # Memoria dei nomi/colori assegnati da Grok per ciascun ID tracciato
        self._custom_names: dict[int, tuple[str | None, str | None]] = {}

        if use_mock is None:
            use_mock = os.getenv("EYES_MOCK", "0") == "1"
        self.use_mock = use_mock
        self._yolo_model = None

        if not self.use_mock:
            try:
                from ultralytics import YOLO  # type: ignore

                self._yolo_model = YOLO(self.model_name)
            except Exception:
                # Se ultralytics non è installato o il modello non è presente offline, usa il tracker locale
                self.use_mock = True

    def set_object_metadata(self, obj_id: int, name: str | None = None, color: str | None = None) -> None:
        """Salva nome descrittivo e colore (es. ricevuti da `name_objects` di Grok) per l'ID specificato."""
        prev_name, prev_color = self._custom_names.get(obj_id, (None, None))
        self._custom_names[obj_id] = (name or prev_name, color or prev_color)

    def _detect_mock(self, frame_bgr: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        """Rileva oggetti colorati sul tavolo o sagome tramite OpenCV (modalità leggera/test)."""
        h_img, w_img = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Maschera per oggetti con saturazione significativa o molto scuri rispetto allo sfondo chiaro
        sat_mask = cv2.inRange(hsv, (0, 50, 40), (180, 255, 255))
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 100, 110))
        mask = cv2.bitwise_or(sat_mask, dark_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_dets: list[tuple[str, float, tuple[float, float, float, float]]] = []

        min_area = (w_img * h_img) * 0.003
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            nx, ny, nw, nh = x / w_img, y / h_img, w / w_img, h / h_img

            # Euristica della classe in base alle proporzioni e dimensioni (per simulazione/test)
            if nh > 0.55 and nw > 0.20:
                label = "person"
                conf = 0.92
            elif h / max(1, w) > 1.6:
                label = "bottle"
                conf = 0.89
            else:
                color = estimate_dominant_color_it(frame_bgr, (nx, ny, nw, nh))
                label = "cup" if color in ("rosso", "giallo", "arancione") else "box"
                conf = 0.87

            raw_dets.append((label, conf, (nx, ny, nw, nh)))

        return raw_dets

    def process_frame(self, frame_bgr: np.ndarray, apply_calibration: bool = False) -> list[EyeObject]:
        """Rileva e traccia gli oggetti nel frame restituendo una lista di `EyeObject`."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h_img, w_img = frame_bgr.shape[:2]
        tracked_items: list[tuple[int, str, float, tuple[float, float, float, float]]] = []

        if not self.use_mock and self._yolo_model is not None:
            results = self._yolo_model.track(
                frame_bgr,
                persist=True,
                conf=self.conf_threshold,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxyn = boxes.xyxyn.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                    raw_for_fallback: list[tuple[str, float, tuple[float, float, float, float]]] = []
                    for i in range(len(xyxyn)):
                        x1, y1, x2, y2 = [float(v) for v in xyxyn[i]]
                        bbox = (
                            max(0.0, min(1.0, x1)),
                            max(0.0, min(1.0, y1)),
                            max(0.0, min(1.0, x2 - x1)),
                            max(0.0, min(1.0, y2 - y1)),
                        )
                        label = str(r.names.get(int(clss[i]), "object"))
                        conf = float(confs[i])
                        if ids is not None:
                            tracked_items.append((int(ids[i]), label, conf, bbox))
                        else:
                            raw_for_fallback.append((label, conf, bbox))

                    if raw_for_fallback:
                        tracked_items.extend(self._fallback_tracker.update(raw_for_fallback))
        else:
            raw_dets = self._detect_mock(frame_bgr)
            tracked_items = self._fallback_tracker.update(raw_dets)

        eye_objects: list[EyeObject] = []
        for obj_id, label, conf, bbox in tracked_items:
            detected_color = estimate_dominant_color_it(frame_bgr, bbox)
            saved_name, saved_color = self._custom_names.get(obj_id, (None, None))
            coords_mm = None
            if apply_calibration and self.calibrator_fn is not None:
                coords_mm = self.calibrator_fn(frame_bgr, bbox)

            eye_objects.append(
                EyeObject(
                    id=obj_id,
                    label=label,
                    name=saved_name,
                    conf=round(conf, 3),
                    bbox=bbox,
                    color=saved_color or detected_color,
                    coords_mm=coords_mm,
                )
            )

        return eye_objects
