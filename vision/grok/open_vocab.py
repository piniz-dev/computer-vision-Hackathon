"""Oggetti fuori dalle 80 classi COCO (penne, matite, ...) seguiti dal vivo.

Grok (detect_objects) dice COSA c'è; YOLO-World cerca quelle classi in ogni frame
e le segue anche mentre si muovono in mano. Gira in un thread suo, sull'ultimo
frame disponibile, così il video non rallenta.

    ov = OpenVocab()                 # carica il modello in background
    ov.set_classes(["pen", "pencil"])
    ov.feed(frame)                   # a ogni frame
    ov.objects                       # ultimi oggetti, formato del gruppo, id da 2000
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

ID_OFFSET = 2000  # YOLO di Visione A usa id < 1000, detect_objects da 1000


def load_vocab(path: Path) -> dict:
    """Classi imparate nelle sessioni precedenti: {label: {name, color, tags, size_cm}}. {} se manca."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_vocab(path: Path, vocab: dict) -> None:
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass  # non salvare il vocabolario non deve fermare la demo


def to_objects(result, max_area: float = 0.5) -> list[dict]:
    """Risultato di ultralytics -> oggetti del gruppo. Scarta i box senza id di tracking
    (non ancora confermati) e quelli più grandi di `max_area` (mani, braccia, sfondo)."""
    b = result.boxes
    if b is None or b.id is None:
        return []
    out = []
    for (x1, y1, x2, y2), conf, cls, tid in zip(b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(),
                                                b.cls.cpu().numpy(), b.id.cpu().numpy()):
        w, h = float(x2 - x1), float(y2 - y1)
        if w * h > max_area:
            continue
        out.append({"id": ID_OFFSET + int(tid), "label": result.names[int(cls)], "name": None,
                    "conf": round(float(conf), 3), "bbox": [float(x1), float(y1), w, h], "color": None,
                    "tags": []})
    return out


class OpenVocab:
    def __init__(self, weights: str = "yolov8s-worldv2.pt", imgsz: int = 480, conf: float = 0.15):
        self.weights, self.imgsz, self.conf = weights, imgsz, conf
        self.classes: list[str] = []
        self.objects: list[dict] = []
        self.status = "carico YOLO-World..."
        self._pending: list[str] | None = None
        self._frame: np.ndarray | None = None
        threading.Thread(target=self._run, daemon=True).start()

    def set_classes(self, labels) -> None:
        """Classi da cercare (in inglese, come le dà Grok). Applicate dal thread del modello."""
        labels = sorted({str(x).strip().lower() for x in labels if str(x).strip()} | set(self.classes))
        if labels != self.classes:
            self._pending = labels

    def forget(self) -> None:
        """Dimentica tutte le classi: YOLO-World smette di cercare finché non arrivano classi nuove."""
        self._pending = []

    def feed(self, frame: np.ndarray) -> None:
        self._frame = frame

    def _run(self) -> None:
        try:
            from ultralytics import YOLOWorld
            model = YOLOWorld(self.weights)
        except Exception as e:
            self.status = f"YOLO-World non disponibile: {e}"
            return
        self.status = "pronto (premi G)"
        while True:
            if self._pending is not None:
                classes, self._pending = self._pending, None
                if classes:
                    self.status = "imparo: " + ", ".join(classes)
                    model.set_classes(classes)
                model.predictor = None  # nuove classi: riparte il tracking
                self.classes, self.objects = classes, []
                self.status = "segue: " + ", ".join(classes) if classes else "pronto (premi G)"
            frame, self._frame = self._frame, None
            if frame is None or not self.classes:
                time.sleep(0.02)
                continue
            try:
                r = model.track(frame, imgsz=self.imgsz, conf=self.conf, persist=True, verbose=False)[0]
                self.objects = to_objects(r)
            except Exception as e:  # un frame storto non deve fermare il thread
                self.status = f"errore YOLO-World: {e}"
