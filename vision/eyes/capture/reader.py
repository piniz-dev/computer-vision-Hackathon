"""Acquisizione sorgenti video per Visione A ("Occhi").

Supporta quattro modalità intercambiabili:
- `webcam`: indice della webcam USB/CSI (es. 0, 1)
- `stream`: URL HTTP/MJPEG o RTSP (es. telecamera del cane o smartphone)
- `replay`: file video (.mp4/.avi) o cartella di immagini (.jpg/.png) in loop continuo
- `synthetic`: generatore sintetico in-memory (tavolo con oggetti colorati) per test senza file o hardware
"""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

SourceMode = Literal["webcam", "stream", "replay", "synthetic"]


def make_disconnected_frame(
    width: int = 640,
    height: int = 480,
    source_name: str = "camera",
    message: str = "Camera Disconnected - Retrying...",
) -> np.ndarray:
    """Genera un frame informativo di fallback quando la sorgente non risponde."""
    frame = np.full((height, width, 3), (30, 30, 35), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (0, 0, 200), 2)
    cv2.putText(
        frame,
        f"[{source_name.upper()}]",
        (40, height // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (40, height // 2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def make_synthetic_frame(
    step: int,
    width: int = 640,
    height: int = 480,
    source: str = "arm",
    include_person_near: bool = False,
) -> np.ndarray:
    """Genera un frame sintetico realistico per test e sviluppo senza telecamera."""
    # Sfondo grigio chiaro uniforme (stile piano di lavoro)
    frame = np.full((height, width, 3), (235, 235, 235), dtype=np.uint8)

    # Oggetto 1: Tazza rossa (rettangolo + manico) che oscilla leggermente o resta stabile
    dx = int(8 * math.sin(step * 0.05)) if source == "dog" else 0
    # BGR rosso: (30, 30, 220)
    x1, y1 = 110 + dx, 140
    cv2.rectangle(frame, (x1, y1), (x1 + 75, y1 + 85), (30, 30, 220), -1)

    # Oggetto 2: Bottiglia blu (rettangolo verticale allungato)
    # BGR blu: (220, 60, 20)
    bx, by = 300 - dx, 110
    cv2.rectangle(frame, (bx, by), (bx + 55, by + 130), (220, 60, 20), -1)

    # Oggetto 3: Scatola/blocco verde inclinato o quadrato
    # BGR verde: (40, 190, 40)
    gx, gy = 460, 250
    cv2.rectangle(frame, (gx, gy), (gx + 80, gy + 70), (40, 190, 40), -1)

    if include_person_near:
        # Sagoma scura grande al centro per simulare una persona vicina
        cv2.rectangle(frame, (220, 40), (420, 440), (80, 70, 90), -1)

    return frame


def generate_sample_video(
    output_path: str | Path,
    num_frames: int = 60,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    source: str = "arm",
) -> Path:
    """Crea un file video MP4 sintetico per testare la modalità replay."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        # Fallback su XVID / AVI se mp4v non fosse disponibile
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    for i in range(num_frames):
        frame = make_synthetic_frame(i, width=width, height=height, source=source)
        writer.write(frame)
    writer.release()
    return out_path


class VideoReader:
    """Lettore sincrono e asincrono (thread-safe) per webcam, stream, replay o sintetico."""

    def __init__(
        self,
        source_name: str = "arm",
        mode: SourceMode = "synthetic",
        target: int | str | Path = 0,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        loop_replay: bool = True,
    ):
        self.source_name = source_name
        self.mode: SourceMode = mode
        self.target = target
        self.fps = max(1.0, float(fps))
        self.width = width
        self.height = height
        self.loop_replay = loop_replay

        self._cap: cv2.VideoCapture | None = None
        self._image_files: list[Path] = []
        self._img_idx: int = 0
        self._step: int = 0
        self._frame_id: int = 0
        self._connected: bool = False

        self._lock = threading.Lock()
        self._latest_frame: np.ndarray = make_disconnected_frame(width, height, source_name)
        self._latest_ts: float = time.time()
        self._running: bool = False
        self._thread: threading.Thread | None = None

        self._open_source()

    def _open_source(self) -> None:
        """Apre o riapre la sorgente video configurata."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        if self.mode == "synthetic":
            self._connected = True
            return

        if self.mode == "replay":
            path = Path(str(self.target))
            if path.is_dir():
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                self._image_files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
                self._img_idx = 0
                self._connected = len(self._image_files) > 0
                return
            elif path.is_file():
                self._cap = cv2.VideoCapture(str(path))
                self._connected = bool(self._cap and self._cap.isOpened())
                return
            else:
                self._connected = False
                return

        if self.mode == "webcam":
            dev = int(self.target) if str(self.target).isdigit() else 0
            self._cap = cv2.VideoCapture(dev)
            if self._cap and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._connected = True
            else:
                self._connected = False
            return

        if self.mode == "stream":
            self._cap = cv2.VideoCapture(str(self.target))
            if self._cap and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._connected = True
            else:
                self._connected = False

    def read_once(self) -> tuple[bool, np.ndarray, int, float]:
        """Legge un singolo frame dalla sorgente (sincrono).

        Restituisce `(ok, frame_bgr, frame_id, timestamp)`.
        Se la sorgente è disconnessa restituisce `ok=False` e il frame di fallback.
        """
        ts = time.time()
        frame: np.ndarray | None = None
        ok = False

        if self.mode == "synthetic":
            frame = make_synthetic_frame(self._step, self.width, self.height, self.source_name)
            self._step += 1
            ok = True

        elif self.mode == "replay" and self._image_files:
            if self._img_idx >= len(self._image_files):
                if self.loop_replay:
                    self._img_idx = 0
                else:
                    ok = False
            if self._img_idx < len(self._image_files):
                loaded = cv2.imread(str(self._image_files[self._img_idx]))
                self._img_idx += 1
                if loaded is not None:
                    frame = loaded
                    ok = True

        elif self._cap is not None and self._cap.isOpened():
            ret, grabbed = self._cap.read()
            if not ret and self.mode == "replay" and self.loop_replay:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, grabbed = self._cap.read()
            if ret and grabbed is not None:
                frame = grabbed
                ok = True
            else:
                self._connected = False

        if not ok or frame is None:
            fallback = make_disconnected_frame(self.width, self.height, self.source_name)
            with self._lock:
                self._latest_frame = fallback
                self._latest_ts = ts
            return False, fallback, self._frame_id, ts

        self._connected = True
        self._frame_id += 1
        with self._lock:
            self._latest_frame = frame
            self._latest_ts = ts
        return True, frame.copy(), self._frame_id, ts

    def start_background(self) -> None:
        """Avvia il thread di acquisizione continua."""
        if self._running:
            return
        self._running = True
        # Cattura subito un primo frame sincrono per avere stato immediato
        self.read_once()
        self._thread = threading.Thread(target=self._loop, name=f"VideoReader-{self.source_name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.perf_counter()
            ok, _, _, _ = self.read_once()
            if not ok and self.mode in ("webcam", "stream"):
                time.sleep(0.5)
                self._open_source()
                continue
            elapsed = time.perf_counter() - t0
            if self.mode in ("replay", "synthetic") and elapsed < interval:
                time.sleep(interval - elapsed)

    def get_latest(self) -> tuple[bool, np.ndarray, int, float]:
        """Restituisce l'ultimo frame acquisito senza bloccare."""
        with self._lock:
            return self._connected, self._latest_frame.copy(), self._frame_id, self._latest_ts

    def stop(self) -> None:
        """Ferma il thread di acquisizione e rilascia le risorse OpenCV."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
