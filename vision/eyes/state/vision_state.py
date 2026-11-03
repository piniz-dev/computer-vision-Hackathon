"""Implementazione thread-safe di `VisionState` per collegare Visione A ("Occhi"),
Visione B ("Cervello" - `vision/grok/voice_tools.py`), App e Robotica.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from vision.grok.voice_tools import VisionState

from ..capture.reader import make_disconnected_frame
from ..marks.som import draw_marks
from ..schemas import DogTelemetry, EyeObject, FramePayload


@dataclass
class _SourceBuffer:
    source: str
    raw_frame: np.ndarray
    marked_frame: np.ndarray
    payload: FramePayload
    last_action: dict | None = None
    history: deque[FramePayload] = field(default_factory=lambda: deque(maxlen=30))


class LiveVisionState(VisionState):
    """Stato condiviso in tempo reale implementato da Visione A."""

    def __init__(self, history_maxlen: int = 30):
        self._lock = threading.RLock()
        self._history_maxlen = history_maxlen
        self._sources: dict[str, _SourceBuffer] = {}

    def _ensure_source(self, source: str) -> _SourceBuffer:
        if source not in self._sources:
            fallback = make_disconnected_frame(640, 480, source)
            payload = FramePayload(source=source, frame_id=0, ts=time.time(), objects=[])
            self._sources[source] = _SourceBuffer(
                source=source,
                raw_frame=fallback,
                marked_frame=fallback.copy(),
                payload=payload,
                history=deque(maxlen=self._history_maxlen),
            )
        return self._sources[source]

    def update(
        self,
        source: str,
        raw_frame: np.ndarray,
        objects: Sequence[EyeObject | dict],
        frame_id: int | None = None,
        ts: float | None = None,
        marked_frame: np.ndarray | None = None,
        dog_telemetry: DogTelemetry | None = None,
        safety_stop: bool = False,
        safety_reason: str | None = None,
    ) -> FramePayload:
        """Aggiorna atomicamente frame originale, frame Set-of-Mark, oggetti e telemetria."""
        parsed_objs = [
            o if isinstance(o, EyeObject) else EyeObject.model_validate(o)
            for o in objects
        ]
        now = ts if ts is not None else time.time()

        with self._lock:
            buf = self._ensure_source(source)
            fid = frame_id if frame_id is not None else (buf.payload.frame_id + 1)
            raw_copy = raw_frame.copy()
            marked_copy = (
                marked_frame.copy()
                if marked_frame is not None
                else draw_marks(raw_copy, parsed_objs, safety_stop=safety_stop)
            )
            payload = FramePayload(
                source=source,
                frame_id=fid,
                ts=now,
                objects=parsed_objs,
                dog_telemetry=dog_telemetry,
                safety_stop=safety_stop,
                safety_reason=safety_reason,
            )
            buf.raw_frame = raw_copy
            buf.marked_frame = marked_copy
            buf.payload = payload
            buf.history.append(payload)
            return payload

    def get_frame(self, source: str) -> np.ndarray:
        """Ultimo frame originale (BGR) per la sorgente indicata."""
        with self._lock:
            return self._ensure_source(source).raw_frame.copy()

    def get_marked_frame(self, source: str) -> np.ndarray:
        """Ultimo frame con riquadri e numeri (Set-of-Mark)."""
        with self._lock:
            return self._ensure_source(source).marked_frame.copy()

    def get_objects(self, source: str) -> list[dict]:
        """Oggetti dell'ultimo frame, come lista di dict compatibile con Visione B e App."""
        with self._lock:
            buf = self._ensure_source(source)
            return [o.model_dump() for o in buf.payload.objects]

    def get_payload(self, source: str) -> FramePayload:
        """Restituisce l'intero `FramePayload` (inclusi `dog_telemetry` e `safety_stop`)."""
        with self._lock:
            return self._ensure_source(source).payload.model_copy(deep=True)

    def get_history(self, source: str) -> list[FramePayload]:
        """Restituisce gli ultimi N payload registrati per la sorgente."""
        with self._lock:
            return list(self._ensure_source(source).history)

    def record_action_start(
        self,
        source: str,
        action: str,
        target_id: int,
        destination_id: int | None = None,
    ) -> dict:
        """Fotografa lo stato 'prima' di un'azione robotica per la verifica successiva."""
        with self._lock:
            buf = self._ensure_source(source)
            record = {
                "action": action,
                "target_id": target_id,
                "destination_id": destination_id,
                "frame_before": buf.raw_frame.copy(),
                "objects_before": [o.model_dump() for o in buf.payload.objects],
                "ts_before": buf.payload.ts,
            }
            buf.last_action = record
            return record

    def get_last_action(self, source: str) -> dict | None:
        """Restituisce l'ultima azione registrata nel formato atteso da `voice_tools.py`."""
        with self._lock:
            act = self._ensure_source(source).last_action
            if act is None:
                return None
            return {
                "action": act["action"],
                "target_id": act["target_id"],
                "destination_id": act["destination_id"],
                "frame_before": act["frame_before"].copy(),
                "objects_before": list(act["objects_before"]),
            }

    def list_sources(self) -> list[str]:
        with self._lock:
            return list(self._sources.keys())
