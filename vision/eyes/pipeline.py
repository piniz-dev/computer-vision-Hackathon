"""Pipeline di percezione locale in tempo reale per Visione A ("Occhi").

Unisce:
- `VideoReader` (acquisizione webcam / stream / replay / synthetic)
- `ObjectTracker` (YOLO + ByteTrack / Mock fallback)
- `ArmCalibrator` (omografia ArUco pixel -> mm + angolo di presa per `arm`)
- `DogNavigator` (errore orizzontale, altezza bbox e safety stop persone per `dog`)
- `draw_marks` (Set-of-Mark con badge numerici)
- `LiveVisionState` (stato condiviso thread-safe per Visione B, App e Robotica)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .calibration.aruco_calib import ArmCalibrator
from .capture.reader import SourceMode, VideoReader
from .dog.telemetry import DogNavigator, check_person_safety_stop
from .marks.som import draw_marks
from .schemas import FramePayload
from .state.vision_state import LiveVisionState
from .tracking.tracker import ObjectTracker
from .zoom.manager import DigitalZoomManager


@dataclass
class SourceNode:
    source: str
    reader: VideoReader
    tracker: ObjectTracker
    calibrator: ArmCalibrator | None = None
    dog_navigator: DogNavigator | None = None
    zoom: DigitalZoomManager | None = None


class EyePipeline:
    """Motore principale di Visione A ("Occhi")."""

    def __init__(
        self,
        state: LiveVisionState | None = None,
        fps: float = 25.0,
    ):
        self.state = state or LiveVisionState()
        self.fps = max(1.0, float(fps))
        self.nodes: dict[str, SourceNode] = {}
        self._running = False
        self._thread: threading.Thread | None = None

    def add_source(
        self,
        source: str = "arm",
        mode: SourceMode = "synthetic",
        target: int | str | Path = 0,
        use_mock_tracker: bool | None = None,
        calibration_file: str | Path | None = None,
        calibrator: ArmCalibrator | None = None,
        model_name: str | None = None,
        conf_threshold: float = 0.40,
        ema_alpha: float = 0.70,
        min_hits: int = 1,
    ) -> SourceNode:
        """Configura e registra una sorgente (`arm` o `dog`)."""
        if calibrator is None and calibration_file is not None and Path(calibration_file).exists():
            calibrator = ArmCalibrator.load(calibration_file)
        elif calibrator is None and source == "arm":
            calibrator = ArmCalibrator()

        calib_fn = calibrator.compute_object_coords if (calibrator and source == "arm") else None
        tracker = ObjectTracker(
            model_name=model_name,
            conf_threshold=conf_threshold,
            use_mock=use_mock_tracker,
            calibrator_fn=calib_fn,
            ema_alpha=ema_alpha,
            min_hits=min_hits,
        )
        reader = VideoReader(source_name=source, mode=mode, target=target, fps=self.fps)
        dog_nav = DogNavigator() if source == "dog" else None
        zoom_mgr = DigitalZoomManager()

        node = SourceNode(
            source=source,
            reader=reader,
            tracker=tracker,
            calibrator=calibrator,
            dog_navigator=dog_nav,
            zoom=zoom_mgr,
        )
        self.nodes[source] = node
        return node

    def step(
        self,
        source: str,
        min_display_conf: float = 0.0,
        show_conf: bool = False,
        apply_zoom: bool = True,
    ) -> FramePayload:
        """Esegue un singolo ciclo completo di percezione per la sorgente indicata."""
        if source not in self.nodes:
            raise KeyError(f"Sorgente '{source}' non registrata nella pipeline.")
        node = self.nodes[source]

        if node.reader._running:
            _ok, frame, frame_id, ts = node.reader.get_latest()
        else:
            _ok, frame, frame_id, ts = node.reader.read_once()

        apply_calib = bool(node.calibrator and node.calibrator.is_calibrated and source == "arm")
        objects = node.tracker.process_frame(frame, apply_calibration=apply_calib)

        dog_telemetry = None
        if node.dog_navigator is not None:
            dog_telemetry, safety_stop, safety_reason = node.dog_navigator.evaluate(objects)
        else:
            safety_stop, safety_reason = check_person_safety_stop(objects)

        marked = draw_marks(
            frame,
            objects,
            show_grasp=apply_calib,
            safety_stop=safety_stop,
            min_conf=min_display_conf,
            show_conf=show_conf,
        )

        if apply_zoom and node.zoom is not None and node.zoom.is_active():
            node.zoom.update_with_tracked_objects(objects)
            marked = node.zoom.apply(marked)

        return self.state.update(
            source=source,
            raw_frame=frame,
            objects=objects,
            frame_id=frame_id,
            ts=ts,
            marked_frame=marked,
            dog_telemetry=dog_telemetry,
            safety_stop=safety_stop,
            safety_reason=safety_reason,
        )

    def start(self) -> None:
        """Avvia i reader e il ciclo continuo di percezione in background."""
        if self._running:
            return
        for node in self.nodes.values():
            node.reader.start_background()
            self.step(node.source)

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="EyePipelineLoop", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        interval = 1.0 / self.fps
        while self._running:
            t0 = time.perf_counter()
            for source in list(self.nodes.keys()):
                try:
                    self.step(source)
                except Exception:
                    pass
            elapsed = time.perf_counter() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def stop(self) -> None:
        """Ferma il loop e tutti i reader video."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        for node in self.nodes.values():
            node.reader.stop()
