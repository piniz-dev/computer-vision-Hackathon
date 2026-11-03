"""Suite di test completa per Visione A ("Occhi") e integrazione con Visione B ("Cervello")."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from vision.eyes.calibration import (
    ArmCalibrator,
    detect_aruco_centers,
    estimate_grasp_angle,
    render_aruco_workspace,
)
from vision.eyes.capture import VideoReader, generate_sample_video, make_synthetic_frame
from vision.eyes.dog import DogNavigator, check_person_safety_stop, compute_dog_telemetry
from vision.eyes.marks import draw_marks
from vision.eyes.pipeline import EyePipeline
from vision.eyes.schemas import EyeObject
from vision.eyes.server import create_app
from vision.eyes.state import LiveVisionState
from vision.eyes.tracking import ObjectTracker, SimpleByteTracker, estimate_dominant_color_it
from vision.eyes.verification import check_appearance, check_disappearance, verify_action_local
from vision.grok.schemas import parse_objects
from vision.grok.voice_tools import VisionState, dispatch


def test_capture_synthetic_and_replay(tmp_path: Path):
    # 1. Modalità sintetica
    reader = VideoReader(source_name="arm", mode="synthetic", width=640, height=480)
    ok, frame, fid, _ts = reader.read_once()
    assert ok is True
    assert fid == 1
    assert frame.shape == (480, 640, 3)

    # 2. Modalità replay da video MP4
    video_path = generate_sample_video(tmp_path / "replay_test.mp4", num_frames=10, fps=30)
    assert video_path.exists()
    rep_reader = VideoReader(source_name="arm", mode="replay", target=video_path, loop_replay=True)
    for i in range(15):  # più frame della durata per verificare il loop continuo
        ok_r, frame_r, fid_r, _ = rep_reader.read_once()
        assert ok_r is True
        assert fid_r == i + 1
        assert frame_r.shape == (480, 640, 3)
    rep_reader.stop()

    # 3. Sorgente inesistente -> fallback senza crash
    bad_reader = VideoReader(source_name="arm", mode="replay", target=tmp_path / "missing.mp4")
    ok_b, fallback, _, _ = bad_reader.read_once()
    assert ok_b is False
    assert fallback.shape == (480, 640, 3)


def test_tracking_stable_ids_and_color():
    tracker = ObjectTracker(use_mock=True)
    f1 = make_synthetic_frame(0)
    f2 = make_synthetic_frame(1)

    objs1 = tracker.process_frame(f1)
    objs2 = tracker.process_frame(f2)

    assert len(objs1) >= 3
    assert len(objs2) == len(objs1)

    # Gli ID devono rimanere stabili tra i due frame consecutivi
    ids1 = [o.id for o in objs1]
    ids2 = [o.id for o in objs2]
    assert ids1 == ids2

    # Bbox normalizzati [0, 1] e compatibili con parse_objects di Visione B
    parsed_b = parse_objects([o.model_dump() for o in objs1])
    assert len(parsed_b) == len(objs1)
    for o in objs1:
        x, y, w, h = o.bbox
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0

    colors = {o.color for o in objs1}
    assert "rosso" in colors
    assert "blu" in colors
    assert "verde" in colors


def test_set_of_mark_does_not_mutate_raw():
    raw = make_synthetic_frame(0)
    raw_copy = raw.copy()
    objs = [
        EyeObject(id=3, label="cup", conf=0.88, bbox=(0.2, 0.2, 0.15, 0.2), color="rosso"),
        EyeObject(id=4, label="pen", conf=0.45, bbox=(0.5, 0.5, 0.05, 0.1), color="blu"),
    ]
    marked = draw_marks(raw, objs, min_conf=0.60, show_conf=True)

    assert np.array_equal(raw, raw_copy), "Il frame raw non deve essere modificato in-place"
    assert not np.array_equal(raw, marked), "Il frame marcato deve contenere i riquadri disegnati"

    # Con min_conf=0.95 nessun oggetto deve essere disegnato -> frame intatto
    marked_none = draw_marks(raw, objs, min_conf=0.95)
    assert np.array_equal(raw, marked_none)


def test_tracker_smoothing_and_min_hits():
    # Tracker con smoothing ed eliminazione flickering (min_hits=2)
    tracker = SimpleByteTracker(ema_alpha=0.5, min_hits=2)
    dets1 = [("cup", 0.9, (0.20, 0.20, 0.10, 0.10))]
    dets2 = [("cup", 0.9, (0.24, 0.22, 0.10, 0.10))]

    # Al frame 1 ha hits=1 < min_hits, quindi non viene ancora emesso (anti-flicker)
    out1 = tracker.update(dets1)
    assert len(out1) == 0

    # Al frame 2 ha hits=2 >= min_hits, viene emesso con coordinate smoothed
    out2 = tracker.update(dets2)
    assert len(out2) == 1
    tid, label, conf, box = out2[0]
    assert tid == 1
    # Coordinate mediate tra (0.20, 0.20) e (0.24, 0.22) -> (0.22, 0.21)
    assert abs(box[0] - 0.22) < 0.01
    assert abs(box[1] - 0.21) < 0.01


def test_aruco_calibration_homography_and_grasp_angle(tmp_path: Path):
    board_img, true_centers = render_aruco_workspace(width=640, height=480)
    detected = detect_aruco_centers(board_img)
    assert set(detected.keys()) == {0, 1, 2, 3}

    calibrator = ArmCalibrator()
    ok, _ = calibrator.calibrate_from_aruco(board_img)
    assert ok is True
    assert calibrator.is_calibrated

    # Verifica precisione sui 4 angoli noti: (0,0), (300,0), (300,200), (0,200)
    c0_px = detected[0]
    x_mm, y_mm = calibrator.pixel_to_mm(*c0_px)
    assert abs(x_mm - 0.0) < 0.5 and abs(y_mm - 0.0) < 0.5

    c2_px = detected[2]
    x2_mm, y2_mm = calibrator.pixel_to_mm(*c2_px)
    assert abs(x2_mm - 300.0) < 0.5 and abs(y2_mm - 200.0) < 0.5

    # Verifica centro tavolo (150 mm, 100 mm)
    mid_px = calibrator.mm_to_pixel(150.0, 100.0)
    mid_mm = calibrator.pixel_to_mm(*mid_px)
    assert abs(mid_mm[0] - 150.0) < 0.5 and abs(mid_mm[1] - 100.0) < 0.5

    # Salvataggio e ricaricamento JSON
    calib_file = calibrator.save(tmp_path / "calibration_arm.json")
    loaded = ArmCalibrator.load(calib_file)
    assert loaded.pixel_to_mm(*mid_px) == mid_mm

    # Stima angolo di presa su barra inclinata di +30 gradi
    canvas = np.full((300, 300, 3), 240, dtype=np.uint8)
    rect = ((150.0, 150.0), (120.0, 24.0), 30.0)
    box_pts = np.int32(cv2.boxPoints(rect))
    cv2.fillPoly(canvas, [box_pts], (20, 20, 200))
    angle = estimate_grasp_angle(canvas, (0.2, 0.2, 0.6, 0.6))
    assert abs(angle - 30.0) <= 5.0


def test_dog_telemetry_and_safety_stop():
    objs = [
        EyeObject(id=1, label="cup", conf=0.9, bbox=(0.05, 0.4, 0.1, 0.2), color="rosso"),   # a sinistra
        EyeObject(id=2, label="bottle", conf=0.85, bbox=(0.45, 0.3, 0.1, 0.4), color="blu"),  # al centro
        EyeObject(id=3, label="box", conf=0.8, bbox=(0.80, 0.4, 0.1, 0.15), color="verde"),   # a destra
    ]

    t_left = compute_dog_telemetry(objs, target_id=1)
    assert t_left.target_visible is True
    assert t_left.error_x < -0.5

    t_center = compute_dog_telemetry(objs, target_id=2)
    assert t_center.target_visible is True
    assert abs(t_center.error_x) < 0.05
    assert abs(t_center.bbox_height - 0.4) < 1e-3

    t_right = compute_dog_telemetry(objs, target_id=3)
    assert t_right.error_x > 0.5

    # Nessuna persona -> safety_stop False
    stop, _ = check_person_safety_stop(objs)
    assert stop is False

    # Persona vicina (bbox alto 0.65) -> safety_stop True
    objs_with_person = objs + [
        EyeObject(id=9, label="person", conf=0.88, bbox=(0.3, 0.1, 0.4, 0.65))
    ]
    nav = DogNavigator()
    nav.set_target(target_id=2)
    tel, stop_now, reason = nav.evaluate(objs_with_person)
    assert tel.target_visible is True
    assert stop_now is True
    assert reason is not None and "Persona vicina" in reason


def test_local_verification_pick_and_place():
    before = [
        EyeObject(id=1, label="cup", conf=0.9, bbox=(0.10, 0.20, 0.15, 0.20), color="rosso"),
        EyeObject(id=5, label="box", conf=0.9, bbox=(0.70, 0.60, 0.20, 0.20), color="verde"),
    ]
    # Caso 1: Pick riuscito (tazza #1 sparita dalla partenza)
    after_pick = [
        EyeObject(id=5, label="box", conf=0.9, bbox=(0.70, 0.60, 0.20, 0.20), color="verde"),
    ]
    res_pick = verify_action_local(before, after_pick, action="pick", target_id=1)
    assert res_pick.success is True
    assert res_pick.disappeared_from_source is True

    # Caso 2: Place riuscito (tazza #1 spostata dentro la scatola #5)
    after_place = [
        EyeObject(id=1, label="cup", conf=0.88, bbox=(0.72, 0.62, 0.12, 0.15), color="rosso"),
        EyeObject(id=5, label="box", conf=0.9, bbox=(0.70, 0.60, 0.20, 0.20), color="verde"),
    ]
    res_place = verify_action_local(before, after_place, action="place", target_id=1, destination_id=5)
    assert res_place.success is True
    assert res_place.disappeared_from_source is True
    assert res_place.arrived_at_destination is True


def test_integration_with_grok_voice_tools(monkeypatch):
    monkeypatch.setenv("GROK_MOCK", "1")
    pipeline = EyePipeline()
    pipeline.add_source("arm", mode="synthetic", use_mock_tracker=True)
    pipeline.step("arm")

    state = pipeline.state
    assert isinstance(state, VisionState)

    # 1. list_objects tramite voice_tools.dispatch di Visione B
    res_list = dispatch("list_objects", {"source": "arm"}, state)
    assert res_list["ok"] is True
    assert len(res_list["objects"]) >= 3
    assert "Vedo" in res_list["say"]

    # 2. select_object risolto in locale grazie a label + color popolati da Visione A
    res_sel = dispatch("select_object", {"source": "arm", "description": "prendi la tazza rossa"}, state)
    assert res_sel["ok"] is True
    assert res_sel["source"] == "local"
    assert res_sel["id"] is not None


def test_fastapi_server_endpoints_and_websocket(monkeypatch):
    monkeypatch.setenv("GROK_MOCK", "1")
    pipeline = EyePipeline()
    pipeline.add_source("arm", mode="synthetic", use_mock_tracker=True)
    pipeline.add_source("dog", mode="synthetic", use_mock_tracker=True)
    app = create_app(pipeline)

    with TestClient(app) as client:
        # Health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert "arm" in r_health.json()["sources"]

        # State JSON
        r_state = client.get("/state/arm")
        assert r_state.status_code == 200
        data = r_state.json()
        assert data["source"] == "arm"
        assert len(data["objects"]) >= 3

        # Frame JPEG Set-of-Mark
        r_jpg = client.get("/frame/arm.jpg")
        assert r_jpg.status_code == 200
        assert r_jpg.headers["content-type"] == "image/jpeg"
        assert len(r_jpg.content) > 1000

        # Dog target & state
        target_id = data["objects"][0]["id"]
        client.post("/dog/target", json={"target_id": target_id})
        r_dog = client.get("/state/dog").json()
        assert r_dog["dog_telemetry"]["target_visible"] is True

        # Voice dispatch bridge
        r_voice = client.post(
            "/voice/dispatch",
            json={"name": "select_object", "arguments": {"source": "arm", "description": "bottiglia blu"}},
        )
        assert r_voice.status_code == 200
        assert r_voice.json()["ok"] is True

        # WebSocket JSON stream
        with client.websocket_connect("/ws/objects/arm") as ws:
            msg = ws.receive_json()
            assert msg["source"] == "arm"
            assert isinstance(msg["objects"], list)
            assert len(msg["objects"]) >= 3
