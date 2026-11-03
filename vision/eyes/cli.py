"""Riga di comando (CLI) per Visione A ("Occhi").

Esempi d'uso:
  python -m vision.eyes.cli inspect --mode synthetic
  python -m vision.eyes.cli generate-replay --output data/replays/sample_arm.mp4
  python -m vision.eyes.cli calibrate-arm --generate-board data/calibration/aruco_board.png --output data/calibration/calibration_arm.json
  python -m vision.eyes.cli run --mode synthetic --port 8000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import uvicorn

from .calibration.aruco_calib import ArmCalibrator, render_aruco_workspace
from .capture.reader import generate_sample_video
from .pipeline import EyePipeline
from .server.app import create_app


def _cmd_inspect(args: argparse.Namespace) -> int:
    pipeline = EyePipeline(fps=args.fps)
    target = args.file if args.mode == "replay" else args.device
    pipeline.add_source(
        source=args.source,
        mode=args.mode,
        target=target,
        use_mock_tracker=args.mock_tracker,
        calibration_file=args.calibration,
    )
    payload = None
    for _ in range(max(1, args.steps)):
        payload = pipeline.step(args.source)

    if args.save_marked and payload is not None:
        out_img = Path(args.save_marked)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), pipeline.state.get_marked_frame(args.source))

    if payload is not None:
        print(json.dumps(payload.model_dump(), indent=2, ensure_ascii=False))
    return 0


def _cmd_generate_replay(args: argparse.Namespace) -> int:
    out = generate_sample_video(
        output_path=args.output,
        num_frames=args.frames,
        fps=args.fps,
        source=args.source,
    )
    print(f"Video di replay generato in: {out}")
    return 0


def _cmd_calibrate_arm(args: argparse.Namespace) -> int:
    if args.generate_board:
        board_img, _ = render_aruco_workspace()
        board_path = Path(args.generate_board)
        board_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(board_path), board_img)
        print(f"Tavola ArUco generata in: {board_path}")
        frame = board_img
    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Errore: impossibile leggere l'immagine {args.image}")
            return 1
    else:
        cap = cv2.VideoCapture(int(args.device))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"Errore: impossibile catturare frame dal dispositivo {args.device}")
            return 1

    calibrator = ArmCalibrator()
    ok, centers = calibrator.calibrate_from_aruco(frame)
    if not ok:
        print(f"Calibrazione fallita: rilevati solo {len(centers)} marker ArUco (servono 4: ID 0,1,2,3).")
        return 1

    saved = calibrator.save(args.output)
    print(f"Calibrazione completata ({len(centers)} marker rilevati). Salvata in: {saved}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    pipeline = EyePipeline(fps=args.fps)
    arm_target = args.file if args.mode == "replay" else args.device
    pipeline.add_source(
        source="arm",
        mode=args.mode,
        target=arm_target,
        use_mock_tracker=args.mock_tracker,
        calibration_file=args.calibration,
    )
    pipeline.add_source(
        source="dog",
        mode=args.mode,
        target=args.dog_target or arm_target,
        use_mock_tracker=args.mock_tracker,
    )
    pipeline.start()
    app = create_app(pipeline)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        pipeline.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision.eyes.cli", description="CLI per Visione A ('Occhi')")
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_insp = sub.add_parser("inspect", help="Esegue N cicli di percezione e stampa il JSON")
    p_insp.add_argument("--source", default="arm", choices=["arm", "dog"])
    p_insp.add_argument("--mode", default="synthetic", choices=["webcam", "replay", "stream", "synthetic"])
    p_insp.add_argument("--device", default="0")
    p_insp.add_argument("--file", default="data/replays/sample_arm.mp4")
    p_insp.add_argument("--steps", type=int, default=1)
    p_insp.add_argument("--fps", type=float, default=30.0)
    p_insp.add_argument("--calibration", default="data/calibration/calibration_arm.json")
    p_insp.add_argument("--save-marked", default=None)
    p_insp.add_argument("--mock-tracker", action="store_true", default=None)
    p_insp.set_defaults(func=_cmd_inspect)

    # generate-replay
    p_rep = sub.add_parser("generate-replay", help="Genera un file video MP4 sintetico per la modalità replay")
    p_rep.add_argument("--output", default="data/replays/sample_arm.mp4")
    p_rep.add_argument("--frames", type=int, default=60)
    p_rep.add_argument("--fps", type=int, default=30)
    p_rep.add_argument("--source", default="arm")
    p_rep.set_defaults(func=_cmd_generate_replay)

    # calibrate-arm
    p_cal = sub.add_parser("calibrate-arm", help="Calibra l'omografia ArUco del braccio")
    p_cal.add_argument("--device", default="0")
    p_cal.add_argument("--image", default=None, help="Immagine con i 4 marker ArUco")
    p_cal.add_argument("--generate-board", default=None, help="Genera e calibra su una tavola ArUco sintetica")
    p_cal.add_argument("--output", default="data/calibration/calibration_arm.json")
    p_cal.set_defaults(func=_cmd_calibrate_arm)

    # run
    p_run = sub.add_parser("run", help="Avvia la pipeline e il server FastAPI (MJPEG + WebSocket)")
    p_run.add_argument("--mode", default="synthetic", choices=["webcam", "replay", "stream", "synthetic"])
    p_run.add_argument("--device", default="0")
    p_run.add_argument("--file", default="data/replays/sample_arm.mp4")
    p_run.add_argument("--dog-target", default=None)
    p_run.add_argument("--calibration", default="data/calibration/calibration_arm.json")
    p_run.add_argument("--fps", type=float, default=25.0)
    p_run.add_argument("--host", default="0.0.0.0")
    p_run.add_argument("--port", type=int, default=8000)
    p_run.add_argument("--mock-tracker", action="store_true", default=None)
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
