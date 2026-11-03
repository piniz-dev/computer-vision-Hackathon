"""Visione A ("Occhi"): percezione locale, tracking, Set-of-Mark, calibrazione ArUco,
telemetria cane, verifica locale e streaming MJPEG/WebSocket.
"""
from .calibration.aruco_calib import ArmCalibrator, detect_aruco_centers, estimate_grasp_angle
from .capture.reader import VideoReader, generate_sample_video, make_synthetic_frame
from .dog.telemetry import DogNavigator, check_person_safety_stop, compute_dog_telemetry
from .marks.som import draw_marks
from .pipeline import EyePipeline
from .schemas import CoordsMM, DogTelemetry, EyeObject, FramePayload, ZoneVerificationResult
from .state.vision_state import LiveVisionState
from .tracking.tracker import ObjectTracker, SimpleByteTracker, compute_iou
from .verification.verifier import check_appearance, check_disappearance, verify_action_local

__all__ = [
    "ArmCalibrator",
    "CoordsMM",
    "DogNavigator",
    "DogTelemetry",
    "EyeObject",
    "EyePipeline",
    "FramePayload",
    "LiveVisionState",
    "ObjectTracker",
    "SimpleByteTracker",
    "VideoReader",
    "ZoneVerificationResult",
    "check_appearance",
    "check_disappearance",
    "check_person_safety_stop",
    "compute_dog_telemetry",
    "compute_iou",
    "detect_aruco_centers",
    "draw_marks",
    "estimate_grasp_angle",
    "generate_sample_video",
    "make_synthetic_frame",
    "verify_action_local",
]
