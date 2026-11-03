"""Calibrazione del braccio robotico per Visione A ("Occhi").

Funzionalità:
- Rilevamento marker ArUco (`DICT_4X4_50` di default) sul piano di lavoro.
- Calcolo della matrice di omografia 2D (`cv2.findHomography`) per mappare
  coordinate pixel `(px, py)` -> coordinate cartesiane `(X_mm, Y_mm)` in millimetri.
- Stima del centro e dell'angolo di presa (`angle_deg` in `[-90, +90]`) tramite
  momenti del secondo ordine / `cv2.minAreaRect` sul contorno dell'oggetto.
- Salvataggio e caricamento della calibrazione in formato JSON.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..schemas import CoordsMM

# Coordinate millimetriche di default dei 4 marker ArUco (ID 0, 1, 2, 3) sul tavolo
DEFAULT_MARKER_COORDS_MM: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),      # Alto-Sinistra
    1: (300.0, 0.0),    # Alto-Destra (300 mm su X)
    2: (300.0, 200.0),  # Basso-Destra (300 mm X, 200 mm Y)
    3: (0.0, 200.0),    # Basso-Sinistra (0 mm X, 200 mm Y)
}


def _get_aruco_dict(dict_name: str = "DICT_4X4_50"):
    dict_id = getattr(cv2.aruco, dict_name, cv2.aruco.DICT_4X4_50)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_aruco_centers(
    frame_bgr: np.ndarray,
    dict_name: str = "DICT_4X4_50",
) -> dict[int, tuple[float, float]]:
    """Rileva i marker ArUco nel frame e restituisce `{marker_id: (cx_px, cy_px)}`."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    aruco_dict = _get_aruco_dict(dict_name)

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    centers: dict[int, tuple[float, float]] = {}
    if ids is not None and len(ids) > 0:
        for marker_corners, marker_id_val in zip(corners, np.asarray(ids).flatten()):
            m_id = int(marker_id_val)
            pts = np.asarray(marker_corners).reshape((4, 2))
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            centers[m_id] = (cx, cy)
    return centers


def render_aruco_workspace(
    width: int = 640,
    height: int = 480,
    marker_size_px: int = 60,
    margin_px: int = 40,
    dict_name: str = "DICT_4X4_50",
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """Genera un'immagine con 4 marker ArUco (ID 0, 1, 2, 3) agli angoli e restituisce `(img, true_centers_px)`."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    aruco_dict = _get_aruco_dict(dict_name)

    positions = {
        0: (margin_px, margin_px),
        1: (width - margin_px - marker_size_px, margin_px),
        2: (width - margin_px - marker_size_px, height - margin_px - marker_size_px),
        3: (margin_px, height - margin_px - marker_size_px),
    }

    centers_px: dict[int, tuple[float, float]] = {}
    for m_id, (x0, y0) in positions.items():
        if hasattr(cv2.aruco, "generateImageMarker"):
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, m_id, marker_size_px)
        else:
            marker_img = cv2.aruco.drawMarker(aruco_dict, m_id, marker_size_px)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        canvas[y0 : y0 + marker_size_px, x0 : x0 + marker_size_px] = marker_bgr
        centers_px[m_id] = (x0 + (marker_size_px - 1) / 2.0, y0 + (marker_size_px - 1) / 2.0)

    return canvas, centers_px


def estimate_grasp_angle(
    frame_bgr: np.ndarray,
    bbox_norm: tuple[float, float, float, float],
) -> float:
    """Stima l'angolo di presa principale dell'oggetto in gradi `[-90.0, +90.0]`.

    Usa i momenti centrali del secondo ordine / `minAreaRect` sulla maschera dell'oggetto
    all'interno del riquadro `bbox_norm = (x, y, w, h)`.
    """
    h_img, w_img = frame_bgr.shape[:2]
    nx, ny, nw, nh = bbox_norm
    x1 = max(0, int(nx * w_img))
    y1 = max(0, int(ny * h_img))
    x2 = min(w_img, int((nx + nw) * w_img))
    y2 = min(h_img, int((ny + nh) * h_img))

    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0

    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Soglia automatica di Otsu; inverti se lo sfondo ai bordi è più chiaro del centro
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.mean(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)

    moments = cv2.moments(thresh, binaryImage=True)
    if moments["m00"] > 10:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        if abs(mu20 - mu02) > 1e-4 or abs(mu11) > 1e-4:
            angle_rad = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
            angle_deg = math.degrees(angle_rad)
            return round(max(-90.0, min(90.0, angle_deg)), 1)

    # Fallback in base al rapporto d'aspetto del bounding box
    rw, rh = x2 - x1, y2 - y1
    return 90.0 if rh > rw * 1.25 else 0.0


class ArmCalibrator:
    """Gestisce l'omografia pixel <-> millimetri e il calcolo di centro/angolo di presa."""

    def __init__(
        self,
        homography: np.ndarray | None = None,
        dict_name: str = "DICT_4X4_50",
    ):
        self.dict_name = dict_name
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        if homography is not None:
            self.set_homography(homography)

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def set_homography(self, H: np.ndarray | list[list[float]]) -> None:
        mat = np.asarray(H, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"La matrice di omografia deve essere 3x3, ricevuto {mat.shape}")
        self._H = mat
        try:
            self._H_inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            self._H_inv = None

    def calibrate_from_points(
        self,
        pixel_points: list[tuple[float, float]],
        world_points_mm: list[tuple[float, float]],
    ) -> np.ndarray:
        """Calcola l'omografia da almeno 4 coppie `(px, py) -> (X_mm, Y_mm)`."""
        if len(pixel_points) < 4 or len(world_points_mm) < 4:
            raise ValueError("Servono almeno 4 punti per calcolare l'omografia.")
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points_mm, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("Impossibile calcolare l'omografia dai punti forniti.")
        self.set_homography(H)
        return self._H

    def calibrate_from_aruco(
        self,
        frame_bgr: np.ndarray,
        marker_coords_mm: dict[int, tuple[float, float]] | None = None,
    ) -> tuple[bool, dict[int, tuple[float, float]]]:
        """Rileva i marker ArUco nell'immagine e calcola l'omografia.

        Restituisce `(success, detected_centers_px)`.
        """
        ref_mm = marker_coords_mm or DEFAULT_MARKER_COORDS_MM
        detected = detect_aruco_centers(frame_bgr, self.dict_name)

        common_ids = [m_id for m_id in ref_mm if m_id in detected]
        if len(common_ids) < 4:
            return False, detected

        px_pts = [detected[m_id] for m_id in common_ids]
        mm_pts = [ref_mm[m_id] for m_id in common_ids]
        self.calibrate_from_points(px_pts, mm_pts)
        return True, detected

    def pixel_to_mm(self, px: float, py: float) -> tuple[float, float]:
        """Converte coordinate pixel `(px, py)` in millimetri `(X_mm, Y_mm)`."""
        if self._H is None:
            raise RuntimeError("Calibrazione non attiva: carica o calcola prima l'omografia.")
        pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(pt, self._H)[0, 0]
        return (round(float(mapped[0]), 2), round(float(mapped[1]), 2))

    def mm_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """Converte coordinate millimetriche `(X_mm, Y_mm)` in pixel `(px, py)`."""
        if self._H_inv is None:
            raise RuntimeError("Omografia inversa non disponibile.")
        pt = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(pt, self._H_inv)[0, 0]
        return (round(float(mapped[0]), 2), round(float(mapped[1]), 2))

    def compute_object_coords(
        self,
        frame_bgr: np.ndarray,
        bbox_norm: tuple[float, float, float, float],
    ) -> CoordsMM | None:
        """Restituisce `CoordsMM(x, y, angle_deg)` per un oggetto con bbox normalizzato."""
        if not self.is_calibrated:
            return None
        h_img, w_img = frame_bgr.shape[:2]
        nx, ny, nw, nh = bbox_norm
        cx_px = (nx + nw / 2.0) * w_img
        cy_px = (ny + nh / 2.0) * h_img
        x_mm, y_mm = self.pixel_to_mm(cx_px, cy_px)
        angle_deg = estimate_grasp_angle(frame_bgr, bbox_norm)
        return CoordsMM(x=x_mm, y=y_mm, angle_deg=angle_deg)

    def save(self, path: str | Path) -> Path:
        """Salva la matrice di omografia su file JSON."""
        if self._H is None:
            raise RuntimeError("Nessuna omografia da salvare.")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dict_name": self.dict_name,
            "homography": self._H.tolist(),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> ArmCalibrator:
        """Carica la calibrazione da un file JSON."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            homography=np.array(data["homography"], dtype=np.float64),
            dict_name=data.get("dict_name", "DICT_4X4_50"),
        )
