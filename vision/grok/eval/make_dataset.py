"""Genera un piccolo dataset sintetico per provare la pipeline di valutazione.

Forme colorate su sfondo uniforme + 20 comandi (inclusi casi ambigui e oggetti
assenti). Al museo si sostituisce eval/data con foto vere nello stesso formato.

    python -m vision.grok.eval.make_dataset
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).parent / "data"
W, H = 640, 480
BGR = {"rosso": (40, 40, 220), "verde": (60, 170, 60), "blu": (200, 90, 30), "giallo": (40, 210, 230),
       "nero": (30, 30, 30)}
LABEL = {"cerchio": "circle", "quadrato": "square", "triangolo": "triangle"}

# scena -> [(id, forma, colore, centro x, centro y, raggio in px, nome)]
SCENES = {
    "scene_01": [(1, "cerchio", "rosso", 120, 150, 50, "cerchio rosso"),
                 (2, "quadrato", "rosso", 320, 140, 55, "quadrato rosso"),
                 (3, "triangolo", "blu", 500, 160, 60, "triangolo blu"),
                 (4, "cerchio", "verde", 300, 350, 50, "cerchio verde")],
    "scene_02": [(1, "quadrato", "giallo", 140, 240, 60, "quadrato giallo"),
                 (2, "quadrato", "blu", 330, 240, 60, "quadrato blu"),
                 (3, "cerchio", "giallo", 510, 240, 55, "cerchio giallo")],
    "scene_03": [(5, "cerchio", "rosso", 170, 240, 90, "cerchio rosso grande"),
                 (6, "cerchio", "rosso", 400, 260, 35, "cerchio rosso piccolo"),
                 (7, "triangolo", "verde", 540, 150, 55, "triangolo verde")],
    "scene_04": [(10, "quadrato", "nero", 100, 120, 50, "quadrato nero"),
                 (11, "triangolo", "giallo", 300, 120, 55, "triangolo giallo"),
                 (12, "cerchio", "blu", 300, 340, 55, "cerchio blu"),
                 (13, "quadrato", "verde", 530, 330, 55, "quadrato verde")],
}

# (scena, comando, id atteso o None, ambiguo atteso)
COMMANDS = [
    ("scene_01", "prendi il cerchio verde", 4, False),
    ("scene_01", "prendi il triangolo", 3, False),
    ("scene_01", "prendi quello rosso", None, True),
    ("scene_01", "prendi il cerchio", None, True),
    ("scene_01", "prendi la stella gialla", None, False),
    ("scene_01", "prendi il numero 2", 2, False),
    ("scene_02", "prendi il quadrato blu", 2, False),
    ("scene_02", "prendi l'oggetto giallo", None, True),
    ("scene_02", "afferra il cerchio giallo", 3, False),
    ("scene_02", "prendi il triangolo", None, False),
    ("scene_02", "indica il quadrato", None, True),
    ("scene_03", "prendi il cerchio rosso piccolo", 6, False),
    ("scene_03", "prendi il cerchio rosso", None, True),
    ("scene_03", "guarda il triangolo verde", 7, False),
    ("scene_03", "prendi il quadrato", None, False),
    ("scene_03", "prendi il cerchio più grande", 5, False),
    ("scene_04", "prendi il quadrato verde", 13, False),
    ("scene_04", "prendi il quadrato", None, True),
    ("scene_04", "prendi quello più a destra", 13, False),
    ("scene_04", "prendi la palla rossa", None, False),
]


def draw_shape(img: np.ndarray, shape: str, color: tuple, cx: int, cy: int, r: int) -> list[float]:
    """Disegna la forma piena e restituisce il bbox normalizzato [x, y, w, h]."""
    if shape == "cerchio":
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    elif shape == "quadrato":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, -1)
    else:
        pts = np.array([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    return [round((cx - r) / W, 4), round((cy - r) / H, 4), round(2 * r / W, 4), round(2 * r / H, 4)]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for i, (scene, items) in enumerate(SCENES.items(), 1):
        img = np.full((H, W, 3), (225, 225, 225), np.uint8)
        objects = [{"id": oid, "label": LABEL[shape], "name": name, "conf": 0.9,
                    "bbox": draw_shape(img, shape, BGR[color], cx, cy, r), "color": color}
                   for oid, shape, color, cx, cy, r, name in items]
        cv2.imwrite(str(DATA / f"{scene}.jpg"), img)
        (DATA / f"{scene}.json").write_text(json.dumps(
            {"source": "arm", "frame_id": i, "ts": 0.0, "objects": objects}, ensure_ascii=False, indent=1),
            encoding="utf-8")
    with (DATA / "commands.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "text", "expected_target_id", "expected_ambiguous"])
        for scene, text, target, amb in COMMANDS:
            w.writerow([scene, text, "" if target is None else target, str(amb).lower()])
    print(f"{len(SCENES)} scene e {len(COMMANDS)} comandi in {DATA}")


if __name__ == "__main__":
    main()
