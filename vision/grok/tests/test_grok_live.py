"""YOLO-World -> oggetti del gruppo, e sagoma GrabCut (senza modelli né rete)."""
from types import SimpleNamespace

import numpy as np
import pytest

from vision.grok.geometry import refine_box
from vision.grok.open_vocab import ID_OFFSET, to_objects


class _T:  # finto tensore torch
    def __init__(self, a):
        self.a = np.asarray(a, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def _result(ids):
    boxes = SimpleNamespace(xyxyn=_T([[0.1, 0.1, 0.2, 0.4], [0.0, 0.0, 0.9, 0.9]]), conf=_T([0.4, 0.8]),
                            cls=_T([0, 1]), id=None if ids is None else _T(ids))
    return SimpleNamespace(boxes=boxes, names={0: "pen", 1: "pencil"})


def test_to_objects():
    objs = to_objects(_result([3, 4]))
    assert len(objs) == 1  # il riquadro grande (81% del frame) è scartato
    o = objs[0]
    assert o["id"] == ID_OFFSET + 3 and o["label"] == "pen" and o["bbox"] == pytest.approx([0.1, 0.1, 0.1, 0.3])
    assert to_objects(_result(None)) == []  # nessun id di tracking: non ancora confermato


def test_refine_box_finds_object_and_rejects_empty():
    frame = np.full((240, 320, 3), 200, np.uint8)
    frame[80:160, 100:140] = (0, 0, 255)  # "penna" rossa: x 100-140, y 80-160
    bbox, contour = refine_box(frame, [80 / 320, 60 / 240, 80 / 320, 120 / 240])
    assert bbox == pytest.approx([100 / 320, 80 / 240, 40 / 320, 80 / 240], abs=0.03)
    assert len(contour) >= 4
    assert refine_box(np.full((240, 320, 3), 200, np.uint8), [0.3, 0.3, 0.3, 0.3]) is None
