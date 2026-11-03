"""Prova veloce con la webcam del PC: Grok descrive cosa vede.

    python -m vision.grok.webcam_demo            finestra live: SPAZIO = chiedi a Grok, Q = esci
    python -m vision.grok.webcam_demo --once     una foto, risposta nel terminale, poi esce
    python -m vision.grok.webcam_demo --cam 1    altra telecamera

Non c'è YOLO qui (è di Visione A): Grok riceve il frame senza oggetti numerati.
Una chiamata per pressione di tasto, mai a ogni frame. L'ultimo frame inviato
viene salvato in logs/webcam_last.jpg.
"""
from __future__ import annotations

import argparse
import textwrap
import threading
import unicodedata
from pathlib import Path

import cv2

from .tasks import describe_scene

LAST = Path("logs/webcam_last.jpg")


def ask(frame) -> str:
    LAST.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(LAST), frame)
    res = describe_scene(frame, [])
    if not res.ok:
        return f"Errore: {res.error}"
    return f"{res.text}  ({res.meta['latency_ms'] / 1000:.1f} s)"


def _ascii(s: str) -> str:
    # cv2.putText non disegna le lettere accentate
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _overlay(frame, text: str) -> None:
    lines = textwrap.wrap(_ascii(text), 60)[:5]
    h = 30 + 26 * len(lines)
    cv2.rectangle(frame, (0, frame.shape[0] - h), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, frame.shape[0] - h + 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {args.cam} non disponibile")
    for _ in range(10):  # lascia assestare esposizione e bilanciamento del bianco
        cap.read()

    if args.once:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Nessun frame dalla webcam")
        print(ask(frame))
        return

    state = {"text": "SPAZIO = chiedi a Grok cosa vede, Q = esci", "busy": False}

    def worker(frame):
        state["text"] = ask(frame)
        print(state["text"])
        state["busy"] = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not state["busy"]:
            # la chiamata va in un thread: il video non si blocca mentre Grok risponde
            state.update(busy=True, text="Grok sta guardando...")
            threading.Thread(target=worker, args=(frame.copy(),), daemon=True).start()
        elif key in (ord("q"), 27):
            break
        _overlay(frame, state["text"])
        cv2.imshow("Grok vede", frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
