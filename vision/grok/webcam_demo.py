"""Prova con la webcam del PC: oggetti con riquadro e numero, click per selezionarli.

    python -m vision.grok.webcam_demo [--cam 0] [--no-yolo]
    python -m vision.grok.webcam_demo --once      una foto: YOLO + Grok, salva logs/webcam_marked.jpg

Nella finestra:
    YOLO (tracker di Visione A) segue dal vivo gli oggetti che conosce (80 classi COCO)
    G       Grok cerca gli altri oggetti, anche quelli che YOLO non conosce (riquadri fermi)
    N       Grok dà un nome a tutti gli oggetti numerati
    SPAZIO  Grok descrive la scena
    click   seleziona l'oggetto: è l'id che si passerebbe al robot
    C       cancella gli oggetti trovati da Grok
    Q / Esc esci

Le chiamate a Grok partono solo da un tasto (mai a ogni frame) e girano in un
thread, così il video non si blocca.
"""
from __future__ import annotations

import argparse
import json
import threading
import unicodedata
from pathlib import Path

import cv2

from .local_resolver import display
from .marks import draw_marks
from .schemas import TrackedObject
from .tasks import describe_scene, detect_objects, name_objects

WIN = "Grok vede"
LOGS = Path("logs")


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


def load_tracker(enabled: bool):
    """Tracker YOLO di Visione A, o None se YOLO non è disponibile."""
    if not enabled:
        return None
    try:
        from vision.eyes.tracking.tracker import ObjectTracker
        tracker = ObjectTracker(use_mock=False)
    except Exception as e:
        print(f"YOLO non disponibile ({e}): solo Grok.")
        return None
    if tracker.use_mock:  # ultralytics mancante: il rilevatore di ripiego a colori non va bene su una webcam vera
        print("YOLO non disponibile (installa ultralytics): solo Grok.")
        return None
    return tracker


def _ascii(s: str) -> str:
    # cv2.putText non disegna le lettere accentate
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def render(frame, objects: list[dict], selected: int | None, text: str):
    out = draw_marks(frame, objects)
    h, w = out.shape[:2]
    for o in objects:
        x, y, bw, bh = o["bbox"]
        x1, y1, x2, y2 = int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)
        if o["id"] == selected:
            cv2.rectangle(out, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 3)
        # nome dentro il riquadro, sotto il numero: fuori si sovrappone ai riquadri vicini
        label, pos = _ascii(o.get("name") or o["label"]), (x1 + 4, min(y2 - 4, y1 + 50))
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)  # bordo
        cv2.putText(out, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(out, _ascii(text)[:110], (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def object_at(objects: list[dict], fx: float, fy: float) -> dict | None:
    """Oggetto sotto il click (coordinate normalizzate); se si sovrappongono, il più piccolo."""
    hits = [o for o in objects
            if o["bbox"][0] <= fx <= o["bbox"][0] + o["bbox"][2] and o["bbox"][1] <= fy <= o["bbox"][1] + o["bbox"][3]]
    return min(hits, key=lambda o: o["bbox"][2] * o["bbox"][3], default=None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--no-yolo", action="store_true", help="solo Grok, senza tracker YOLO")
    ap.add_argument("--once", action="store_true", help="una foto, YOLO + Grok, poi esce")
    args = ap.parse_args()

    tracker = load_tracker(not args.no_yolo)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Webcam {args.cam} non disponibile")
    for _ in range(10):  # lascia assestare esposizione e bilanciamento del bianco
        cap.read()

    def yolo(frame) -> list[dict]:
        return [o.model_dump() for o in tracker.process_frame(frame)] if tracker else []

    if args.once:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Nessun frame dalla webcam")
        found = yolo(frame)
        res = detect_objects(frame, known=found)
        objects = found + [o.model_dump() for o in res.objects]
        LOGS.mkdir(exist_ok=True)
        cv2.imwrite(str(LOGS / "webcam_marked.jpg"), render(frame, objects, None, ""))
        print(f"YOLO: {len(found)} oggetti · Grok: {len(res.objects)} oggetti in "
              f"{res.meta.get('latency_ms', 0) / 1000:.1f} s" + ("" if res.ok else f" · errore: {res.error}"))
        for o in objects:
            print(f"  {o['id']:>5}  {o['label']:<15} {display_obj(o)}  bbox={[round(v, 2) for v in o['bbox']]}")
        print(f"Immagine con i riquadri: {LOGS / 'webcam_marked.jpg'}")
        return

    # Stato condiviso tra il ciclo video, il click e i thread delle chiamate a Grok.
    # ponytail: un dict senza lock basta per una demo (un solo thread Grok alla volta).
    st = {"grok": [], "objects": [], "selected": None, "busy": False,
          "text": "G = Grok trova oggetti   N = nomi   SPAZIO = descrivi   click = seleziona   Q = esci"}

    def on_click(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or st.get("size") is None:
            return
        w, h = st["size"]
        o = object_at(st["objects"], x / w, y / h)
        st["selected"] = o["id"] if o else None
        st["text"] = f"Selezionato {o['id']}: {display_obj(o)}" if o else "Nessun oggetto qui"
        if o:
            print("Oggetto selezionato (da passare al robot):", json.dumps(o, ensure_ascii=False))

    def run_grok(job, frame, objects):
        try:
            if job == "detect":
                yolo_objs = [o for o in objects if o["id"] < 1000]
                res = detect_objects(frame, known=yolo_objs)
                if res.ok:
                    st["grok"] = [o.model_dump() for o in res.objects]
                st["text"] = (f"Grok ha trovato {len(res.objects)} oggetti" if res.ok else f"Errore: {res.error}")
            elif job == "name":
                res = name_objects(draw_marks(frame, objects), objects)
                for n in res.objects:
                    if tracker and n.id < 1000:
                        tracker.set_object_metadata(n.id, n.name, n.color)
                    for o in st["grok"]:
                        if o["id"] == n.id:
                            o.update(name=n.name, color=n.color)
                st["text"] = f"Nomi aggiornati ({len(res.objects)})" if res.ok else f"Errore: {res.error}"
            else:
                res = describe_scene(frame, objects)
                st["text"] = res.text if res.ok else f"Errore: {res.error}"
            st["text"] += f"  ({res.meta.get('latency_ms', 0) / 1000:.1f} s)"
            print(st["text"])
        finally:
            st["busy"] = False

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)
    jobs = {ord("g"): "detect", ord("n"): "name", ord(" "): "describe"}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st["size"] = (frame.shape[1], frame.shape[0])
        st["objects"] = yolo(frame) + st["grok"]
        key = cv2.waitKey(1) & 0xFF
        if key in jobs and not st["busy"]:
            st.update(busy=True, text="Grok sta guardando...")
            threading.Thread(target=run_grok, args=(jobs[key], frame.copy(), list(st["objects"])), daemon=True).start()
        elif key == ord("c"):
            st.update(grok=[], text="Oggetti di Grok cancellati")
        elif key in (ord("q"), 27):
            break
        cv2.imshow(WIN, render(frame, st["objects"], st["selected"], st["text"]))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
