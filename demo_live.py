"""Demo dal vivo sul PC: Visione A ("Occhi") + Visione B ("Cervello") nella stessa finestra.

  [G]          -> Grok trova gli oggetti (anche quelli che YOLO non conosce, es. penne e matite)
                  e YOLO-World impara a seguirli dal vivo, anche se li muovi in mano
  [N]          -> Grok dà un nome breve a tutti gli oggetti numerati
  [S]          -> Scrivi un comando nella finestra (es. 'prendi la penna blu'), Invio per mandarlo
  [D] / SPAZIO -> Grok descrive la scena
  [Click Sx]   -> Seleziona un oggetto sotto il cursore (sagoma + pannello); la scelta va al
                  robot in logs/selection.json (anche da [S] e quando la selezione si riaggancia)
  [Z]          -> Zoom sull'oggetto attualmente selezionato (con S o Click Sx)
  [Click Dx]   -> Dezooma (riporta alla vista panoramica 1.0x, come [0] / [R])
  [C]          -> Dimentica tutto: oggetti di Grok e classi imparate (che altrimenti restano
                  in logs/vocab.json e tornano al prossimo avvio)
  [1-9]        -> Seleziona l'oggetto con quel numero e zooma
  [K]          -> Mostra/nascondi percentuale di confidenza
  [+] / [-]    -> Alza/abbassa la soglia minima di confidenza (+/- 5%)
  [T]          -> Imposta target per il cane robotico
  [H]          -> Aiuto
  [Q] / ESC    -> Esci

Senza XAI_API_KEY nel .env la demo parte in modalità finta (GROK_MOCK=1).

Utilizzo:
  python demo_live.py                 # Webcam 0 con YOLOv8 ONNX + YOLO-World + Grok (.env)
  python demo_live.py --cam 1         # Seconda webcam
  python demo_live.py --no-yolo       # Solo Grok (senza YOLO)
  python demo_live.py --no-world      # Senza YOLO-World (gli oggetti di Grok restano fermi)
  python demo_live.py --once          # Scatta una foto, YOLO + Grok, salva in logs/webcam_marked.jpg ed esce
  python demo_live.py --replay        # Da video/simulazione replay
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
import threading
import time
import unicodedata
from pathlib import Path

import cv2

from vision.eyes.depth import CLASS_PRIORS_CM, estimate_distance_m, format_distance
from vision.eyes.marks import draw_marks
from vision.eyes.pipeline import EyePipeline
from vision.grok.geometry import refine_box
from vision.grok.handoff import send_selection
from vision.grok.local_resolver import display
from vision.grok.marks import object_at, reacquire
from vision.grok.open_vocab import OpenVocab, load_vocab, save_vocab
from vision.grok.schemas import TrackedObject
from vision.grok.tasks import describe_scene, detect_objects, name_objects, select

# vision.grok.client ha già caricato il .env. Senza chiave si va in modalità finta.
if not os.getenv("XAI_API_KEY") and not os.getenv("GROK_MOCK"):
    os.environ["GROK_MOCK"] = "1"

WIN = "Grok Hackathon - Occhi + Cervello Live"
LOGS_DIR = Path("logs")
VOCAB_PATH = LOGS_DIR / "vocab.json"          # classi imparate da Grok, ricaricate al prossimo avvio
SELECTION_PATH = LOGS_DIR / "selection.json"  # ultima scelta, letta dal braccio (formato in vision/grok/handoff.py)
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_COLOR = {"ok": (90, 220, 90), "wait": (0, 200, 255), "err": (70, 70, 255)}  # BGR
HELP = [
    "G       Grok trova gli oggetti (poi YOLO-World li segue)",
    "N       Grok da' i nomi",
    "S       scrivi un comando (es. prendi la penna blu)",
    "D/Spazio  descrivi la scena",
    "Click Sx  seleziona oggetto (o seleziona telecamera in griglia/PiP)",
    "Z       zoom sull'oggetto selezionato",
    "Click Dx  dezooma (oppure 0 / R)",
    "V / Tab   telecamera successiva (1 -> 2 -> ... -> N)",
    "B         telecamera precedente",
    "M         layout telecamere (singola / griglia N-cam / PiP)",
    "C       dimentica tutto (oggetti di Grok e classi imparate)",
    "+/-     soglia confidenza    K  mostra %",
    "T       target cane",
    "H       chiudi aiuto    Q/Esc  esci",
]


def _ascii(s: str) -> str:
    """cv2.putText non sa scrivere le lettere accentate: "è" -> "e"."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()


def detect_cameras(max_devices: int = 8) -> list[int]:
    """Scansiona e restituisce gli indici di tutte le webcam funzionanti sul sistema."""
    available = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def get_grid_cell(x: int, y: int, w: int, h: int, n: int) -> tuple[int, float, float]:
    """Mappa il click (x, y) sul canvas (w, h) all'indice della telecamera e coordinate normalizzate."""
    if n <= 1:
        return 0, x / max(1, w), y / max(1, h)
    elif n == 2:
        hw = w // 2
        if x < hw:
            return 0, x / max(1, hw), y / max(1, h)
        else:
            return 1, (x - hw) / max(1, w - hw), y / max(1, h)
    else:
        # Layout 2x2 per 3 o 4 telecamere
        cw, ch = w // 2, h // 2
        col = 0 if x < cw else 1
        row = 0 if y < ch else 1
        idx = row * 2 + col
        rx = (x if col == 0 else (x - cw)) / max(1, cw)
        ry = (y if row == 0 else (y - ch)) / max(1, ch)
        return idx, rx, ry


def _compose_grid(canvas, frames: dict[str, np.ndarray], sources: list[str], labels: dict[str, str], active_src: str) -> np.ndarray:
    """Compone una griglia dinamica per N telecamere con evidenziazione della camera attiva."""
    h, w = canvas.shape[:2]
    n = len(sources)
    if n <= 1:
        return frames.get(sources[0], canvas)

    out = np.zeros_like(canvas)
    if n == 2:
        hw = w // 2
        for idx, src in enumerate(sources[:2]):
            f = frames.get(src)
            if f is None or f.size == 0:
                continue
            x1 = 0 if idx == 0 else hw
            x2 = hw if idx == 0 else w
            tile = cv2.resize(f, (x2 - x1, h))
            out[:, x1:x2] = tile
            is_act = (src == active_src)
            col = (0, 255, 100) if is_act else (120, 120, 120)
            cv2.rectangle(out, (x1, 0), (x2 - 1, h - 1), col, 3 if is_act else 1)
            lbl = labels.get(src, src.upper())
            tag = f"{lbl} [ATTIVA]" if is_act else f"{lbl} [V/Clicca]"
            _box(out, x1 + 10, h - 38, x1 + 250, h - 10, 0.75)
            _text(out, tag, x1 + 16, h - 18, col, 0.44, 2 if is_act else 1)
        cv2.line(out, (hw, 0), (hw, h), (0, 230, 255), 2)
        return out

    # N >= 3: layout 2x2
    cols, rows = 2, 2
    cw, ch = w // 2, h // 2
    for idx, src in enumerate(sources[:4]):
        r, c = idx // cols, idx % cols
        x1, y1 = c * cw, r * ch
        x2 = (x1 + cw) if c < cols - 1 else w
        y2 = (y1 + ch) if r < rows - 1 else h
        f = frames.get(src)
        if f is not None and f.size > 0:
            tile = cv2.resize(f, (x2 - x1, y2 - y1))
            out[y1:y2, x1:x2] = tile
        is_act = (src == active_src)
        col = (0, 255, 100) if is_act else (120, 120, 120)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), col, 3 if is_act else 1)
        lbl = labels.get(src, src.upper())
        tag = f"{lbl} [ATTIVA]" if is_act else f"{lbl} [V/Clicca]"
        _box(out, x1 + 8, y2 - 32, x1 + 240, y2 - 8, 0.75)
        _text(out, tag, x1 + 14, y2 - 14, col, 0.40, 2 if is_act else 1)

    # Se n == 3, la 4a cella (in basso a destra) mostra la scheda riepilogo multi-cam
    if n == 3:
        x1, y1, x2, y2 = cw, ch, w, h
        _box(out, x1, y1, x2, y2, 0.90)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 230, 255), 1)
        _text(out, "PANNELLO MULTI-CAMERA (N=3)", x1 + 18, y1 + 30, (0, 230, 255), 0.52, 2)
        _text(out, f"Camera Attiva: {labels.get(active_src, active_src)}", x1 + 18, y1 + 60, (0, 255, 100), 0.44, 2)
        _text(out, "Elenco Telecamere:", x1 + 18, y1 + 90, (230, 230, 230), 0.44, 1)
        for i, s in enumerate(sources):
            mark = "-> [ATTIVA]" if s == active_src else "   [in ascolto]"
            col_s = (0, 255, 120) if s == active_src else (180, 180, 180)
            _text(out, f" Cam {i+1}: {labels.get(s, s)} {mark}", x1 + 22, y1 + 112 + 20 * i, col_s, 0.38)
        _text(out, "Scorciatoie Tastiera:", x1 + 18, y1 + 185, (230, 230, 230), 0.44, 1)
        _text(out, " [V] / [TAB] : Telecamera successiva", x1 + 22, y1 + 205, (190, 190, 190), 0.38)
        _text(out, " [B]         : Telecamera precedente", x1 + 22, y1 + 225, (190, 190, 190), 0.38)
        _text(out, " [M]         : Cambia layout (griglia / singola / PiP)", x1 + 22, y1 + 245, (190, 190, 190), 0.38)
        _text(out, " Clicca su un riquadro per selezionare", x1 + 22, y1 + 265, (0, 230, 255), 0.38)

    # Linee divisorie
    cv2.line(out, (cw, 0), (cw, h), (0, 230, 255), 2)
    cv2.line(out, (0, ch), (w, ch), (0, 230, 255), 2)
    return out


def display_obj(o: dict) -> str:
    return display(TrackedObject.model_validate(o))


def _learn_size(label: str, size_cm) -> None:
    """Aggiunge la misura reale stimata da Grok alla tabella usata da Visione A per la distanza.
    ponytail: scriviamo in memoria nel dict di A (setdefault: le misure che hanno già restano);
    se A aggiunge un modo ufficiale per registrare misure, usare quello."""
    if size_cm and len(size_cm) == 2 and all(v > 0 for v in size_cm):
        CLASS_PRIORS_CM.setdefault(label.strip().lower(), (float(size_cm[0]), float(size_cm[1])))


def _distance(o: dict) -> float | None:
    return o.get("distance_m") or estimate_distance_m(o.get("label", "object"), o["bbox"])


def _source(obj_id: int) -> str:
    """Chi ha trovato l'oggetto, dagli intervalli di id: YOLO < 1000 <= Grok < 2000 <= YOLO-World."""
    return "YOLO" if obj_id < 1000 else "Grok" if obj_id < 2000 else "YOLO-World"


def _box(frame, x1, y1, x2, y2, alpha=0.75):
    """Rettangolo scuro semitrasparente (sfondo dei pannelli)."""
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha)).astype(roi.dtype) + int(18 * alpha)


def _text(frame, s, x, y, color=(255, 255, 255), scale=0.45, thick=1):
    cv2.putText(frame, _ascii(s), (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def _draw_top(frame, fps, n_objs, zoom, live_status, active_cam_idx=0, n_cams=1, active_label="", view_mode="single"):
    w = frame.shape[1]
    _box(frame, 0, 0, w, 44)
    cam_hint = f"   V/B cam (1..{n_cams})   M layout ({view_mode})" if n_cams > 1 else ""
    _text(frame, f"H aiuto   G trova   N nomi   S comando   Z zoom sel   Click Dx dezooma{cam_hint}   D descrivi   Q esci", 10, 18, (0, 230, 255))
    cam_badge = f"CAM {active_cam_idx + 1}/{n_cams}: {active_label}" if n_cams > 1 else f"CAM: {active_label}"
    _text(frame, f"[{cam_badge}]  FPS {fps:.0f}  |  {n_objs} oggetti  |  zoom {zoom:.1f}x  |  live: {live_status}",
          10, 36, (190, 190, 190), 0.4)


def _draw_status(frame, msg, kind, latency):
    h, w = frame.shape[:2]
    lines = textwrap.wrap(_ascii(msg), max(30, w // 9))[:3] or [""]
    top = h - 16 - 20 * len(lines)
    _box(frame, 0, top, w, h)
    cv2.rectangle(frame, (0, top), (6, h), STATUS_COLOR[kind], -1)
    for i, line in enumerate(lines):
        _text(frame, line, 16, top + 20 + 20 * i, scale=0.5)
    if latency:
        _text(frame, latency, w - 150, top + 20, (150, 150, 150), 0.4)


def _draw_panel(frame, obj, sel_id, last_cmd):
    """Scheda dell'oggetto selezionato, in alto a destra."""
    w = frame.shape[1]
    if obj is None:
        lines = [(f"#{sel_id} non e' piu' visibile", (0, 200, 255))]
    else:
        lines = [(f"#{obj['id']}  {display_obj(obj)}", (0, 255, 255)),
                 (f"tipo: {obj['label']} ({_source(obj['id'])})", (230, 230, 230)),
                 (f"colore: {obj.get('color') or '-'}", (230, 230, 230)),
                 (f"distanza: {format_distance(_distance(obj))}", (230, 230, 230)),
                 (f"sicurezza: {int(round(obj.get('conf', 0) * 100))}%", (230, 230, 230))]
        if obj.get("tags"):
            lines.append((f"tag: {', '.join(obj['tags'][:4])}", (230, 230, 230)))
        if last_cmd and last_cmd.get("target_id") == obj["id"]:
            lines.append((f"azione: {last_cmd['action']} ({last_cmd['source']})", (90, 220, 90)))
    x1, y1 = w - 290, 54
    _box(frame, x1, y1, w - 8, y1 + 14 + 22 * len(lines))
    for i, (s, color) in enumerate(lines):
        _text(frame, s[:38], x1 + 10, y1 + 24 + 22 * i, color)


def _draw_input(frame, buf):
    h, w = frame.shape[:2]
    y = h // 2 - 30
    _box(frame, 20, y, w - 20, y + 64, 0.85)
    cv2.rectangle(frame, (20, y), (w - 20, y + 64), (0, 230, 255), 2)
    _text(frame, "Comando per il robot  (Invio = manda, Esc = annulla)", 34, y + 22, (0, 230, 255), 0.45)
    blink = "_" if int(time.time() * 2) % 2 else " "
    _text(frame, buf[-55:] + blink, 34, y + 50, scale=0.6)


def _draw_help(frame):
    h, w = frame.shape[:2]
    x1, y1 = max(10, w // 2 - 250), max(50, h // 2 - 130)
    _box(frame, x1, y1, x1 + 500, y1 + 40 + 22 * len(HELP), 0.85)
    _text(frame, "Comandi", x1 + 14, y1 + 26, (0, 230, 255), 0.55)
    for i, s in enumerate(HELP):
        _text(frame, s, x1 + 14, y1 + 52 + 22 * i)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cams", nargs="*", default=None, help="Elenco indici o stream delle telecamere (es. --cams 0 2 4) oppure 'auto'")
    parser.add_argument("--cam", type=int, default=0, help="Indice prima webcam (default 0)")
    parser.add_argument("--cam2", type=str, default=None, help="Indice o stream seconda webcam ('auto', numero es. 2, stream o 'none')")
    parser.add_argument("--replay", action="store_true", help="Usa simulazione replay al posto della webcam")
    parser.add_argument("--no-yolo", action="store_true", help="Disabilita YOLO locale (solo rilevamento via Grok)")
    parser.add_argument("--no-world", action="store_true", help="Disabilita YOLO-World (oggetti di Grok fermi)")
    parser.add_argument("--once", action="store_true", help="Scatta una foto, rileva con YOLO + Grok, salva logs/webcam_marked.jpg ed esce")
    parser.add_argument("--model", type=str, default="data/models/yolov8n.onnx", help="Pesi YOLO (.onnx o .pt)")
    parser.add_argument("--min-conf", type=float, default=0.40, help="Soglia minima di confidenza per visualizzare riquadri (default 0.40)")
    args = parser.parse_args()

    min_conf = max(0.05, min(0.95, args.min_conf))
    show_conf = True  # K lo cambia

    # Nomi e ruoli standard per le N telecamere
    STANDARD_NAMES = ["arm", "dog", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8"]
    STANDARD_ROLES = ["ARM (Braccio)", "DOG (Cane/Mobile)", "CAM 3 (Panoramica)", "CAM 4", "CAM 5", "CAM 6", "CAM 7", "CAM 8"]

    cam_targets = []
    if args.replay:
        cam_targets = [("arm", "data/replays/sample_arm.mp4", "replay", "ARM (Replay)")]
    elif args.cams:
        if len(args.cams) == 1 and args.cams[0].lower() == "auto":
            print("[Cam] Scansione automatica telecamere collegate...")
            avail = detect_cameras(8)
            cam_indices = avail if avail else [args.cam]
        else:
            cam_indices = [(int(c) if c.isdigit() else c) for c in args.cams]
        for i, t in enumerate(cam_indices):
            sname = STANDARD_NAMES[i] if i < len(STANDARD_NAMES) else f"cam{i+1}"
            role = STANDARD_ROLES[i] if i < len(STANDARD_ROLES) else f"CAM {i+1}"
            mode = "webcam" if isinstance(t, int) else "stream"
            lbl = f"{role} [{f'Dev {t}' if mode == 'webcam' else 'Stream'}]"
            cam_targets.append((sname, t, mode, lbl))
    elif args.cam2 is not None:
        cam_targets.append(("arm", args.cam, "webcam", f"ARM [Dev {args.cam}]"))
        val = str(args.cam2).strip()
        if val.lower() == "auto":
            avail = detect_cameras(8)
            others = [c for c in avail if c != args.cam]
            if others:
                cam_targets.append(("dog", others[0], "webcam", f"DOG [Dev {others[0]}]"))
        elif val.lower() not in ("none", "no", "false", "off", ""):
            t = int(val) if val.isdigit() else val
            m = "webcam" if isinstance(t, int) else "stream"
            cam_targets.append(("dog", t, m, f"DOG [{f'Dev {t}' if m == 'webcam' else 'Stream'}]"))
    else:
        # Default: auto-detection intelligente di tutte le telecamere connesse!
        print("[Cam] Scansione automatica di tutte le telecamere collegate...")
        avail = detect_cameras(8)
        cam_indices = avail if avail else [args.cam]
        for i, t in enumerate(cam_indices):
            sname = STANDARD_NAMES[i] if i < len(STANDARD_NAMES) else f"cam{i+1}"
            role = STANDARD_ROLES[i] if i < len(STANDARD_ROLES) else f"CAM {i+1}"
            lbl = f"{role} [Dev {t}]"
            cam_targets.append((sname, t, "webcam", lbl))

    print(f"\n[Cam] {len(cam_targets)} telecamere pronte:")
    for sname, t, m, lbl in cam_targets:
        print(f"  * {lbl} (source='{sname}', device={t}, mode='{m}')")
    print()

    # Video e tracker YOLO di Visione A per tutte le N telecamere
    pipeline = EyePipeline(fps=30.0)
    for sname, t, m, _lbl in cam_targets:
        pipeline.add_source(
            source=sname,
            mode="replay" if args.replay else m,
            target=t,
            use_mock_tracker=args.no_yolo,
            model_name=args.model,
            conf_threshold=0.25,
            ema_alpha=0.65,
        )

    # Avvia acquisizione fluida in background per tutte le sorgenti registrate
    for node in pipeline.nodes.values():
        node.reader.start_background()

    arm_node = pipeline.nodes.get("arm")
    zoom_mgr = arm_node.zoom if arm_node else None

    # --no-yolo: tracker senza modello e senza oggetti finti, così gli oggetti arrivano solo da Grok
    if args.no_yolo:
        for node in pipeline.nodes.values():
            node.tracker.use_mock = False
            node.tracker._yolo_model = None
            node.tracker._onnx_net = None

    # --once: una foto, YOLO + Grok, immagine salvata e risultato nel terminale
    if args.once:
        print("[Demo Once] Acquisizione fotogramma...")
        pipeline.step("arm")
        raw_frame = pipeline.state.get_frame("arm")
        yolo_objs = pipeline.state.get_objects("arm")

        print(f"[Demo Once] YOLO ha rilevato {len(yolo_objs)} oggetti. Invio a Grok...")
        res = detect_objects(raw_frame, known=yolo_objs)
        grok_objs = []
        if res.ok:
            for o in res.objects:
                _learn_size(o.label, o.size_cm)
                d = o.model_dump()
                d["distance_m"] = estimate_distance_m(d["label"], d["bbox"])
                grok_objs.append(d)
        all_objs = yolo_objs + grok_objs

        marked = draw_marks(raw_frame, all_objs, show_conf=show_conf, show_dist=True)
        LOGS_DIR.mkdir(exist_ok=True)
        out_path = LOGS_DIR / "webcam_marked.jpg"
        cv2.imwrite(str(out_path), marked)

        print("\n" + "=" * 60)
        print(" RISULTATO SCATTO SINGOLO:")
        print(f" YOLO: {len(yolo_objs)} oggetti | Grok: {len(grok_objs)} oggetti")
        if res.meta:
            print(f" Latenza Grok: {res.meta.get('latency_ms', 0):.0f} ms")
            for why in res.meta.get("discard_reasons", []):
                print(f" Scartato: {why}")
        if not res.ok:
            print(f" Errore Grok: {res.error}")
        print("=" * 60)
        for o in all_objs:
            dist_txt = format_distance(_distance(o))
            print(f"  #{o['id']:<4} {o['label']:<15} {display_obj(o)} | dist={dist_txt:<6} | bbox={[round(v, 2) for v in o['bbox']]}")
        print(f"\nImmagine salvata in: {out_path.resolve()}\n")
        pipeline.stop()
        return

    ov = None if args.no_world else OpenVocab()  # carica in background, il video parte subito

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 980, 620)

    cam_sources = [c[0] for c in cam_targets]
    cam_labels = {c[0]: c[3] for c in cam_targets}

    # Stato condiviso tra ciclo video, click e thread delle chiamate a Grok.
    # ponytail: dict senza lock, basta per una demo (una sola chiamata Grok alla volta).
    st = {
        "cam_sources": cam_sources,
        "cam_labels": cam_labels,
        "active_idx": 0,
        "active_source": cam_sources[0],
        "view_mode": "single", # 'single', 'grid', 'pip'
        "grok_objects": [],   # riquadri fermi di Grok (finché YOLO-World non segue quella classe)
        "objects": [],        # tutti gli oggetti dell'ultimo frame
        "frame_size": (640, 480),
        "meta": {},           # id -> {name, color, tags} dati da Grok con N
        "label_meta": load_vocab(VOCAB_PATH),  # label -> {name, color, tags, size_cm}, anche dalle sessioni prima
        "selected_id": None,
        "sel_last": None,     # ultimo dict dell'oggetto selezionato, per riagganciarlo se cambia id
        "sel_seen": 0.0,      # quando l'abbiamo visto l'ultima volta
        "sel_cmd": {},        # command/action/destination_id della scelta, riscritti se si riaggancia
        "last_cmd": None,     # ultimo comando S andato a buon fine, per il pannello
        "typing": None,       # testo del comando mentre lo scrivi (None = non stai scrivendo)
        "help": False,
        "is_busy": False,     # una chiamata a Grok è in corso
        "status": (f"Pronto. {len(cam_sources)} telecamere connesse. V/B = cambia cam, M = layout, G = trova.", "ok") if len(cam_sources) > 1 else ("Pronto. G = Grok trova gli oggetti, click = seleziona, H = aiuto.", "ok"),
        "latency_msg": "",
        "dog_info": "",
    }

    def set_status(msg: str, kind: str = "ok") -> None:
        st["status"] = (msg, kind)

    # Classi imparate nelle sessioni precedenti: YOLO-World le segue subito, senza premere G
    for label, m in st["label_meta"].items():
        _learn_size(label, m.get("size_cm"))
    if ov and st["label_meta"]:
        ov.set_classes(st["label_meta"])
        set_status(f"Ricordo {', '.join(st['label_meta'])}: le seguo dal vivo. C = dimentica tutto, H = aiuto.")

    def latency(res) -> str:
        return f"{res.meta.get('latency_ms', 0) / 1000:.1f} s" if res.meta else ""

    def reset_zoom() -> None:
        if zoom_mgr is not None:
            zoom_mgr.reset()
            set_status("Zoom resettato (1.0x)")

    def send(obj: dict) -> str:
        """Scrive la scelta in logs/selection.json per il braccio; restituisce il testo per la barra."""
        obj["distance_m"] = _distance(obj)
        obj["source_camera"] = st["active_source"]
        msg = send_selection(SELECTION_PATH, obj, **st["sel_cmd"])
        if msg is None:
            return "NON inviato al robot (logs/selection.json non scrivibile)"
        print("\n[ROBOT] logs/selection.json:\n" + json.dumps(msg, ensure_ascii=False, indent=2, default=str))
        return "inviato al robot"

    def select_obj(obj: dict, command: str | None = None, action: str | None = None,
                   destination_id: int | None = None) -> str:
        """Seleziona l'oggetto e lo manda al braccio. command/action ci sono solo se arriva da S."""
        st["selected_id"] = obj["id"]
        st["sel_last"], st["sel_seen"] = obj, time.time()
        st["sel_cmd"] = {"command": command, "action": action, "destination_id": destination_id}
        sent = send(obj)
        set_status(f"Selezionato #{obj['id']}: {display_obj(obj)} - distanza {format_distance(obj['distance_m'])}, "
                   f"{sent}  (Premi [Z] per zoomare)")
        return sent

    # Click sinistro = seleziona l'oggetto sotto il cursore, click destro = dezooma
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            return reset_zoom()
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        win_w, win_h = st["frame_size"]
        n_cams = len(st["cam_sources"])

        # Clic su miniatura PiP -> scambio telecamere immediato
        if n_cams > 1 and st["view_mode"] == "pip":
            other_sources = [s for s in st["cam_sources"] if s != st["active_source"]]
            pw, ph, pad = 180, 120, 8
            for k, o_src in enumerate(other_sources[:3]):
                y2 = win_h - 14 - k * (ph + pad)
                y1 = y2 - ph
                x1 = win_w - 14 - pw
                x2 = win_w - 14
                if x1 <= x <= x2 and y1 <= y <= y2:
                    st["active_idx"] = st["cam_sources"].index(o_src)
                    st["active_source"] = o_src
                    set_status(f"Telecamera attiva cambiata da PiP: {st['cam_labels'].get(o_src, o_src)}")
                    return

        # Clic in modalità griglia -> attiva la telecamera cliccata e seleziona l'oggetto
        if n_cams > 1 and st["view_mode"] == "grid":
            cell_idx, fx, fy = get_grid_cell(x, y, win_w, win_h, n_cams)
            if cell_idx < n_cams:
                clicked_src = st["cam_sources"][cell_idx]
                if st["active_source"] != clicked_src:
                    st["active_idx"] = cell_idx
                    st["active_source"] = clicked_src
                    set_status(f"Telecamera attiva: {st['cam_labels'].get(clicked_src, clicked_src)}")
                cam_objs = pipeline.state.get_objects(clicked_src)
                hit = object_at(cam_objs, fx, fy)
                if hit is not None:
                    select_obj(hit)
                return

        # Modalità singola
        fx, fy = x / max(1, win_w), y / max(1, win_h)
        cur_node = pipeline.nodes.get(st["active_source"])
        cur_zoom = cur_node.zoom if cur_node else None
        if cur_zoom is not None and cur_zoom.is_active():  # da coordinate dello zoom a quelle del frame intero
            fx, fy = cur_zoom.zoomed_to_full_coords(fx, fy)
        clicked_obj = object_at(st["objects"], fx, fy)
        if clicked_obj is not None:
            select_obj(clicked_obj)
        else:
            set_status("Nessun oggetto qui. Clicca su un oggetto o premi [S] per selezionarlo.", "wait")

    cv2.setMouseCallback(WIN, on_mouse_click)

    # Chiamate a Grok: ognuna in un thread, così il video non si ferma
    def run_grok_detect(raw_frame, yolo_objs):
        res = detect_objects(raw_frame, known=yolo_objs)
        if not res.ok:
            return set_status(f"Errore Grok: {res.error}", "err")
        for o in res.objects:
            _learn_size(o.label, o.size_cm)
            st["label_meta"].setdefault(o.label, {"name": o.name, "color": o.color, "tags": o.tags,
                                                  "size_cm": o.size_cm})
        save_vocab(VOCAB_PATH, st["label_meta"])
        st["grok_objects"] = [dict(o.model_dump(), distance_m=estimate_distance_m(o.label, o.bbox))
                              for o in res.objects]
        names = ", ".join(o.name for o in res.objects) or "nessuno"
        if ov and res.objects:
            ov.set_classes(o.label for o in res.objects)
            names += ". YOLO-World ora li segue dal vivo"
        dropped = res.meta.get("discard_reasons", [])
        if dropped:
            names += f" (scartati {len(dropped)} riquadri dubbi)"
            print("[Grok] scartati: " + "; ".join(dropped))
        set_status(f"Grok ha trovato {len(res.objects)} oggetti: {names}")
        st["latency_msg"] = latency(res)

    def run_describe(raw_img, objs):
        res = describe_scene(raw_img, objs)
        set_status(res.text) if res.ok else set_status(f"Errore: {res.error}", "err")
        st["latency_msg"] = latency(res)

    def run_names(marked_img, objs):
        res = name_objects(marked_img, objs)
        if not res.ok:
            return set_status(f"Errore: {res.error}", "err")
        by_id = {o["id"]: o for o in objs}
        for n in res.objects:
            meta = {"name": n.name, "color": n.color, "tags": n.tags}
            st["meta"][n.id] = meta
            if n.id >= 2000:  # gli id di YOLO-World cambiano quando perde l'oggetto: il nome si ricorda per classe
                lbl = by_id[n.id]["label"]
                st["label_meta"][lbl] = {**st["label_meta"].get(lbl, {}), **meta}  # size_cm resta
            if arm_node and n.id < 1000:
                arm_node.tracker.set_object_metadata(n.id, name=n.name, color=n.color)
            for go in st["grok_objects"]:
                if go["id"] == n.id:
                    go.update(meta)
        save_vocab(VOCAB_PATH, st["label_meta"])
        set_status("Nomi: " + (", ".join(f"#{n.id} {n.name}" for n in res.objects) or "nessuno"))
        st["latency_msg"] = latency(res)

    def run_select(marked_img, objs, text):
        res = select(marked_img, objs, text)
        st["latency_msg"] = f"{latency(res)} ({res.source})"
        if not res.ok:
            return set_status(f"Errore: {res.error}", "err")
        if res.ambiguous:
            return set_status(f"Quale? {res.question}", "wait")
        hit = next((o for o in objs if o["id"] == res.target_id), None)
        if hit is None:
            return set_status(f"Non trovato: {res.reason}", "err")
        st["last_cmd"] = {"target_id": res.target_id, "action": res.action, "source": res.source}
        sent = select_obj(hit, text, res.action, res.destination_id)
        set_status(f"'{text}' -> #{hit['id']} {display_obj(hit)}, azione: {res.action}, {sent}")

    def start(job, *job_args, msg):
        st["is_busy"] = True
        set_status(msg, "wait")

        def run():
            try:
                job(*job_args)
            finally:
                st["is_busy"] = False

        threading.Thread(target=run, daemon=True).start()

    print("\n" + "=" * 65)
    print("  DEMO LIVE: OCCHI (Parte A) + CERVELLO (Parte B)   -   H nella finestra = aiuto")
    print("=" * 65 + "\n")

    t_prev = time.perf_counter()
    fps_smooth = 30.0
    win_shown = False  # la finestra può risultare "non visibile" nei primi frame: non è la X

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.90 * fps_smooth + 0.10 * (1.0 / dt)

            active_src = st["active_source"]
            cur_node = pipeline.nodes.get(active_src)
            zoom_mgr = cur_node.zoom if cur_node else None

            # Acquisisce e processa la telecamera attiva
            pipeline.step(active_src, min_display_conf=min_conf, show_conf=show_conf, apply_zoom=False)
            raw = pipeline.state.get_frame(active_src)
            if raw is None or raw.size == 0:
                time.sleep(0.01)
                continue
            st["frame_size"] = (raw.shape[1], raw.shape[0])
            yolo_objs = pipeline.state.get_objects(active_src)

            # Se in griglia o PiP, acquisisce anche le altre telecamere per tenerle vive
            n_cams = len(st["cam_sources"])
            if n_cams > 1 and st["view_mode"] in ("grid", "pip"):
                for src in st["cam_sources"]:
                    if src != active_src and src in pipeline.nodes:
                        pipeline.step(src, min_display_conf=min_conf, show_conf=show_conf, apply_zoom=False)

            # YOLO-World: oggetti vivi delle classi trovate da Grok, con i nomi dati da Grok
            live = []
            if ov:
                ov.feed(raw)
                live = [dict(o) for o in ov.objects]
                seen = {o["label"] for o in live}
                if seen:  # quella classe ora si muove dal vivo: via il riquadro fermo di Grok
                    st["grok_objects"] = [o for o in st["grok_objects"] if o["label"] not in seen]
                for o in live:
                    o.update(st["label_meta"].get(o["label"], {}))
            all_objs = yolo_objs + live + st["grok_objects"]
            for o in all_objs:
                o.update(st["meta"].get(o["id"], {}))
            st["objects"] = all_objs

            frame_display = draw_marks(raw, all_objs, min_conf=min_conf, show_conf=show_conf)

            sel = next((o for o in all_objs if o["id"] == st["selected_id"]), None)
            # Selezionato sparito: forse il tracker gli ha dato un altro id. Lo cerchiamo per 3 s
            if sel is None and st["sel_last"] and time.time() - st["sel_seen"] < 3.0:
                sel = reacquire(all_objs, st["sel_last"])
                if sel is not None:
                    old = st["selected_id"]
                    st["selected_id"] = sel["id"]
                    if st["last_cmd"] and st["last_cmd"]["target_id"] == old:
                        st["last_cmd"]["target_id"] = sel["id"]
                    if zoom_mgr is not None and zoom_mgr.target_id == old:
                        zoom_mgr.set_target_object(sel["id"])
                    set_status(f"#{sel['id']} riagganciato (prima #{old}): {display_obj(sel)}, {send(sel)}")
            # Oggetto selezionato: sagoma, oppure riquadro bianco se la sagoma non si trova
            if sel is not None:
                st["sel_last"], st["sel_seen"] = sel, time.time()
                fit = refine_box(raw, sel["bbox"])
                if fit is not None:
                    cv2.drawContours(frame_display, [fit[1]], -1, (255, 255, 255), 4, cv2.LINE_AA)
                    cv2.drawContours(frame_display, [fit[1]], -1, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    h_f, w_f = frame_display.shape[:2]
                    bx, by, bw, bh = sel["bbox"]
                    cv2.rectangle(frame_display, (int(bx * w_f) - 3, int(by * h_f) - 3),
                                  (int((bx + bw) * w_f) + 3, int((by + bh) * h_f) + 3), (255, 255, 255), 3)

            if zoom_mgr is not None and zoom_mgr.is_active():
                zoom_mgr.update_with_tracked_objects(all_objs)
                frame_display = zoom_mgr.apply(frame_display)

            # Modalità visualizzazione multi-camera (Grid o PiP)
            if n_cams > 1:
                if st["view_mode"] == "grid":
                    frames_dict = {}
                    for s in st["cam_sources"]:
                        if s == active_src:
                            frames_dict[s] = frame_display
                        else:
                            mf = pipeline.state.get_marked_frame(s)
                            frames_dict[s] = mf if (mf is not None and mf.size > 0) else raw
                    frame_display = _compose_grid(frame_display, frames_dict, st["cam_sources"], st["cam_labels"], active_src)
                elif st["view_mode"] == "pip":
                    other_sources = [s for s in st["cam_sources"] if s != active_src]
                    fh, fw = frame_display.shape[:2]
                    pw, ph, pad = 180, 120, 8
                    for k, o_src in enumerate(other_sources[:3]):
                        o_frame = pipeline.state.get_marked_frame(o_src)
                        if o_frame is not None and o_frame.size > 0:
                            mini = cv2.resize(o_frame, (pw, ph))
                            cv2.rectangle(mini, (0, 0), (pw - 1, ph - 1), (0, 230, 255), 2)
                            _box(mini, 0, 0, pw, 22, 0.75)
                            lbl = st["cam_labels"].get(o_src, o_src.upper())
                            _text(mini, f"{lbl} (clic)", 6, 15, (0, 255, 255), 0.35, 1)
                            y2 = fh - 14 - k * (ph + pad)
                            y1 = y2 - ph
                            x1 = fw - 14 - pw
                            x2 = fw - 14
                            if y1 >= 50:
                                frame_display[y1:y2, x1:x2] = mini

            # Interfaccia sopra il video
            _draw_top(frame_display, fps_smooth, sum(1 for o in all_objs if o.get("conf", 0) >= min_conf),
                      zoom_mgr.current_factor if zoom_mgr else 1.0, ov.status if ov else "spento",
                      active_cam_idx=st["active_idx"], n_cams=n_cams,
                      active_label=st["cam_labels"].get(active_src, active_src),
                      view_mode=st["view_mode"])
            msg, kind = st["status"]
            _draw_status(frame_display, ("[Grok lavora...] " if st["is_busy"] else "") + msg,
                         "wait" if st["is_busy"] else kind, st["latency_msg"])
            if st["selected_id"] is not None:
                _draw_panel(frame_display, sel, st["selected_id"], st["last_cmd"])
            if st["dog_info"]:
                _text(frame_display, st["dog_info"], 10, 64, (0, 165, 255), 0.5, 2)
            if st["typing"] is not None:
                _draw_input(frame_display, st["typing"])
            elif st["help"]:
                _draw_help(frame_display)

            cv2.imshow(WIN, frame_display)
            key = cv2.waitKey(15) & 0xFF
            visible = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1
            if win_shown and not visible:  # chiusa con la X
                break
            win_shown = win_shown or visible
            if key == 255:
                continue

            # Mentre scrivi un comando, i tasti vanno nel testo
            if st["typing"] is not None:
                if key == 27:
                    st["typing"] = None
                    set_status("Comando annullato")
                elif key in (13, 10):
                    text, st["typing"] = st["typing"].strip(), None
                    if text and not st["is_busy"]:
                        marked = draw_marks(raw, all_objs)
                        start(run_select, marked, list(all_objs), text, msg=f"Cerco: '{text}'...")
                elif key == 8:
                    st["typing"] = st["typing"][:-1]
                elif 32 <= key < 127:
                    st["typing"] += chr(key)
                continue

            if key in (27, ord("q"), ord("Q")):
                break

            elif key in (ord("h"), ord("H")):
                st["help"] = not st["help"]

            elif key in (ord("v"), ord("V"), 9):  # V o TAB: telecamera successiva
                if len(st["cam_sources"]) > 1:
                    st["active_idx"] = (st["active_idx"] + 1) % len(st["cam_sources"])
                    st["active_source"] = st["cam_sources"][st["active_idx"]]
                    set_status(f"Telecamera attiva: {st['cam_labels'].get(st['active_source'], st['active_source'])} [{st['active_idx']+1}/{len(st['cam_sources'])}]")
                else:
                    set_status("1 sola telecamera connessa (nessun'altra cam da selezionare)", "wait")

            elif key in (ord("b"), ord("B")):  # B: telecamera precedente
                if len(st["cam_sources"]) > 1:
                    st["active_idx"] = (st["active_idx"] - 1) % len(st["cam_sources"])
                    st["active_source"] = st["cam_sources"][st["active_idx"]]
                    set_status(f"Telecamera attiva: {st['cam_labels'].get(st['active_source'], st['active_source'])} [{st['active_idx']+1}/{len(st['cam_sources'])}]")
                else:
                    set_status("1 sola telecamera connessa", "wait")

            elif key in (ord("m"), ord("M")):  # M: cambia modalità layout
                if len(st["cam_sources"]) > 1:
                    modes = ["single", "grid", "pip"]
                    cur_idx = modes.index(st["view_mode"])
                    st["view_mode"] = modes[(cur_idx + 1) % len(modes)]
                    lbls = {"single": "Singola (tutto schermo)", "grid": "Griglia multi-camera", "pip": "Picture-in-Picture (PiP)"}
                    set_status(f"Layout telecamere: {lbls[st['view_mode']]}")
                else:
                    set_status("Layout multi-cam richiede 2 o più telecamere", "wait")

            elif key in (ord("g"), ord("G")) and not st["is_busy"]:
                start(run_grok_detect, raw.copy(), list(yolo_objs), msg="Grok cerca gli oggetti nella scena...")

            elif key in (ord("d"), ord("D"), 32) and not st["is_busy"]:
                start(run_describe, raw.copy(), list(all_objs), msg="Grok descrive la scena...")

            # N e S mandano a Grok il frame con i soli numeri, senza HUD né zoom
            elif key in (ord("n"), ord("N")) and not st["is_busy"]:
                start(run_names, draw_marks(raw, all_objs), list(all_objs), msg="Grok da' i nomi agli oggetti...")

            elif key in (ord("s"), ord("S")):
                st["typing"], st["help"] = "", False

            # C: dimentica gli oggetti di Grok, le classi di YOLO-World e vocab.json
            elif key in (ord("c"), ord("C")):
                st["grok_objects"], st["label_meta"] = [], {}
                if ov:
                    ov.forget()
                save_vocab(VOCAB_PATH, {})
                set_status("Dimenticato tutto: oggetti di Grok e classi imparate. G = ricomincia.")

            elif key in (ord("k"), ord("K")):
                show_conf = not show_conf
                set_status(f"Percentuale di confidenza: {'visibile' if show_conf else 'nascosta'}")

            # + e - anche con = _ ] [ (stesso tasto senza Shift su layout diversi)
            elif key in (ord("+"), ord("="), ord("]")):
                min_conf = min(0.95, round(min_conf + 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            elif key in (ord("-"), ord("_"), ord("[")):
                min_conf = max(0.05, round(min_conf - 0.05, 2))
                set_status(f"Soglia di confidenza: {int(min_conf * 100)}%")

            elif key in (ord("z"), ord("Z")):
                target_obj = sel or st["sel_last"]
                if target_obj is not None and zoom_mgr is not None:
                    zoom_mgr.set_target_object(target_obj["id"])
                    zoom_mgr.zoom_to_bbox(target_obj["bbox"])
                    set_status(
                        f"Zoom su #{target_obj['id']} ({display_obj(target_obj)}). "
                        f"Click destro del mouse (o 0/R) per dezoomare."
                    )
                else:
                    set_status("Seleziona prima un oggetto (con S o Click sinistro) e poi premi Z per zoomare.", "wait")

            elif ord("1") <= key <= ord("9"):
                target_num = key - ord("0")
                if zoom_mgr is not None:
                    matched = [o for o in all_objs if o.get("id") == target_num]
                    if matched:
                        select_obj(matched[0])
                        zoom_mgr.set_target_object(target_num)
                        zoom_mgr.zoom_to_bbox(matched[0]["bbox"])
                        set_status(f"Zoom su #{target_num} ({display_obj(matched[0])}). Click destro = dezooma.")
                    else:
                        set_status(f"Oggetto #{target_num} non visibile", "wait")

            elif key in (ord("0"), ord("r"), ord("R")):
                reset_zoom()

            elif key in (ord("t"), ord("T")):
                target = sel or (all_objs[0] if all_objs else None)
                if target:
                    st["dog_info"] = f"Target cane: #{target['id']}"
                    set_status(f"Target cane impostato su #{target['id']} ({display_obj(target)}).")
                else:
                    set_status("Nessun oggetto da agganciare.", "err")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
